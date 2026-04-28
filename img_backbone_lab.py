"""
Foundation Model Feature Visualization
- CLIP, DINOv2, SAM, Grounded SAM 네 모델의 feature 추출
- Heatmap (이미지 위 overlay) + Distribution (t-SNE) 모두 시각화
- test/ 폴더 안의 모든 이미지 처리
- 모델 로딩 / Warmup / 순수 추론 시간 분리 측정

설치:
  pip install open_clip_torch matplotlib scikit-learn pillow opencv-python
  pip install git+https://github.com/facebookresearch/segment-anything.git
  pip install git+https://github.com/IDEA-Research/GroundingDINO.git
  pip install xformers   # (선택) DINOv2 가속

  # SAM 체크포인트 다운로드:
  wget -P ckpt/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

  # GroundingDINO 체크포인트 다운로드:
  wget -P ckpt/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
"""

import os
import glob
import math
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torchvision.transforms as T
import cv2

# ──────────────────────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_DIR = './test'
OUTPUT_DIR = './feature_viz'
DINO_IMG_SIZE = 518          # 14의 배수 (518 = 14 * 37 patches)
SAM_CHECKPOINT = 'ckpt/sam_vit_b_01ec64.pth'
GDINO_CHECKPOINT = 'ckpt/groundingdino_swint_ogc.pth'
GDINO_BOX_THRESHOLD = 0.35
GDINO_TEXT_THRESHOLD = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('ckpt', exist_ok=True)

# CLIP 텍스트 쿼리 (자율주행 시나리오)
TEXT_QUERIES = [
    "a road",
    "a vehicle",
    "a pedestrian",
    "a building",
    "vegetation",
    "the sky",
    "a traffic sign",
    "an indoor scene",
]


# ──────────────────────────────────────────────────────────
# 시간 측정 헬퍼
# ──────────────────────────────────────────────────────────
def sync():
    """GPU 연산 동기화 (정확한 시간 측정용)"""
    if DEVICE == 'cuda':
        torch.cuda.synchronize()


class Timer:
    """시간 측정 컨텍스트 매니저"""
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        sync()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        sync()
        self.elapsed = time.time() - self.start


# ──────────────────────────────────────────────────────────
# 1. 이미지 로드 유틸
# ──────────────────────────────────────────────────────────
def load_image_paths(folder):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, ext.upper())))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    print(f"Found {len(paths)} images in {folder}")
    return paths


def overlay_heatmap(img_pil, heatmap, alpha=0.5, cmap='jet'):
    """이미지 위에 heatmap을 컬러로 overlay"""
    img = np.array(img_pil.convert('RGB'))
    h, w = img.shape[:2]

    hm = cv2.resize(heatmap.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)

    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    cm = plt.get_cmap(cmap)
    hm_color = (cm(hm_norm)[..., :3] * 255).astype(np.uint8)

    blended = (img * (1 - alpha) + hm_color * alpha).astype(np.uint8)
    return blended


# ──────────────────────────────────────────────────────────
# 2. DINOv2 - patch feature → PCA RGB + attention heatmap
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def run_dinov2(image_paths):
    print("\n[DINOv2]")
    timings = {}

    # ── 모델 로딩 ────────────────────────────────────────
    with Timer() as t:
        dinov2 = torch.hub.load('facebookresearch/dinov2',
                                 'dinov2_vitb14').to(DEVICE).eval()
    timings['load'] = t.elapsed
    print(f"  Load:    {t.elapsed:.2f}s")

    transform = T.Compose([
        T.Resize((DINO_IMG_SIZE, DINO_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # ── Warmup (첫 forward는 CUDA 커널 컴파일로 느림) ────
    with Timer() as t:
        if DEVICE == 'cuda':
            dummy = torch.randn(1, 3, DINO_IMG_SIZE, DINO_IMG_SIZE).to(DEVICE)
            _ = dinov2.forward_features(dummy)
    timings['warmup'] = t.elapsed
    print(f"  Warmup:  {t.elapsed:.2f}s")

    cls_features = []
    patch_features_list = []
    per_image_inference = []     # 순수 forward 시간만
    per_image_total = []          # I/O + transform + forward

    # ── 이미지별 처리 ─────────────────────────────────────
    for path in image_paths:
        # 전체 (I/O 포함)
        sync(); t_total_start = time.time()

        img = Image.open(path).convert('RGB')
        x = transform(img).unsqueeze(0).to(DEVICE)

        # 순수 추론만
        sync(); t_inf_start = time.time()
        out = dinov2.forward_features(x)
        sync(); t_inf = time.time() - t_inf_start

        patch_feat = out['x_norm_patchtokens'][0].cpu().numpy()
        cls_feat = out['x_norm_clstoken'][0].cpu().numpy()
        cls_features.append(cls_feat)
        patch_features_list.append(patch_feat)

        sync(); t_total = time.time() - t_total_start

        per_image_inference.append(t_inf)
        per_image_total.append(t_total)

    n = len(image_paths)
    timings['inference_per_img'] = np.mean(per_image_inference)
    timings['inference_total'] = np.sum(per_image_inference)
    timings['per_image_total'] = np.mean(per_image_total)
    timings['total_pipeline'] = np.sum(per_image_total)

    print(f"  Inference (pure):  {np.mean(per_image_inference)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_image_inference):.2f}s)")
    print(f"  Inference (+I/O):  {np.mean(per_image_total)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_image_total):.2f}s)")
    print(f"  Per-image breakdown: "
          f"{[f'{t*1000:.0f}ms' for t in per_image_inference]}")

    return np.array(cls_features), patch_features_list, timings


def viz_dinov2_heatmap(image_paths, patch_features_list):
    """DINOv2 patch feature를 PCA RGB + attention overlay로 시각화"""
    n = len(image_paths)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (path, feat) in enumerate(zip(image_paths, patch_features_list)):
        img = Image.open(path).convert('RGB')

        n_patches = feat.shape[0]
        gs = int(np.sqrt(n_patches))

        # PCA → 3D (RGB)
        pca = PCA(n_components=3)
        feat_3d = pca.fit_transform(feat)
        feat_3d = (feat_3d - feat_3d.min(0)) / (feat_3d.max(0) - feat_3d.min(0) + 1e-8)
        rgb_grid = feat_3d.reshape(gs, gs, 3)

        fg_map = feat_3d[:, 0].reshape(gs, gs)

        axes[i, 0].imshow(img.resize((DINO_IMG_SIZE, DINO_IMG_SIZE)))
        axes[i, 0].set_title(f'Original\n{os.path.basename(path)}', fontsize=9)
        axes[i, 0].axis('off')

        rgb_resized = cv2.resize(rgb_grid, (DINO_IMG_SIZE, DINO_IMG_SIZE),
                                  interpolation=cv2.INTER_CUBIC)
        rgb_resized = np.clip(rgb_resized, 0, 1)   # 보간으로 살짝 벗어난 값 클립
        axes[i, 1].imshow(rgb_resized)
        axes[i, 1].set_title('DINOv2 PCA-RGB\n(patch features)', fontsize=9)
        axes[i, 1].axis('off')

        overlay = overlay_heatmap(img.resize((DINO_IMG_SIZE, DINO_IMG_SIZE)),
                                   fg_map, alpha=0.5, cmap='jet')
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Foreground Attention\n(1st PC overlay)', fontsize=9)
        axes[i, 2].axis('off')

        center_idx = (gs // 2) * gs + (gs // 2)
        sim_norm = F.normalize(torch.tensor(feat), dim=-1).numpy()
        center_norm = sim_norm[center_idx:center_idx+1]
        cos_sim = (sim_norm @ center_norm.T).flatten().reshape(gs, gs)

        sim_overlay = overlay_heatmap(img.resize((DINO_IMG_SIZE, DINO_IMG_SIZE)),
                                       cos_sim, alpha=0.5, cmap='viridis')
        axes[i, 3].imshow(sim_overlay)
        axes[i, 3].set_title('Self-Similarity\n(from center patch)', fontsize=9)
        axes[i, 3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'dinov2_heatmap.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# 3. CLIP - 이미지 feature + 텍스트 유사도
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def run_clip(image_paths):
    print("\n[CLIP]")
    timings = {}

    try:
        import open_clip
    except ImportError:
        print("  Install: pip install open_clip_torch")
        return None, None, None

    # ── 모델 로딩 ────────────────────────────────────────
    with Timer() as t:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        model = model.to(DEVICE).eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
    timings['load'] = t.elapsed
    print(f"  Load:    {t.elapsed:.2f}s")

    # ── 텍스트 인코딩 (1회) ───────────────────────────────
    with Timer() as t:
        text_tokens = tokenizer(TEXT_QUERIES).to(DEVICE)
        text_features = F.normalize(model.encode_text(text_tokens), dim=-1)
    timings['text_encode'] = t.elapsed
    print(f"  Text encoding ({len(TEXT_QUERIES)} queries): {t.elapsed*1000:.1f} ms")

    # ── Warmup ────────────────────────────────────────────
    with Timer() as t:
        if DEVICE == 'cuda':
            dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            _ = model.encode_image(dummy)
    timings['warmup'] = t.elapsed
    print(f"  Warmup:  {t.elapsed:.2f}s")

    image_features = []
    similarities = []
    per_image_inference = []
    per_image_total = []

    for path in image_paths:
        sync(); t_total_start = time.time()

        img = Image.open(path).convert('RGB')
        x = preprocess(img).unsqueeze(0).to(DEVICE)

        sync(); t_inf_start = time.time()
        feat = F.normalize(model.encode_image(x), dim=-1)
        sim = (feat @ text_features.T).softmax(dim=-1)
        sync(); t_inf = time.time() - t_inf_start

        image_features.append(feat[0].cpu().numpy())
        similarities.append(sim[0].cpu().numpy())

        sync(); t_total = time.time() - t_total_start

        per_image_inference.append(t_inf)
        per_image_total.append(t_total)

    timings['inference_per_img'] = np.mean(per_image_inference)
    timings['inference_total'] = np.sum(per_image_inference)
    timings['per_image_total'] = np.mean(per_image_total)
    timings['total_pipeline'] = np.sum(per_image_total)

    print(f"  Inference (pure):  {np.mean(per_image_inference)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_image_inference):.2f}s)")
    print(f"  Inference (+I/O):  {np.mean(per_image_total)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_image_total):.2f}s)")
    print(f"  Per-image breakdown: "
          f"{[f'{t*1000:.0f}ms' for t in per_image_inference]}")

    return np.array(image_features), similarities, timings


def viz_clip(image_paths, similarities):
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (path, sim) in enumerate(zip(image_paths, similarities)):
        img = Image.open(path).convert('RGB')

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'{os.path.basename(path)}', fontsize=10)
        axes[i, 0].axis('off')

        colors = plt.cm.viridis(sim / (sim.max() + 1e-8))
        axes[i, 1].barh(range(len(TEXT_QUERIES)), sim, color=colors)
        axes[i, 1].set_yticks(range(len(TEXT_QUERIES)))
        axes[i, 1].set_yticklabels(TEXT_QUERIES, fontsize=9)
        axes[i, 1].set_xlabel('Similarity (softmax)')
        axes[i, 1].set_title('CLIP Text-Image Similarity', fontsize=10)
        axes[i, 1].invert_yaxis()
        top1 = sim.argmax()
        axes[i, 1].get_yticklabels()[top1].set_fontweight('bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'clip_heatmap.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# 4. SAM - automatic mask generation
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def run_sam(image_paths):
    print("\n[SAM]")
    timings = {}

    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        from segment_anything import SamPredictor
    except ImportError:
        print("  Install: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return None, None, None

    if not os.path.exists(SAM_CHECKPOINT):
        print(f"  Checkpoint not found: {SAM_CHECKPOINT}")
        print("  wget -P ckpt/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return None, None, None

    # ── 모델 로딩 ────────────────────────────────────────
    with Timer() as t:
        sam = sam_model_registry['vit_b'](checkpoint=SAM_CHECKPOINT).to(DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)
        predictor = SamPredictor(sam)
    timings['load'] = t.elapsed
    print(f"  Load:    {t.elapsed:.2f}s")

    all_masks = []
    image_embeddings = []

    # 단계별 시간 누적
    per_img_set_image = []          # set_image (encoder forward)
    per_img_mask_gen = []            # automatic mask generation
    per_img_total = []

    for path in image_paths:
        sync(); t_total_start = time.time()

        img_np = np.array(Image.open(path).convert('RGB'))

        # 1) automatic mask generation (가장 무거움)
        sync(); t_mg_start = time.time()
        masks = mask_generator.generate(img_np)
        sync(); t_mg = time.time() - t_mg_start

        # 2) image embedding (set_image = encoder forward)
        sync(); t_se_start = time.time()
        predictor.set_image(img_np)
        embed = predictor.get_image_embedding()
        sync(); t_se = time.time() - t_se_start

        all_masks.append(masks)
        global_feat = embed.mean(dim=(2, 3))[0].cpu().numpy()
        image_embeddings.append(global_feat)

        sync(); t_total = time.time() - t_total_start

        per_img_set_image.append(t_se)
        per_img_mask_gen.append(t_mg)
        per_img_total.append(t_total)

    timings['set_image_per_img'] = np.mean(per_img_set_image)
    timings['mask_gen_per_img'] = np.mean(per_img_mask_gen)
    timings['inference_per_img'] = np.mean(per_img_set_image) + np.mean(per_img_mask_gen)
    timings['inference_total'] = np.sum(per_img_set_image) + np.sum(per_img_mask_gen)
    timings['per_image_total'] = np.mean(per_img_total)
    timings['total_pipeline'] = np.sum(per_img_total)

    print(f"  set_image (encoder):  {np.mean(per_img_set_image)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_img_set_image):.2f}s)")
    print(f"  mask_gen (auto):      {np.mean(per_img_mask_gen)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_img_mask_gen):.2f}s)")
    print(f"  Inference (+I/O):     {np.mean(per_img_total)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_img_total):.2f}s)")
    print(f"  Per-image breakdown:  "
          f"{[f'{t*1000:.0f}ms' for t in per_img_total]}")

    return np.array(image_embeddings), all_masks, timings


def viz_sam(image_paths, all_masks):
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = axes[None, :]

    np.random.seed(42)
    for i, (path, masks) in enumerate(zip(image_paths, all_masks)):
        img = np.array(Image.open(path).convert('RGB'))

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original\n{os.path.basename(path)}', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img)
        if len(masks) > 0:
            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            overlay = np.zeros((*img.shape[:2], 4))
            for m in sorted_masks:
                color = np.concatenate([np.random.rand(3), [0.5]])
                overlay[m['segmentation']] = color
            axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f'SAM Masks ({len(masks)} regions)', fontsize=10)
        axes[i, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'sam_heatmap.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# 5. Grounding DINO - open-set detection (backbone feature)
# ──────────────────────────────────────────────────────────
def _find_gdino_config():
    try:
        import groundingdino
        cfg = os.path.join(os.path.dirname(groundingdino.__file__),
                           'config', 'GroundingDINO_SwinT_OGC.py')
        if os.path.exists(cfg):
            return cfg
    except Exception:
        pass
    return 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'


@torch.no_grad()
def run_grounding_dino(image_paths):
    print("\n[Grounding DINO]")
    timings = {}

    try:
        from groundingdino.util.inference import load_model, load_image, predict
    except ImportError:
        print("  Install: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
        return None, None, None

    gdino_config = _find_gdino_config()
    if not os.path.exists(gdino_config):
        print(f"  GroundingDINO config not found: {gdino_config}")
        return None, None, None
    if not os.path.exists(GDINO_CHECKPOINT):
        print(f"  Checkpoint not found: {GDINO_CHECKPOINT}")
        print("  wget -P ckpt/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        return None, None, None

    # ── 모델 로딩 ────────────────────────────────────────
    with Timer() as t:
        gdino = load_model(gdino_config, GDINO_CHECKPOINT)
        gdino = gdino.to(DEVICE).eval()
    timings['load'] = t.elapsed
    print(f"  Load:    {t.elapsed:.2f}s")

    # Swin-T backbone의 마지막 스테이지 feature를 캡처하는 hook
    _backbone_feats = {}

    def _hook(module, inp, out):
        try:
            feats = out[0] if isinstance(out, (list, tuple)) else out
            last = feats[-1] if isinstance(feats, (list, tuple)) else feats
            t = last.tensors if hasattr(last, 'tensors') else last
            if t.dim() == 4:
                _backbone_feats['feat'] = t.mean(dim=(2, 3))   # (B, C)
            elif t.dim() == 3:
                _backbone_feats['feat'] = t.mean(dim=1)         # (B, C)
        except Exception:
            pass

    hook_handle = gdino.backbone.register_forward_hook(_hook)

    text_prompt = ' . '.join(TEXT_QUERIES)
    all_results = []
    image_features = []
    per_image_inference = []
    per_image_total = []

    for path in image_paths:
        sync(); t_total_start = time.time()

        _, image_tensor = load_image(path)
        img_np = np.array(Image.open(path).convert('RGB'))
        h, w = img_np.shape[:2]

        sync(); t_inf_start = time.time()
        boxes, logits, phrases = predict(
            model=gdino,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=GDINO_BOX_THRESHOLD,
            text_threshold=GDINO_TEXT_THRESHOLD,
            device=DEVICE,
        )
        sync(); t_inf = time.time() - t_inf_start

        feat = _backbone_feats.get('feat')
        if feat is not None:
            image_features.append(feat[0].cpu().numpy())

        all_results.append({
            'boxes': boxes.cpu().numpy() if len(boxes) > 0 else np.array([]),
            'labels': phrases,
            'scores': logits.cpu().numpy() if len(logits) > 0 else np.array([]),
        })

        sync(); t_total = time.time() - t_total_start
        per_image_inference.append(t_inf)
        per_image_total.append(t_total)

    hook_handle.remove()

    timings['inference_per_img'] = np.mean(per_image_inference)
    timings['inference_total'] = np.sum(per_image_inference)
    timings['per_image_total'] = np.mean(per_image_total)
    timings['total_pipeline'] = np.sum(per_image_total)

    print(f"  Inference (pure):  {np.mean(per_image_inference)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_image_inference):.2f}s)")
    print(f"  Inference (+I/O):  {np.mean(per_image_total)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_image_total):.2f}s)")
    print(f"  Per-image breakdown: "
          f"{[f'{t*1000:.0f}ms' for t in per_image_inference]}")

    feats_arr = np.array(image_features) if image_features else None
    return feats_arr, all_results, timings


def viz_grounding_dino(image_paths, all_results):
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (path, result) in enumerate(zip(image_paths, all_results)):
        img = np.array(Image.open(path).convert('RGB'))
        h, w = img.shape[:2]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original\n{os.path.basename(path)}', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img)
        for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
            cx, cy, bw, bh = box
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            rect = plt.Rectangle(
                (x1, y1), bw * w, bh * h,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            axes[i, 1].add_patch(rect)
            axes[i, 1].text(
                x1, y1 - 4, f'{label} {score:.2f}',
                color='cyan', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5)
            )

        n_det = len(result['labels'])
        axes[i, 1].set_title(f'Grounding DINO ({n_det} detections)', fontsize=10)
        axes[i, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'grounding_dino_heatmap.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# 6. Grounded SAM - text-prompted detection + segmentation
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def run_grounded_sam(image_paths):
    print("\n[Grounded SAM]")
    timings = {}

    try:
        from groundingdino.util.inference import load_model, load_image, predict
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("  Install: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
        print("           pip install git+https://github.com/facebookresearch/segment-anything.git")
        return None, None, None

    gdino_config = _find_gdino_config()
    if not os.path.exists(gdino_config):
        print(f"  GroundingDINO config not found: {gdino_config}")
        return None, None, None
    if not os.path.exists(GDINO_CHECKPOINT):
        print(f"  Checkpoint not found: {GDINO_CHECKPOINT}")
        print("  wget -P ckpt/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        return None, None, None
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"  SAM checkpoint not found: {SAM_CHECKPOINT}")
        print("  wget -P ckpt/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return None, None, None

    # ── 모델 로딩 ────────────────────────────────────────
    with Timer() as t:
        gdino = load_model(gdino_config, GDINO_CHECKPOINT)
        gdino = gdino.to(DEVICE).eval()
        sam = sam_model_registry['vit_b'](checkpoint=SAM_CHECKPOINT).to(DEVICE)
        predictor = SamPredictor(sam)
    timings['load'] = t.elapsed
    print(f"  Load:    {t.elapsed:.2f}s")

    # TEXT_QUERIES를 Grounding DINO caption 형식으로 변환
    text_prompt = ' . '.join(TEXT_QUERIES)

    all_results = []
    image_features = []
    per_image_inference = []
    per_image_total = []

    for path in image_paths:
        sync(); t_total_start = time.time()

        image_source, image_tensor = load_image(path)
        img_np = np.array(Image.open(path).convert('RGB'))
        h, w = img_np.shape[:2]

        sync(); t_inf_start = time.time()

        # 1) Grounding DINO: text-prompted box detection
        boxes, logits, phrases = predict(
            model=gdino,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=GDINO_BOX_THRESHOLD,
            text_threshold=GDINO_TEXT_THRESHOLD,
            device=DEVICE,
        )

        # 2) SAM: box-prompted segmentation
        predictor.set_image(img_np)
        masks_list = np.array([])
        if len(boxes) > 0:
            # normalized cx,cy,w,h → pixel x1,y1,x2,y2
            boxes_xyxy = boxes.clone()
            boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
            boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
            boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
            boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h

            transformed_boxes = predictor.transform.apply_boxes_torch(
                boxes_xyxy.to(DEVICE), img_np.shape[:2]
            )
            masks_tensor, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks_list = masks_tensor[:, 0].cpu().numpy()  # (N, H, W)

        embed = predictor.get_image_embedding()
        global_feat = embed.mean(dim=(2, 3))[0].cpu().numpy()

        sync(); t_inf = time.time() - t_inf_start

        all_results.append({
            'boxes': boxes.cpu().numpy() if len(boxes) > 0 else np.array([]),
            'labels': phrases,
            'scores': logits.cpu().numpy() if len(logits) > 0 else np.array([]),
            'masks': masks_list,
        })
        image_features.append(global_feat)

        sync(); t_total = time.time() - t_total_start
        per_image_inference.append(t_inf)
        per_image_total.append(t_total)

    timings['inference_per_img'] = np.mean(per_image_inference)
    timings['inference_total'] = np.sum(per_image_inference)
    timings['per_image_total'] = np.mean(per_image_total)
    timings['total_pipeline'] = np.sum(per_image_total)

    print(f"  Inference (pure):  {np.mean(per_image_inference)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_image_inference):.2f}s)")
    print(f"  Inference (+I/O):  {np.mean(per_image_total)*1000:>7.1f} ms/img"
          f"  (total {np.sum(per_image_total):.2f}s)")
    print(f"  Per-image breakdown: "
          f"{[f'{t*1000:.0f}ms' for t in per_image_inference]}")

    return np.array(image_features), all_results, timings


def viz_grounded_sam(image_paths, all_results):
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = axes[None, :]

    np.random.seed(42)
    for i, (path, result) in enumerate(zip(image_paths, all_results)):
        img = np.array(Image.open(path).convert('RGB'))
        h, w = img.shape[:2]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original\n{os.path.basename(path)}', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img)

        boxes = result['boxes']
        labels = result['labels']
        scores = result['scores']
        masks = result['masks']

        # 마스크 overlay
        if len(masks) > 0:
            overlay = np.zeros((*img.shape[:2], 4))
            for mask in masks:
                color = np.concatenate([np.random.rand(3), [0.4]])
                overlay[mask > 0] = color
            axes[i, 1].imshow(overlay)

        # 박스 + 레이블 표시
        for box, label, score in zip(boxes, labels, scores):
            cx, cy, bw, bh = box
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            rect = plt.Rectangle(
                (x1, y1), bw * w, bh * h,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            axes[i, 1].add_patch(rect)
            axes[i, 1].text(
                x1, y1 - 4, f'{label} {score:.2f}',
                color='lime', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5)
            )

        n_det = len(labels)
        axes[i, 1].set_title(f'Grounded SAM ({n_det} detections)', fontsize=10)
        axes[i, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'grounded_sam_heatmap.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# 6. Feature 분포 시각화 (t-SNE)
# ──────────────────────────────────────────────────────────
def viz_distribution(features_dict, image_paths):
    print("\n[Distribution] t-SNE visualization...")

    valid = {k: v for k, v in features_dict.items() if v is not None}
    if not valid:
        print("  No features to visualize")
        return

    n_models = len(valid)
    n_cols = min(3, n_models)
    n_rows = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 9 * n_rows))
    axes = np.array(axes).flatten()
    # 빈 칸 숨기기
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img.thumbnail((50, 50))
        thumbs.append(np.array(img))

    for ax, (name, feats) in zip(axes, valid.items()):
        n = len(feats)

        if n < 4:
            reducer = PCA(n_components=2)
            method = 'PCA'
        else:
            perp = min(30, n - 1)
            reducer = TSNE(n_components=2, perplexity=perp,
                           random_state=42, init='pca')
            method = 't-SNE'

        embed = reducer.fit_transform(feats)

        ax.scatter(embed[:, 0], embed[:, 1], s=200, alpha=0.3,
                   c=range(n), cmap='tab10')

        for j, (x, y) in enumerate(embed):
            ab = AnnotationBbox(OffsetImage(thumbs[j], zoom=1),
                                (x, y), frameon=True, pad=0.1)
            ax.add_artist(ab)

        ax.set_title(f'{name} Feature Space ({method})', fontsize=12)
        ax.set_xlabel(f'{method} dim 1')
        ax.set_ylabel(f'{method} dim 2')
        ax.grid(True, alpha=0.3)

        margin = (embed.max(0) - embed.min(0)) * 0.15
        ax.set_xlim(embed[:, 0].min() - margin[0], embed[:, 0].max() + margin[0])
        ax.set_ylim(embed[:, 1].min() - margin[1], embed[:, 1].max() + margin[1])

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'feature_distribution.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# 7. Cross-model 유사도 비교
# ──────────────────────────────────────────────────────────
def viz_cross_model_similarity(features_dict, image_paths):
    print("\n[Cross-Model] Similarity matrix...")

    valid = {k: v for k, v in features_dict.items() if v is not None}
    if len(valid) < 2:
        return

    n_models = len(valid)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    for ax, (model_name, feats) in zip(axes, valid.items()):
        feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sim_matrix = feats_norm @ feats_norm.T

        im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f'{model_name} Cosine Similarity', fontsize=11)

        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f'{sim_matrix[i,j]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if sim_matrix[i,j] < 0.5 else 'black')

        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'cross_model_similarity.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print(f"Device:    {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"Test dir:  {TEST_DIR}")
    print(f"Output:    {OUTPUT_DIR}")
    print("=" * 70)

    image_paths = load_image_paths(TEST_DIR)
    n_images = len(image_paths)

    all_timings = {}

    # ── 1) DINOv2 ───────────────────────────────────────
    print("\n" + "─" * 70)
    dino_cls, dino_patches, t_dino = run_dinov2(image_paths)
    all_timings['DINOv2'] = t_dino
    viz_dinov2_heatmap(image_paths, dino_patches)

    # ── 2) CLIP ─────────────────────────────────────────
    print("\n" + "─" * 70)
    clip_feats, clip_sims, t_clip = run_clip(image_paths)
    if clip_feats is not None:
        all_timings['CLIP'] = t_clip
        viz_clip(image_paths, clip_sims)

    # ── 3) SAM ──────────────────────────────────────────
    print("\n" + "─" * 70)
    sam_feats, sam_masks, t_sam = run_sam(image_paths)
    if sam_masks is not None:
        all_timings['SAM'] = t_sam
        viz_sam(image_paths, sam_masks)

    # ── 4) Grounding DINO ───────────────────────────────
    print("\n" + "─" * 70)
    gdino_feats, gdino_results, t_gdino = run_grounding_dino(image_paths)
    if gdino_results is not None:
        all_timings['GroundingDINO'] = t_gdino
        viz_grounding_dino(image_paths, gdino_results)

    # ── 5) Grounded SAM ─────────────────────────────────
    print("\n" + "─" * 70)
    gsam_feats, gsam_results, t_gsam = run_grounded_sam(image_paths)
    if gsam_results is not None:
        all_timings['GroundedSAM'] = t_gsam
        viz_grounded_sam(image_paths, gsam_results)

    # ── 6) Feature 분포 (t-SNE) ─────────────────────────
    print("\n" + "─" * 70)
    feature_dict = {
        'DINOv2': dino_cls,
        'CLIP': clip_feats,
        'SAM': sam_feats,
        'GroundingDINO': gdino_feats,
        'GroundedSAM': gsam_feats,
    }
    viz_distribution(feature_dict, image_paths)

    # ── 7) Cross-model 유사도 ───────────────────────────
    viz_cross_model_similarity(feature_dict, image_paths)

    # ──────────────────────────────────────────────────────
    # 시간 측정 요약 표
    # ──────────────────────────────────────────────────────
    W = 84
    print("\n" + "=" * W)
    print(f"TIMING SUMMARY ({n_images} images on {DEVICE.upper()})")
    print("=" * W)
    print(f"{'Model':<14} {'Load':>7} {'Warmup':>7} "
          f"{'Inf/img':>10} {'Inf+IO/img':>12} {'Total Inf':>11} {'FPS':>7}")
    print("─" * W)
    for name, t in all_timings.items():
        load = t.get('load', 0)
        warmup = t.get('warmup', 0)
        inf_per = t.get('inference_per_img', 0) * 1000
        total_per = t.get('per_image_total', 0) * 1000
        total_inf = t.get('inference_total', 0)
        fps = 1.0 / t['inference_per_img'] if t.get('inference_per_img', 0) > 0 else 0
        print(f"{name:<14} {load:>5.2f}s  {warmup:>5.2f}s  "
              f"{inf_per:>8.1f}ms  {total_per:>10.1f}ms  "
              f"{total_inf:>9.2f}s  {fps:>6.2f}")
    print("=" * W)

    # SAM의 단계별 시간 (있을 때)
    if 'SAM' in all_timings:
        sam_t = all_timings['SAM']
        print("\nSAM detailed breakdown:")
        print(f"  set_image (encoder forward): {sam_t['set_image_per_img']*1000:>7.1f} ms/img")
        print(f"  mask_gen   (auto generation): {sam_t['mask_gen_per_img']*1000:>7.1f} ms/img")
        print(f"  → mask_gen이 전체의 "
              f"{sam_t['mask_gen_per_img']/(sam_t['set_image_per_img']+sam_t['mask_gen_per_img'])*100:.0f}% 차지")

    print("\n" + "=" * 70)
    print(f"Done! Check {OUTPUT_DIR}/")
    print("  - dinov2_heatmap.png         : Patch feature PCA-RGB + attention")
    print("  - clip_heatmap.png           : Text-image similarity")
    print("  - sam_heatmap.png            : Automatic segmentation masks")
    print("  - grounding_dino_heatmap.png : Open-set text-prompted detection boxes")
    print("  - grounded_sam_heatmap.png   : Text-prompted boxes + SAM masks")
    print("  - feature_distribution.png   : t-SNE comparison across models")
    print("  - cross_model_similarity.png : Image similarity matrix per model")
    print("=" * 70)