# Backbone Laboratory

자율주행 perception 파이프라인에서 사용할 **최적의 vision foundation model 백본을 선정**하기 위해 만든 분석 레포지토리입니다. DINOv2, MAE, CLIP, SigLIP, SAM, Grounding DINO, Grounded SAM 일곱 모델의 representation 품질, 추론 속도, 활용 가능성을 정량/정성적으로 비교합니다.

---

## Why this repo?

자율주행 perception, 라벨링 자동화, end-to-end 학습의 mid-level representation 등 다양한 downstream task에서 어떤 foundation model을 백본으로 쓸 것인가는 중요한 설계 결정입니다. 이 레포는 다음을 비교 분석합니다:

- **Feature representation의 품질** (PCA-RGB, t-SNE, cosine similarity)
- **추론 속도** (model load / warmup / pure inference / I/O 포함 시간 분리 측정)
- **출력의 활용성** (each model이 제공하는 정보의 형태)

---

## Models Overview

다섯 foundation model은 **궁극적인 목표와 출력 구조가 모두 다릅니다**. 이를 이해하는 것이 백본 선정의 출발점입니다.

| 모델 | 궁극 목표 | 학습 방식 | 출력 헤드 | 입출력 |
|------|----------|----------|----------|-------|
| **DINOv2** | 범용 vision representation | Self-supervised (self-distillation) | ❌ (encoder only) | image → feature |
| **MAE** | 범용 vision representation | Self-supervised (masked reconstruction) | ❌ (encoder only) | image → feature |
| **CLIP** | 이미지-텍스트 정렬 | Contrastive (image-text pair) | ❌ (encoder × 2) | image+text → 공통 embedding |
| **SigLIP** | 이미지-텍스트 정렬 (개선) | Contrastive (sigmoid loss) | ❌ (encoder × 2) | image+text → 공통 embedding |
| **SAM** | 임의 객체 segmentation | Supervised (1B mask 데이터) | ✅ Mask Decoder | image+prompt → mask |
| **Grounding DINO** | 텍스트 기반 오픈셋 검출 | Contrastive (image-text-box) | ✅ Detection Head | image+text → boxes |
| **Grounded SAM** | 텍스트 기반 검출 + 정밀 분할 | Grounding DINO + SAM 결합 | ✅ Box + Mask | image+text → boxes+masks |

---

### DINOv2 — "좋은 vision feature를 뽑는 백본"

```
Image → ViT-B/14 Encoder → patch features [N×D] + CLS token [D]
```

- **목표**: 어떤 vision task에든 transfer 가능한 강력한 범용 feature
- **backbone**: ViT-B/14 (patch size 14, image size 518×518 = 37×37 patches)
- **출력 헤드**: 없음 (학습 시 projection head는 inference에서 제거)
- **자율주행 활용**: M2M 학습의 mid-representation, LiDAR-카메라 fusion feature, BEV encoder 초기화

---

### MAE — "마스크 복원으로 학습한 범용 백본"

```
Image → random masking (75%) → ViT-B/16 Encoder → patch features [196×D] + CLS token [D]
```

- **목표**: 마스킹된 이미지 패치를 복원하며 강력한 visual representation 학습
- **backbone**: ViT-B/16 (patch size 16, image size 224×224 = 14×14 = 196 patches)
- **출력 헤드**: 없음 (pretrain 시 decoder는 inference에서 제거)
- **DINOv2와의 차이**: DINOv2는 self-distillation(contrastive), MAE는 masked autoencoding(generative). MAE는 low-level texture 복원 학습이라 patch feature의 질감·구조 민감도가 다름
- **자율주행 활용**: DINOv2 대비 경량 대안, downstream fine-tuning에 강점

---

### CLIP — "이미지와 텍스트를 같은 공간에 매핑"

```
Image → ViT-B/32 Encoder → image_embedding [D]  ─┐
Text  → Text Encoder     → text_embedding  [D]  ─┴─ cosine similarity
```

- **목표**: 이미지와 자연어 설명을 같은 latent space에서 비교 가능하게
- **backbone**: ViT-B/32 (OpenAI pretrained)
- **출력 헤드**: 없음 (두 encoder 출력을 같은 차원으로 정렬)
- **자율주행 활용**: Zero-shot 객체 분류 ("a vehicle", "a pedestrian"), 멀티모달 fusion

---

### SigLIP — "CLIP의 개선판: sigmoid loss + 공간 feature"

```
Image → ViT-B/16 Encoder → patch features [196×D] → mean pool → image_embedding [D]  ─┐
Text  → Text Encoder     →                            text_embedding [D]               ─┴─ sigmoid similarity
```

- **목표**: CLIP과 동일하게 이미지-텍스트 정렬, 단 sigmoid binary loss로 학습 (softmax 불필요)
- **backbone**: ViT-B/16 (SigLIP은 CLS 토큰 없음 — 196개 패치 토큰만 존재)
- **출력 헤드**: 없음
- **CLIP과의 차이**:
  - **Loss**: softmax contrastive → **sigmoid binary** (각 (image, text) 쌍을 독립적으로 판단)
  - **CLS 토큰**: 없음. 모든 196 patch token이 풍부한 공간 정보를 가짐
  - **Similarity 해석**: 확률처럼 0~1 범위이지만 `sigmoid`이므로 여러 카테고리가 동시에 높을 수 있음
- **자율주행 활용**: CLIP 대체 시 더 나은 patch-level feature, zero-shot 분류 정확도 향상

---

### SAM — "프롬프트 기반 segmentation 전문가"

```
Image  → ViT-B Encoder  → image_embedding  ─┐
Prompt → Prompt Encoder → prompt_embedding ─┴─→ Mask Decoder ★ → masks
```

- **목표**: 어떤 객체든 prompt만 주면 정확한 mask 예측
- **backbone**: ViT-B image encoder
- **출력 헤드**: ✅ **Mask Decoder** (Two-way Transformer + dynamic mask MLP)
  - 3개 후보 마스크 (whole / part / subpart) + IoU 점수
- **자율주행 활용**: 라벨링 자동화, instance segmentation, LiDAR 클러스터를 prompt로 활용

---

### Grounding DINO — "텍스트로 원하는 객체를 찾아내는 detector"

```
Image → Swin-T Backbone → image features  ─┐
Text  → BERT Encoder    → text features   ─┴─→ Feature Fusion → boxes + labels
```

- **목표**: 자연어 쿼리로 임의의 객체를 bounding box 형태로 검출 (오픈셋 detection)
- **backbone**: Swin-T (image) + BERT (text)
- **출력 헤드**: ✅ **Detection Head** (box regression + text-grounded classification)
- **특징**: `TEXT_QUERIES`를 `. `으로 구분한 caption으로 입력 → 각 쿼리에 해당하는 박스 + 신뢰도 출력
- **자율주행 활용**: Zero-shot 오픈셋 객체 검출, 라벨링 자동화의 첫 단계

---

### Grounded SAM — "텍스트 검출 + 정밀 분할의 결합"

```
Image + Text → Grounding DINO → boxes
Image + boxes → SAM Predictor → masks
```

- **목표**: 텍스트로 원하는 객체를 찾고, SAM으로 픽셀 단위 마스크까지 생성
- **backbone**: Grounding DINO (Swin-T + BERT) + SAM (ViT-B)
- **출력 헤드**: ✅ **Box + Mask** (두 모델의 출력 결합)
- **특징**: Grounding DINO가 찾은 박스를 SAM에 point/box prompt로 전달
- **자율주행 활용**: 텍스트 기반 자동 annotation pipeline, open-vocabulary instance segmentation

---

### 함께 쓰는 패턴

```
Grounded SAM        : text → Grounding DINO(box) → SAM(mask)
DINOv2 / MAE + SAM  : patch feature 클러스터링 → SAM prompt → 자동 라벨링
CLIP / SigLIP + SAM : SAM 모든 마스크 → 각 영역 feature → open-vocabulary segmentation
DINOv2 vs MAE       : 동일 ViT-B 구조, 다른 pretext task → patch feature 품질 직접 비교
CLIP vs SigLIP      : 동일 image-text 목표, softmax vs sigmoid → 텍스트 유사도 특성 비교
```

---

## What this repo analyzes

`img_backbone_lab.py`를 실행하면 다음 9가지 시각화가 생성됩니다:

| 출력 파일 | 내용 | 용도 |
|----------|------|------|
| `dinov2_heatmap.png` | Patch feature PCA-RGB + foreground attention + self-similarity | DINOv2의 spatial representation 품질 평가 |
| `mae_heatmap.png` | Patch feature PCA-RGB + foreground attention + self-similarity | MAE의 spatial representation 품질 평가 (DINOv2와 직접 비교) |
| `clip_heatmap.png` | 이미지-텍스트 카테고리 유사도 막대그래프 (softmax) | CLIP의 zero-shot 분류 능력 평가 |
| `siglip_heatmap.png` | Patch PCA-RGB + foreground attention + 텍스트 유사도 막대그래프 (sigmoid) | SigLIP의 spatial feature + 텍스트 정렬 동시 평가 |
| `sam_heatmap.png` | Automatic segmentation 마스크 overlay | SAM의 segmentation 결과 품질 평가 |
| `grounding_dino_heatmap.png` | 텍스트 쿼리 기반 검출 박스 + 신뢰도 레이블 | Grounding DINO의 오픈셋 검출 결과 평가 |
| `grounded_sam_heatmap.png` | 텍스트 기반 검출 박스 + SAM 정밀 마스크 overlay | Grounded SAM의 검출-분할 파이프라인 평가 |
| `feature_distribution.png` | t-SNE / PCA로 일곱 모델의 feature 공간 비교 | 어떤 모델이 이미지를 잘 구분하는지 |
| `cross_model_similarity.png` | 이미지 간 cosine similarity matrix (모델별) | 모델별 유사도 인식 차이 |

또한 모델별로 다음 시간을 분리 측정합니다:

| 측정 항목 | 설명 |
|----------|------|
| Model load | 체크포인트 로드 시간 |
| Warmup | CUDA 커널 컴파일 (첫 forward) |
| Pure inference | forward pass만 (I/O 제외) |
| Inference + I/O | 이미지 읽기 + 전처리 + forward (실사용 시나리오) |

---

## Installation

**Clone repo**

```sh
git clone https://github.com/soyeongkim/backbone_lab.git
cd backbone_lab
```

**Conda 환경 생성 및 패키지 설치**

```sh
conda create -n backbonlab python=3.11 -y
conda activate backbonlab

pip install "numpy<2"
pip install torch torchvision
pip install matplotlib scikit-learn pillow opencv-python
pip install open_clip_torch
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install timm                              # MAE
pip install transformers sentencepiece protobuf  # SigLIP
```

**Grounding DINO 설치**

> CUDA 버전 불일치 문제가 있는 경우 CUDA extension 없이 설치합니다.

```sh
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
# setup.py의 get_extensions()에 return [] 추가 후 설치 (CUDA op 없이 Python fallback 사용)
pip install -e . --no-build-isolation
cd ..
```

> **주의 — editable install 경로 문제**
>
> `pip install -e .`는 설치 시점의 **절대 경로**를 등록합니다. 프로젝트 폴더를 이동하거나 다른 머신에서 클론한 경우 `import groundingdino`가 실패할 수 있습니다.
>
> 증상: `pip show groundingdino`는 설치된 것처럼 보이지만 실행 시 `ModuleNotFoundError` 발생
>
> 해결: `GroundingDINO/` 디렉토리에서 재설치
> ```sh
> cd GroundingDINO
> pip install -e . --no-build-isolation
> cd ..
> ```

**체크포인트 다운로드**

```sh
mkdir -p ckpt

# SAM
wget -P ckpt/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Grounding DINO
wget -P ckpt/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**(Optional) DINOv2 가속용 xFormers**

```sh
pip install xformers
```

---

## Usage

테스트할 이미지를 `test/` 폴더에 넣고:

```sh
python img_backbone_lab.py
```

결과는 `feature_viz/` 폴더에 저장됩니다.

**텍스트 쿼리 변경** (`img_backbone_lab.py` 상단):

```python
TEXT_QUERIES = [
    "a road",
    "a vehicle",
    "a pedestrian",
    ...
]
```

CLIP 유사도 비교, Grounding DINO 검출, Grounded SAM 마스크 생성 모두 이 리스트를 공유합니다.

**검출 임계값 조정**:

```python
GDINO_BOX_THRESHOLD  = 0.35   # 박스 신뢰도 임계값 (낮출수록 더 많이 검출)
GDINO_TEXT_THRESHOLD = 0.25   # 텍스트-박스 매칭 임계값
```

---

## Selection Guide

자율주행 시나리오별 백본 선택 가이드:

| 시나리오 | 추천 모델 | 이유 |
|----------|----------|------|
| LiDAR + 카메라 fusion feature | **DINOv2** | 강력한 visual representation, downstream에 유리 |
| 경량 self-supervised 백본 | **MAE** | DINOv2보다 작은 patch size(16), fine-tuning에 강점 |
| 객체 카테고리 인식 (zero-shot) | **SigLIP** | CLIP 대비 patch feature 풍부, sigmoid로 다중 카테고리 동시 판단 |
| 이미지-텍스트 유사도 비교 | **CLIP / SigLIP** | CLIP은 softmax 단일 선택, SigLIP은 sigmoid 다중 분류에 적합 |
| 정밀 객체 마스크 (라벨링) | **SAM** | 실제 segmentation 마스크 생성 |
| 텍스트 기반 오픈셋 검출 | **Grounding DINO** | 자연어로 원하는 객체만 bounding box 추출 |
| 자동 annotation pipeline | **Grounded SAM** | 텍스트 → 박스 → 마스크 end-to-end 생성 |
| End-to-End 학습 backbone | **DINOv2** | M2M 학습의 mid-representation에 적합 |
| 데이터셋 라벨 자동 생성 | **Grounded SAM** | 텍스트 쿼리만으로 pixel-level 마스크 생성 |
