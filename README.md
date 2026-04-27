# Backbone Laboratory

자율주행 perception 파이프라인에서 사용할 **최적의 vision foundation model 백본을 선정**하기 위해 만든 분석 레포지토리입니다. DINOv2, CLIP, SAM 세 모델의 representation 품질, 추론 속도, 활용 가능성을 정량/정성적으로 비교합니다.

---

## Why this repo?

자율주행 perception, 라벨링 자동화, end-to-end 학습의 mid-level representation 등 다양한 downstream task에서 어떤 foundation model을 백본으로 쓸 것인가는 중요한 설계 결정입니다. 이 레포는 다음을 비교 분석합니다:

- **Feature representation의 품질** (PCA-RGB, t-SNE, cosine similarity)
- **추론 속도** (model load / warmup / pure inference / I/O 포함 시간 분리 측정)
- **출력의 활용성** (each model이 제공하는 정보의 형태)

---

## Models Overview

세 foundation model은 **궁극적인 목표와 출력 구조가 모두 다릅니다**. 이를 이해하는 것이 백본 선정의 출발점입니다.

| 모델 | 궁극 목표 | 학습 방식 | 출력 헤드 | 입출력 |
|------|----------|----------|----------|-------|
| **DINOv2** | 범용 vision representation | Self-supervised (self-distillation) | ❌ (encoder only) | image → feature |
| **CLIP** | 이미지-텍스트 정렬 | Contrastive (image-text pair) | ❌ (encoder × 2) | image+text → 공통 embedding |
| **SAM** | 임의 객체 segmentation | Supervised (1B mask 데이터) | ✅ Mask Decoder | image+prompt → mask |

### DINOv2 — "좋은 vision feature를 뽑는 백본"

```
Image → ViT Encoder → patch features [N, D] + CLS token [D]
```

- **목표**: 어떤 vision task에든 transfer 가능한 강력한 범용 feature
- **출력 헤드**: 없음 (학습 시 projection head는 inference에서 제거)
- **자율주행 활용**: M2M 학습의 mid-representation, LiDAR-카메라 fusion feature, BEV encoder 초기화

### CLIP — "이미지와 텍스트를 같은 공간에 매핑"

```
Image → Image Encoder → image_embedding [D]  ─┐
Text  → Text Encoder  → text_embedding  [D]  ─┴─ cosine similarity
```

- **목표**: 이미지와 자연어 설명을 같은 latent space에서 비교 가능하게
- **출력 헤드**: 없음 (두 encoder 출력을 같은 차원으로 정렬)
- **자율주행 활용**: Zero-shot 객체 분류 ("a vehicle", "a pedestrian"), 멀티모달 fusion

### SAM — "프롬프트 기반 segmentation 전문가"

```
Image  → Image Encoder  → image_embedding  ─┐
Prompt → Prompt Encoder → prompt_embedding ─┴─→ Mask Decoder ★ → masks
```

- **목표**: 어떤 객체든 prompt만 주면 정확한 mask 예측
- **출력 헤드**: ✅ **Mask Decoder** (Two-way Transformer + dynamic mask MLP)
  - 3개 후보 마스크 (whole/part/subpart) + IoU 점수
- **자율주행 활용**: 라벨링 자동화, instance segmentation, LiDAR 클러스터를 prompt로 활용

### 함께 쓰는 패턴

세 모델은 보완적이라 조합해서 사용하는 것이 일반적입니다:

```
Grounded-SAM:  텍스트(CLIP-like) → 박스 → SAM mask
DINOv2 + SAM:  DINOv2 feature 클러스터링 → SAM prompt → 자동 라벨링
CLIP + SAM:    SAM 모든 마스크 → 각 영역 CLIP feature → open-vocabulary segmentation
```

---

## What this repo analyzes

`vit_dino_test.py`를 실행하면 다음 5가지 시각화가 생성됩니다:

| 출력 파일 | 내용 | 용도 |
|----------|------|------|
| `dinov2_heatmap.png` | Patch feature PCA-RGB + foreground attention + self-similarity | DINOv2의 spatial representation 품질 평가 |
| `clip_heatmap.png` | 이미지-텍스트 카테고리 유사도 막대그래프 | CLIP의 zero-shot 분류 능력 평가 |
| `sam_heatmap.png` | Automatic segmentation 마스크 overlay | SAM의 segmentation 결과 품질 평가 |
| `feature_distribution.png` | t-SNE로 세 모델의 feature 공간 비교 | 어떤 모델이 이미지를 잘 구분하는지 |
| `cross_model_similarity.png` | 이미지 간 cosine similarity matrix | 모델별 유사도 인식 차이 |

또한 모델별로 다음 시간을 분리 측정합니다:

- Model load
- Warmup (CUDA kernel compilation)
- Pure inference (forward만)
- Inference + I/O (실제 사용 시나리오)

---

## Installation

- **Clone repo**

    ```sh
    git clone https://github.com/soyeongkim/foundation_model_analysis.git
    cd foundation_model_analysis
    ```

- **Conda 환경 생성 및 패키지 설치**

    ```sh
    conda create -n labeling python=3.11 -y
    conda activate labeling

    pip install "numpy<2"
    pip install torch torchvision
    pip install matplotlib scikit-learn pillow opencv-python
    pip install git+https://github.com/openai/CLIP.git
    pip install open_clip_torch
    pip install git+https://github.com/facebookresearch/segment-anything.git
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    ```

- (Optional) DINOv2 가속용 xFormers 설치

    ```sh
    pip install xformers
    ```

---

## Usage

테스트할 이미지를 `test/` 폴더에 넣고:

```sh
python vit_dino_test.py
```

결과는 `feature_viz/` 폴더에 저장됩니다.

---

## Selection Guide

자율주행 시나리오별 백본 선택 가이드:

| 시나리오 | 추천 모델 | 이유 |
|----------|----------|------|
| LiDAR + 카메라 fusion feature | **DINOv2** | 강력한 visual representation, downstream에 유리 |
| 객체 카테고리 인식 (zero-shot) | **CLIP** | 자연어 쿼리로 분류 가능 |
| 정밀 객체 마스크 (라벨링) | **SAM** | 실제 segmentation 마스크 생성 |
| End-to-End 학습 backbone | **DINOv2** | M2M 학습의 mid-representation에 적합 |
| 데이터셋 라벨 자동 생성 | **CLIP + SAM** | 텍스트로 검색 + 정밀 분할 |