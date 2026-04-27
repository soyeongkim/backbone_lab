# foundation_model_analysis
Foundation model analysis playground !


# Installation
- Clone Repo
    ```sh
    git clone https://github.com/soyeongkim/foundation_model_analysis.git
    ```
- conda 환경 생성 및 패키지 설치 (FM 관련)
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

- 분석 코드 실행
    ```sh
    python vit_dino_test.py
    ```
