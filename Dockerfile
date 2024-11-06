# 베이스 이미지 설정 (여기서는 PyTorch가 사전 설치된 이미지 사용)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 컨테이너 내 작업 디렉토리 설정
WORKDIR /workspace

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 레포지토리 클론
RUN git clone https://github.com/matejgrcic/DenseHybrid.git

# 작업 디렉토리 이동
WORKDIR /workspace/DenseHybrid

# ckp 폴더 생성
RUN mkdir ckp

COPY ckp/dlv3+_cityscapes_ade_negative_finetune.pth /workspace/DenseHybrid/ckp/

# Python 패키지 설치 (requirements.txt에 기반)
RUN pip install --upgrade pip

# 추가로 필요한 pip 패키지 설치
RUN pip install \
    blosc2==2.0.0 \
    certifi==2023.7.22 \
    charset-normalizer==3.2.0 \
    click==8.1.7 \
    cmake==3.27.2 \
    contourpy==1.1.0 \
    cox==0.1.post3 \
    cycler==0.11.0 \
    cython==3.0.0 \
    dill==0.3.7 \
    easydict==1.10 \
    einops==0.6.1 \
    faiss-gpu==1.7.2 \
    filelock==3.12.2 \
    fonttools==4.42.1 \
    fsspec==2023.9.2 \
    gensim==4.3.2 \
    gitdb==4.0.10 \
    gitpython==3.1.32 \
    gputil==1.4.0 \
    grpcio==1.57.0 \
    gurobipy==10.0.2 \
    huggingface-hub==0.17.3 \
    idna==3.4 \
    jinja2==3.1.2 \
    joblib==1.3.2 \
    kiwisolver==1.4.4 \
    lit==16.0.6 \
    markupsafe==2.1.3 \
    matplotlib==3.7.2 \
    mpmath==1.3.0 \
    msgpack==1.0.5 \
    networkx==3.1 \
    nltk==3.8.1 \
    numexpr==2.8.5 \
    numpy==1.25.2 \
    nvidia-cublas-cu11==11.10.3.66 \
    nvidia-cuda-cupti-cu11==11.7.101 \
    nvidia-cuda-nvrtc-cu11==11.7.99 \
    nvidia-cuda-runtime-cu11==11.7.99 \
    nvidia-cudnn-cu11==8.5.0.96 \
    nvidia-cufft-cu11==10.9.0.58 \
    nvidia-curand-cu11==10.2.10.91 \
    nvidia-cusolver-cu11==11.4.0.1 \
    nvidia-cusparse-cu11==11.7.4.91 \
    nvidia-nccl-cu11==2.14.3 \
    nvidia-nvtx-cu11==11.7.91 \
    opencv-python==4.8.1.78 \
    packaging==23.1 \
    pandas==2.0.3 \
    pillow==10.0.0 \
    protobuf==4.24.1 \
    psutil==5.9.5 \
    py-cpuinfo==9.0.0 \
    py3nvml==0.2.7 \
    pyparsing==3.0.9 \
    python-dateutil==2.8.2 \
    pytz==2023.3 \
    pyyaml==6.0.1 \
    regex==2023.8.8 \
    requests==2.31.0 \
    safetensors==0.4.0 \
    scikit-learn==1.3.0 \
    scipy==1.11.2 \
    seaborn==0.12.2 \
    six==1.16.0 \
    smart-open==6.3.0 \
    smmap==5.0.0 \
    sympy==1.12 \
    tables==3.8.0 \
    tensorboardx==2.6.2.2 \
    termcolor==2.3.0 \
    texttable==1.6.7 \
    threadpoolctl==3.2.0 \
    timm==0.9.0 \
    tokenizers==0.14.1 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    tqdm==4.66.1 \
    transformers==4.34.0 \
    triton==2.0.0 \
    typing-extensions==4.7.1 \
    tzdata==2023.3 \
    urllib3==2.0.4 \
    xmltodict==0.13.0

# 환경 활성화 및 실행을 위한 커맨드 설정
CMD ["/bin/bash"]
