FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y vim wget git libsndfile1 espeak-ng && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir torchmetrics==0.11.1 librosa==0.10.2 phonemizer==3.2.1 pypinyin==0.48.0 lhotse==1.27.0 matplotlib==3.9.2 h5py==3.11.0 wandb==0.17.5

# Install k2 from Hugging Face
RUN pip install --no-cache-dir https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.23.4.dev20230224+cuda11.6.torch1.13.1-cp310-cp310-linux_x86_64.whl

# Install torchaudio with CUDA support
RUN pip install --no-cache-dir torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Clone and install icefall
WORKDIR /workspace
RUN git clone https://github.com/k2-fsa/icefall && \
    cd icefall && \
    pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/workspace/icefall:$PYTHONPATH

# Ensure that libpython3.10.so.1.0 is available in the system path
RUN cp /opt/conda/lib/libpython3.10.so.1.0 /usr/lib/x86_64-linux-gnu/

# Install vall-e
RUN pip install --no-cache-dir -e git+https://github.com/lifeiteng/vall-e.git#egg=valle

# Install additional Python dependencies
RUN pip install --no-cache-dir beartype==0.1.1 lion-pytorch==0.2.2 accelerate==0.33.0

# Set the working directory to /workspace/app
WORKDIR /workspace/app
