FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

RUN conda update conda -y

# Install libraries in the brand new image. 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         git \
         python3-opencv \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for all the subsequent Dockerfile instructions.
WORKDIR /opt/program

COPY . GroundingDINO/

RUN mkdir weights ; cd weights ; wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth ; cd ..

RUN python -m pip install 'setuptools>=77' 'packaging>=24.2' wheel ninja
RUN FORCE_CUDA=1 python -m pip install --no-build-isolation 'torch-ms-deform-attn'
RUN python -m pip install ./GroundingDINO

COPY docker_test.py docker_test.py

CMD [ "python", "docker_test.py" ]