FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL IMAGE="NiChart_DLMUSE"
LABEL VERSION="0.1.7"
LABEL CI_IGNORE="True"

RUN apt-get update && \
    apt-get -y install gcc \
    mono-mcs \
    gnupg2 \
    git \
    htop \
    zip \
    unzip \
    g++ && \ 
    apt-key del 3bf863cc && \ 
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \ 
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \ 
    pip install Cmake

# RUN mkdir /NiChart_DLMUSE 
# COPY ./ /NiChart_DLMUSE/
# RUN cd /NiChart_DLMUSE && pip install .

RUN cd / && \
    git clone https://github.com/CBICA/NiChart_DLMUSE && \
    cd /NiChart_DLMUSE && pip install . 

CMD ["NiChart_DLMUSE" ]
