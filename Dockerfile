FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
ARG UID
ARG USER_NAME

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    ca-certificates \
    curl \
    cmake \
    ffmpeg \
    git \
    python3-pip \
    python3-setuptools \
    python3-dev \
    sudo \
    ssh \
    unzip \
    vim \
    wget && rm -rf /var/lib/apt/lists/*

# RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh && \
#     bash /tmp/miniconda.sh -bfp /usr/local && \
#     rm -rf /tmp/miniconda.sh
# RUN conda update -y conda

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

RUN adduser $USER_NAME --u $UID --quiet --gecos "" --disabled-password && \
    echo "$USER_NAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME

RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config
RUN echo "UsePAM no" >> /etc/ssh/sshd_config

USER $USER_NAME

EXPOSE 6006 6007 6008 6009
