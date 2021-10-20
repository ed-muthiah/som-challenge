# ======================================================================================================================
# = Common things.
# ======================================================================================================================
FROM nvidia/cuda:11.4.2-devel-ubuntu18.04
RUN apt-get update -y
ARG DEBIAN_FRONTEND=noninteractive

# ======================================================================================================================
# = NVIDIA settings.
# ======================================================================================================================
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility

# ======================================================================================================================
# = Install system dependencies.
# ======================================================================================================================
RUN apt-get -y install software-properties-common apt-transport-https ca-certificates
RUN add-apt-repository ppa:deadsnakes/ppa 
RUN update-ca-certificates
RUN apt update 
RUN apt-get -y --no-install-recommends install \
    git zip unzip sudo vim wget cmake m4 build-essential autopoint autoconf automake libtool pkg-config \
    python3.9 python3-pip python3-dev python3-venv python3-all python3-all-dev libpython3-all-dev python3-tk virtualenv \
	ffmpeg libsm6 libxext6 python3-setuptools

# ======================================================================================================================
# = Install python dependencies.
# ======================================================================================================================

RUN mkdir /app
WORKDIR /app
COPY . .
RUN pip3 install --upgrade setuptools pip wheel
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/Joeclinton1/google-images-download.git
WORKDIR /google-images-download
RUN /bin/bash -c python3 setup.py install
WORKDIR ..

# ======================================================================================================================
# = RUN
# ======================================================================================================================

EXPOSE 80
CMD ["streamlit", "main.py --server.port 80"]