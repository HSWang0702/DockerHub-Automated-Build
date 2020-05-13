FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
MAINTAINER Oscar<oscar.wang@coretronic.com>

RUN apt-get update

RUN apt-get install -y python-software-properties software-properties-common git vim locales libsm6 libxrender1 libxext-dev -y

RUN apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 libnvinfer-dev=5.1.5-1+cuda10.0

RUN apt-get install -y pkg-config zip g++ zlib1g-dev unzip python3-tk

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

RUN add-apt-repository ppa:jonathonf/python-3.6 -y && apt-get update && apt-get install python3.6 python3-pip python3.6-dev -y

RUN python3.6 -m pip install opencv-python scipy h5py pandas scikit-image matplotlib keras Pillow sklearn contextlib2 lxml jupyter imutils flask

RUN python3.6 -m pip install tensorflow-gpu==1.14.0

RUN python3.6 -m pip install -U gast==0.2.2
RUN python3.6 -m pip install -U numpy==1.16.4

ENV QT_X11_NO_MITSHM=1
RUN ldconfig

RUN mkdir /workspace
WORKDIR /workspace

CMD ["bash"]