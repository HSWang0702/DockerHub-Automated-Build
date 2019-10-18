FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
MAINTAINER Oscar<oscar.wang@coretronic.com>

RUN apt-get update

RUN apt-get install python-software-properties software-properties-common git vim -y

RUN add-apt-repository ppa:jonathonf/python-3.6 -y && apt-get update && apt-get install python3.6 python3-pip python3.6-dev -y

RUN python3.6 -m pip install opencv-python scipy h5py pandas scikit-image matplotlib keras Pillow sklearn

RUN python3.6 -m pip install 'tensorflow-gpu==1.13.1'

ENV QT_X11_NO_MITSHM=1
RUN ldconfig

RUN mkdir /workspace
WORKDIR /workspace

CMD ["bash"]
