FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
MAINTAINER hswang0702 <hswang0702@gmail.com>

RUN apt-get update
RUN apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libopenblas-dev vim git python-software-properties software-properties-common libgflags-dev libgoogle-glog-dev liblmdb-dev wget curl zip autoconf automake libtool -y
RUN apt-get install --no-install-recommends libboost-all-dev -y

RUN apt-get install libturbojpeg && ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so 

RUN add-apt-repository ppa:jonathonf/python-3.6 -y && apt-get update && apt-get install python3.6 python3-pip python3.6-dev -y

COPY Caffe-ssd-f5091ca285e0d0cd231453fd97f35059e5379a64.tar.gz /opt/Caffe-ssd-f5091ca285e0d0cd231453fd97f35059e5379a64.tar.gz
RUN cd /opt && tar -xvzf Caffe-ssd-f5091ca285e0d0cd231453fd97f35059e5379a64.tar.gz && rm Caffe-ssd-f5091ca285e0d0cd231453fd97f35059e5379a64.tar.gz && \
mv Caffe-ssd.git Caffe-ssd && cd Caffe-ssd/python && for req in $(cat requirements.txt); do python3.6 -m pip install $req; done
RUN python3.6 -m pip install opencv-python

COPY NVCaffe-6d619b02d12fa8e721e8237aa0e27faa507500d4.tar.gz /opt/NVCaffe-6d619b02d12fa8e721e8237aa0e27faa507500d4.tar.gz
RUN cd /opt && tar -xvzf NVCaffe-6d619b02d12fa8e721e8237aa0e27faa507500d4.tar.gz && rm NVCaffe-6d619b02d12fa8e721e8237aa0e27faa507500d4.tar.gz
RUN cd /opt && mv NVCaffe.git NVCaffe

RUN wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda10.0-cudnn7-devel-ubuntu16.04/libboost_python-py36.a && \
mv libboost_python-py36.a /usr/lib/x86_64-linux-gnu/libboost_python-py36.a && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda10.0-cudnn7-devel-ubuntu16.04/libboost_python-py36.so.1.58.0 && \
mv libboost_python-py36.so.1.58.0 /usr/lib/x86_64-linux-gnu/libboost_python-py36.so.1.58.0 && \
ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py36.so.1.58.0 /usr/lib/x86_64-linux-gnu/libboost_python-py36.so && \
ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py36.so /usr/lib/x86_64-linux-gnu/libboost_python3.so

RUN ln -s /usr/lib/x86_64-linux-gnu/libpython3.6m.so /usr/lib/x86_64-linux-gnu/libpython3.6.so

RUN apt-get autoremove libprotobuf-dev protobuf-compiler -y
RUN cd /opt && git clone -b v3.1.0 --depth 1 https://github.com/protocolbuffers/protobuf.git && cd /opt/protobuf && git checkout -b v3.1.0 && ./autogen.sh && ./configure --prefix=/opt/protobuf3.1.0 && make -j10 && make install
ENV CPLUS_INCLUDE_PATH=/opt/protobuf3.1.0/include/:/usr/local/cuda-10.0/targets/x86_64-linux/include/
ENV PATH=/opt/protobuf3.1.0/bin:$PATH
ENV LIBRARY_PATH=/opt/protobuf3.1.0/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/protobuf3.1.0/lib:$LD_LIBRARY_PATH

RUN cd /opt/Caffe-ssd && make clean && make -j10 && make pycaffe
RUN cd /opt/NVCaffe && make clean && make -j10 && make pycaffe
ENV PYTHONPATH=/opt/caffe/python
ENV QT_X11_NO_MITSHM=1
RUN ldconfig

RUN mkdir /workspace
WORKDIR /workspace

CMD ["bash"]

