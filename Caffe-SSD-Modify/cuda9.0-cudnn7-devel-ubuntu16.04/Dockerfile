FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER hswang0702 <hswang0702@gmail.com>

RUN apt-get update
RUN apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libopenblas-dev vim git python-software-properties software-properties-common libgflags-dev libgoogle-glog-dev liblmdb-dev wget curl zip autoconf automake libtool -y
RUN apt-get install --no-install-recommends libboost-all-dev -y

RUN add-apt-repository ppa:jonathonf/python-3.6 -y && apt-get update && apt-get install python3.6 python3-pip python3.6-dev -y

RUN cd /opt && git clone -b ssd --depth 1 https://github.com/weiliu89/caffe.git
RUN cd /opt/caffe/python && for req in $(cat requirements.txt); do python3.6 -m pip install $req; done
RUN python3.6 -m pip install opencv-python

RUN wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/Makefile && \
mv Makefile /opt/caffe/Makefile && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/Makefile.config && \
mv Makefile.config /opt/caffe/Makefile.config

RUN wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/libboost_python-py36.a && \
mv libboost_python-py36.a /usr/lib/x86_64-linux-gnu/libboost_python-py36.a && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/libboost_python-py36.so.1.58.0 && \
mv libboost_python-py36.so.1.58.0 /usr/lib/x86_64-linux-gnu/libboost_python-py36.so.1.58.0 && \
ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py36.so.1.58.0 /usr/lib/x86_64-linux-gnu/libboost_python-py36.so && \
ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py36.so /usr/lib/x86_64-linux-gnu/libboost_python3.so

RUN ln -s /usr/lib/x86_64-linux-gnu/libpython3.6m.so /usr/lib/x86_64-linux-gnu/libpython3.6.so

RUN wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/Makefile && \
mv Makefile /opt/caffe/Makefile && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/Makefile.config && \
mv Makefile.config /opt/caffe/Makefile.config

RUN wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/include/caffe/filler.hpp && \
mv filler.hpp /opt/caffe/include/caffe/filler.hpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/include/caffe/solver.hpp && \
mv solver.hpp /opt/caffe/include/caffe/solver.hpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/include/caffe/layers/focal_loss_layer.hpp && \
mv focal_loss_layer.hpp /opt/caffe/include/caffe/layers/focal_loss_layer.hpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/include/caffe/layers/multibox_loss_layer.hpp && \
mv multibox_loss_layer.hpp /opt/caffe/include/caffe/layers/multibox_loss_layer.hpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/include/caffe/layers/prior_box_clustered.hpp && \
mv prior_box_clustered.hpp /opt/caffe/include/caffe/layers/prior_box_clustered.hpp

RUN wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/data_transformer.cpp && \
mv data_transformer.cpp /opt/caffe/src/caffe/data_transformer.cpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/solver.cpp && \
mv solver.cpp /opt/caffe/src/caffe/solver.cpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/proto/caffe.proto && \
mv caffe.proto /opt/caffe/src/caffe//proto/caffe.proto && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/util/math_functions.cpp && \
mv math_functions.cpp /opt/caffe/src/caffe/util/math_functions.cpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/util/math_functions.cu && \
mv math_functions.cu /opt/caffe/src/caffe/util/math_functions.cu && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/util/bbox_util.cpp && \
mv bbox_util.cpp /opt/caffe/src/caffe//util/bbox_util.cpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/util/bbox_util.cu && \
mv bbox_util.cu /opt/caffe/src/caffe//util/bbox_util.cu && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/layers/detection_output_layer.cu && \
mv detection_output_layer.cu /opt/caffe/src/caffe//layers/detection_output_layer.cu && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/layers/focal_loss_layer.cpp && \
mv focal_loss_layer.cpp /opt/caffe/src/caffe//layers/focal_loss_layer.cpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/layers/focal_loss_layer.cu && \
mv focal_loss_layer.cu /opt/caffe/src/caffe//layers/focal_loss_layer.cu && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/layers/multibox_loss_layer.cpp && \
mv multibox_loss_layer.cpp /opt/caffe/src/caffe//layers/multibox_loss_layer.cpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/layers/prior_box_clustered_layer.cpp && \
mv prior_box_clustered_layer.cpp /opt/caffe/src/caffe//layers/prior_box_clustered_layer.cpp && \
wget https://raw.githubusercontent.com/hswang0702/DockerHub-Automated-Build/master/Caffe-SSD-Modify/cuda9.0-cudnn7-devel-ubuntu16.04/caffe/src/caffe/layers/softmax_loss_layer.cu && \
mv softmax_loss_layer.cu /opt/caffe/src/caffe//layers/softmax_loss_layer.cu

RUN apt-get autoremove libprotobuf-dev protobuf-compiler -y
RUN cd /opt && git clone -b v3.1.0 --depth 1 https://github.com/protocolbuffers/protobuf.git && cd /opt/protobuf && git checkout -b v3.1.0 && ./autogen.sh && ./configure --prefix=/opt/protobuf3.1.0 && make -j10 && make install
ENV CPLUS_INCLUDE_PATH=/opt/protobuf3.1.0/include/
ENV PATH=/opt/protobuf3.1.0/bin:$PATH
ENV LIBRARY_PATH=/opt/protobuf3.1.0/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/protobuf3.1.0/lib:$LD_LIBRARY_PATH

RUN cd /opt/caffe && make clean && make -j10 && make pycaffe
ENV PYTHONPATH=/opt/caffe/python
ENV QT_X11_NO_MITSHM=1
RUN ldconfig

RUN mkdir /workspace
WORKDIR /workspace

CMD ["bash"]

