FROM hswang0702/caffe-ssd-modify:cuda9.0-cudnn7-devel-ubuntu16.04
MAINTAINER hswang0702 <hswang0702@gmail.com>

RUN apt-get install cpio pciutils -y

COPY l_openvino_toolkit_p_2019.1.094.tgz /opt/l_openvino_toolkit_p_2019.1.094.tgz
COPY openvino_configure.txt /opt/l_openvino_toolkit_p_2019.1.094/openvino_configure.txt

RUN cd /opt && tar -xvzf l_openvino_toolkit_p_2019.1.094.tgz && rm l_openvino_toolkit_p_2019.1.094.tgz && cd l_openvino_toolkit_p_2019.1.094 && \
./install.sh -s openvino_configure.txt

RUN cd /opt/intel/openvino/install_dependencies/ && sed -i 's/sudo -E//g' install_openvino_dependencies.sh && \
chmod +x install_openvino_dependencies.sh && ./install_openvino_dependencies.sh

RUN cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites && \
sed -i 's/sudo -E//g' install_prerequisites.sh && ./install_prerequisites_caffe.sh

RUN python3.6 /opt/intel/openvino_2019.1.094/deployment_tools/model_optimizer/mo/front/caffe/proto/generate_caffe_pb2.py --input_proto /opt/caffe/src/caffe/proto/caffe.proto

RUN sed -i 's/.prior_box_param/.prior_box_clustered_param/g' /opt/intel/openvino_2019.1.094/deployment_tools/model_optimizer/extensions/front/caffe/priorbox_clustered_ext.py && \
sed -i "s/'flip':/# 'flip':/g" /opt/intel/openvino_2019.1.094/deployment_tools/model_optimizer/extensions/front/caffe/priorbox_clustered_ext.py && \
sed -i "/merge_attrs(param, update_attrs)/a \\\n        # Force setting flip because it doesn't exist in proto so merge_attrs will not set it" /opt/intel/openvino_2019.1.094/deployment_tools/model_optimizer/extensions/front/caffe/priorbox_clustered_ext.py && \
sed -i "/# Force setting flip because it doesn't exist in proto so merge_attrs will not set it/a \        mapping_rule.update(\{'flip': 0\})" /opt/intel/openvino_2019.1.094/deployment_tools/model_optimizer/extensions/front/caffe/priorbox_clustered_ext.py

RUN cd /opt/intel/openvino/deployment_tools/demo && sed -i 's/sudo -E//g' demo_squeezenet_download_convert_run.sh && \
sed -i 's/build_dir=.*/build_dir=$samples_path/g' demo_squeezenet_download_convert_run.sh && \
sed -i '/make \$NUM_THREADS classification_sample/a make \$NUM_THREADS interactive_face_detection_demo' demo_squeezenet_download_convert_run.sh && \
./demo_squeezenet_download_convert_run.sh

RUN sed -i 's/sudo -E//g' /opt/intel/openvino/bin/setupvars.sh && sed -i 's/source /. /g' /opt/intel/openvino/bin/setupvars.sh
RUN sed -i "\$a\\\nsource /opt/intel/openvino/bin/setupvars.sh" /root/.bashrc