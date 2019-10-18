FROM hswang0702/caffe-ssd-modify:cuda10.0-cudnn7-devel-ubuntu16.04m
MAINTAINER hswang0702 <hswang0702@gmail.com>

RUN apt-get install cpio pciutils -y

COPY l_openvino_toolkit_p_2018.3.343.tgz /opt/l_openvino_toolkit_p_2018.3.343.tgz
COPY openvino_configure.txt /opt/l_openvino_toolkit_p_2018.3.343/openvino_configure.txt

RUN cd /opt && tar -xvzf l_openvino_toolkit_p_2018.3.343.tgz && rm l_openvino_toolkit_p_2018.3.343.tgz && cd l_openvino_toolkit_p_2018.3.343 && \
./install.sh -s openvino_configure.txt

RUN cd /opt/intel/computer_vision_sdk/install_dependencies/ && sed -i 's/sudo -E//g' install_cv_sdk_dependencies.sh && \
chmod +x install_cv_sdk_dependencies.sh && ./install_cv_sdk_dependencies.sh

RUN cd /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/install_prerequisites && \
sed -i 's/sudo -E//g' install_prerequisites.sh && sed -i 's/python_binary=python3$/python_binary=python3.6/g' install_prerequisites.sh  && ./install_prerequisites_caffe.sh

RUN python3.6 /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo/front/caffe/proto/generate_caffe_pb2.py --input_proto /opt/Caffe-ssd/src/caffe/proto/caffe.proto

#RUN sed -i 's/.prior_box_param/.prior_box_clustered_param/g' /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/caffe/priorbox_clustered_ext.py && \
#sed -i "s/'flip':/# 'flip':/g" /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/caffe/priorbox_clustered_ext.py && \
#sed -i "/merge_attrs(param, update_attrs)/a \\\n        # Force setting flip because it doesn't exist in proto so merge_attrs will not set it" /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/caffe/priorbox_clustered_ext.py && \
#sed -i "/# Force setting flip because it doesn't exist in proto so merge_attrs will not set it/a \        mapping_rule.update(\{'flip': 0\})" /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/caffe/priorbox_clustered_ext.py

RUN sed -i "s/'input_width':/# 'input_width':/g" /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/caffe/detection_output.py && \
sed -i "s/'input_height':/# 'input_height':/g" /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/caffe/detection_output.py && \
sed -i "s/'normalized':/# 'normalized':/g" /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/caffe/detection_output.py

#RUN sed -i "s/'width':/# 'width':/g" /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/caffe/priorbox_ext.py && \
#sed -i "s/'height'/# 'height'/g" /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/caffe/priorbox_ext.py

COPY interactive_face_detection_fddb /opt/intel/computer_vision_sdk/inference_engine/samples/interactive_face_detection_fddb
RUN sed -i "/add_subdirectory(interactive_face_detection_sample)/a add_subdirectory(interactive_face_detection_fddb)" /opt/intel/computer_vision_sdk/inference_engine/samples/CMakeLists.txt



RUN cd /opt/intel/computer_vision_sdk/deployment_tools/demo && sed -i 's/sudo -E//g' demo_squeezenet_download_convert_run.sh && \
sed -i 's/build_dir=.*/build_dir=$samples_path/g' demo_squeezenet_download_convert_run.sh && \
sed -i '/make \$NUM_THREADS classification_sample/a make \$NUM_THREADS interactive_face_detection_demo' demo_squeezenet_download_convert_run.sh && \
./demo_squeezenet_download_convert_run.sh

RUN sed -i 's/sudo -E//g' /opt/intel/computer_vision_sdk/bin/setupvars.sh && sed -i 's/source /. /g' /opt/intel/computer_vision_sdk/bin/setupvars.sh
RUN sed -i "\$a\\\nsource /opt/intel/computer_vision_sdk/bin/setupvars.sh" /root/.bashrc
