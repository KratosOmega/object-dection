#################################################################
# install tensorflow
# make sure the tensorflow version == 1.9.0 (required by object-detection API)
# python -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 2
# python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3
#################################################################
pip install tensorflow

#pip install --upgrade pip  # for Python 2.7
#pip3 install --upgrade pip # for Python 3.n

#pip install --upgrade tensorflow      # for Python 2.7
#pip3 install --upgrade tensorflow     # for Python 3.n
#pip install --upgrade tensorflow-gpu  # for Python 2.7 and GPU
#pip3 install --upgrade tensorflow-gpu # for Python 3.n and GPU

#pip install --upgrade tensorflow-gpu==1.4.1 # for a specific version






#################################################################
# tensorflow dependencies
#################################################################
#echo "deb http://archive.ubuntu.com/ubuntu/ vivid universe" | sudo tee -a "/etc/apt/sources.list"
sudo apt-get install protobuf-compiler python3-pil python3-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib










#################################################################
# Add Libraries to PYTHONPATH
# Note: if .sh failed, then manually cd to
# tensorflow/models/research/
# and run export command as below
# also note that [pwd] should be updated with your correspounding
# pwd path
#################################################################
# ! NOTE: before tensorflow can be used, must compile the
#         protobuf correspondingly to your tensorflow /models/research path
cd ../../../models/research/
#export PYTHONPATH=$PYTHONPATH:~/Desktop/auvsi/models/research:~/Desktop/auvsi/models/research/slim #export PYTHONPATH=$PYTHONPATH:[pwd]:[pwd]/slim

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim










#################################################################
# pip version conflict after upgrade to new pip version
#################################################################
#hash -r pip










#################################################################
# Protobuf Compilation
# if there is compilation error then do
# Manual protobuf-compiler installation and usage
#################################################################
# From tensorflow/models/research/
#$ protoc object_detection/protos/*.proto --python_out=.

# Manual protobuf-compiler installation and usage
# From tensorflow/models/research/
#$ wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
#$ unzip protobuf.zip

# From tensorflow/models/research/
#$ ./bin/protoc object_detection/protos/*.proto --python_out=.








































sudo apt-get install python-tk
sudo apt install python-pli
pip2 install --user lxml









# build tensorflow from source & checkout branch 'r1.9'
git clone https://github.com/tensorflow/tensorflow


# install bazel follow by https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu

# configure tensorflow
sudo apt-get install python-numpy python-dev python-pip python-wheel

cd ./tensorflow
./configure

# Build the pip package
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package





