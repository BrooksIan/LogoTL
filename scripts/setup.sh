# Set Up Script
echo '####### Start Project Setup #######'

# Create Anaconda Tensorflow Package
echo '####### Conda Setup #######'
conda create -n tensorflow pip python=3.8
conda activate tensorflow

# Install Tensorflow
echo '  <><><> Install Tensorflow 2 <><><>'
#pip3 install --ignore-installed --upgrade tensorflow==2.2.0
pip install --ignore-installed --upgrade tensorflow==2.2.0
pip install tensorflowjs

# TensorFlow Object Detection API Installation
echo '  <><><> Object Detection API <><><> '
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git

# Install CV2
pip install opencv-python 

# Protobuf Installation/Compilation
# Download Protobuffer Writers
# From tensorflow/models/research/
cd ~/tensorflow/models/research
wget -O protobuf.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip
unzip protobuf.zip
export PATH=$PATH:/home/cdsw/tensorflow/models/research/bin
pip install protobuf-compiler

# COCO API installation
echo '  <><><> COCO API Install <><><> '
pip install cython

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools /home/cdsw/tensorflow/models/research

# Object Detection API Install 
echo '  <><><> OD API Install <><><> '
# From within TensorFlow/models/research/
cd /home/cdsw/tensorflow/models/research/
cp object_detection/packages/tf2/setup.py .
python3 -m pip install .

# Test Installation
echo '  <><><> Test Project Installation  <><><> '
cd /home/cdsw/tensorflow/models/research/
python3 object_detection/builders/model_builder_tf2_test.py

# Download Base TF Model
echo '  <><><> Download Base TF Model  <><><> '
cd
#./scripts/downloadModel.sh
