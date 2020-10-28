# Set Up Script
echo '####### Start Project Setup d#######'


#Install Tensorflow
pip3 install tensorflow
pip install tensorflow

#Prerequisite for Tensorflow Models
pip3 install pillow
pip3 install lxml
pip3 install jupyter
pip3 install matplotlib
pip3 install opencv

#Clone Tensorflow Model Git Repo and Build Project
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git
cd /home/cdsw/tensorflow/models/research
python setup.py build
python setup.py install

#Download Original Tensorflow Model
cd
mkdir pre-trained-model
cd pre-trained-model
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar -xzf ssd_inception_v2_coco_2018_01_28.tar.gz

#Install Tensorflow - Object Detection Tools

#COCO API Install
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ~/tensorflow/models/research/
cd ~/tensorflow/models/research/
#protoc object_detection/protos/*.proto --python_out=.

#Download Protobuffer Writers
# From tensorflow/models/research/
cd ~/tensorflow/models/research
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip

# From tensorflow/models/research/
cd ~/tensorflow/models/research
./bin/protoc object_detection/protos/*.proto --python_out=.

# From tensorflow/models/research/
cd ~/tensorflow/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd
git clone https://github.com/Paperspace/DataAugmentationForObjectDetection.git

#Export Inference Graph From Home Directory
cd
export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research/slim
export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research/object_detection

export PATH=$PATH:~/.local/bin 