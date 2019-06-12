#Install Tensorflow
pip3 install tensorflow
pip install tensorflow

#Prerequisite for Tensorflow Models
pip3 install pillow
pip3 install lxml
pip3 install jupyter
pip3 install matplotlib
pip3 install opencv

#Clone Tensorflow Model Git Repo
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git

#Convert XML Labels to CSV
python xml_to_csv.py -i Images/train -o annotations/train_labels.csv
python xml_to_csv.py -i Images/test -o annotations/test_labels.csv

#Convert CSV to TF-Record
python3 generate_tfrecord.py --label0=Cloudera --label1=Hortonworks --csv_input=annotations/train_labels.csv --img_path=Images/train  --output_path=annotations/train.record
python3 generate_tfrecord.py --label0=Cloudera --label1=Hortonworks --csv_input=annotations/test_labels.csv --img_path=Images/test  --output_path=annotations/test.record

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

# From tensorflow/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip

# From tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.

# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#Train the Model
#python3 tensorflow/models/research/object_detection/legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config
