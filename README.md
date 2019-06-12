# Cloudera and Hortonworks Logo - Object Detection
## Data Science
### Object Detection Using Tensorflow
#### Retrain Existing Tensorflow Models

**Level**: Moderate

**Language**: Python

**Requirements**: 
- Python 

**Author**: Ian R Brooks

**Follow**: [LinkedIn - Ian Brooks PhD](https://www.linkedin.com/in/ianrbrooksphd/)

**Entire Project Inspired and/or stolen from**: [Tensorflow Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) 

**Additional Links**:
[Object Detection Tutorial Link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html "link1")
[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection "link2")
[Free Object Labeling Tool](https://github.com/tzutalin/labelImg "link3")
[Another Great Object Detection Tutorial]( https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9 "link4")
[Yet Another Object Detection Tutorial](https://3sidedcube.com/guide-retraining-object-detection-models-tensorflow/ "link5")
[Convert Tensorflow Model for TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tpu_exporters.md "link6")

## Corporate Logo Object Detection

![objectdetection](https://github.com/BrooksIan/LogoTL/blob/master/Images/test/both.jpg "objdect")

This Github repo is designed to be optmized for Cloudera Data Science Workbench (CDSW), but it's not required.  

In this project, the included scripts and images will create an Cloudera and Hortonworks logo Object Detection Tensorflow model  

## CDSW Run Instructions

1.  In CSDW, download the project using the git url for [here](https://github.com/BrooksIan/LogoTL.git) 
2.  Open a new session, run the CDSW-build.sh script at the terminal prompt. 

## Command Line Instructions

1. Download the project using the git url for [here](https://github.com/BrooksIan/LogoTL.git) 

2. Install Tensorflow
```bash
pip3 install tensorflow
pip install tensorflow
```
3.  Download Tensorflow Models Git Repo
```bash
#Clone Tensorflow Model Git Repo
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git
```

4. Convert XML Image Labels to CSV (Optional - CSV files have been provided in Annotations Dir)
```bash
#Convert XML Labels to CSV
python xml_to_csv.py -i Images/train -o annotations/train_labels.csv
python xml_to_csv.py -i Images/test -o annotations/test_labels.csv
```

5. Convert CSV Labels to Tensorflow Record 
```bash
#Convert CSV to TF-Record
python3 generate_tfrecord.py --label0=Cloudera --label1=Hortonworks --csv_input=annotations/train_labels.csv --img_path=Images/train  --output_path=annotations/train.record
python3 generate_tfrecord.py --label0=Cloudera --label1=Hortonworks --csv_input=annotations/test_labels.csv --img_path=Images/test  --output_path=annotations/test.record
```

6. Download Original Tensorflow Model
```bash
#Download Original SSD Tensorflow Model
cd
mkdir pre-trained-model
cd pre-trained-model
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar -xzf ssd_inception_v2_coco_2018_01_28.tar.gz
```

7. Install COCO API
```bash
#COCO API Install
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ~/tensorflow/models/research/
```

8. Download Google's Protobuffer Tools
```bash
cd ~/tensorflow/models/research/
# From tensorflow/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
```

9. Create Protobuffers for Object Dectection Model
```bash
# From tensorflow/models/research/
cd ~/tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.
```

10.  Export Path to the Protobuffer Output Library
```bash
# From tensorflow/models/research/
cd ~/tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

11. Retrain Object Detection Model to Create New Model
```bash
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config
```
  If Everything goes to plan, then you should see this type of output with steps.  Please keep in mind this could take hours if using CPU(s) to complete:

```bash
Use standard file APIs to check for files with this prefix.
INFO:tensorflow:Restoring parameters from pre-trained-model/ssd_inception_v2_coco_2018_01_28/model.ckpt
INFO:tensorflow:Restoring parameters from pre-trained-model/ssd_inception_v2_coco_2018_01_28/model.ckpt
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Starting Session.
INFO:tensorflow:Starting Session.
INFO:tensorflow:Saving checkpoint to path training/model.ckpt
INFO:tensorflow:Saving checkpoint to path training/model.ckpt
INFO:tensorflow:Starting Queues.
INFO:tensorflow:Starting Queues.
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:Recording summary at step 0.
INFO:tensorflow:Recording summary at step 0.
INFO:tensorflow:global step 1: loss = 18.2354 (21.366 sec/step)
INFO:tensorflow:global step 1: loss = 18.2354 (21.366 sec/step)
INFO:tensorflow:global step 2: loss = 17.8257 (3.002 sec/step)
INFO:tensorflow:global step 2: loss = 17.8257 (3.002 sec/step)
INFO:tensorflow:global step 3: loss = 16.4008 (2.777 sec/step)
INFO:tensorflow:global step 3: loss = 16.4008 (2.777 sec/step)
INFO:tensorflow:global step 4: loss = 15.9959 (2.743 sec/step)
INFO:tensorflow:global step 4: loss = 15.9959 (2.743 sec/step)
INFO:tensorflow:global step 5: loss = 15.4355 (2.243 sec/step)
```
12.  Export Model Graphs
13.  Convert Tensorflow Model to Tensorflow Lite Model
