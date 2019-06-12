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
[Object Detection Link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html "link1")

## Corporate Logo Object Detection

![objectdetection](https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwjA3dOD1eTiAhWHoJ4KHROpBWcQjRx6BAgBEAU&url=https%3A%2F%2Fwww.analyticsvidhya.com%2Fblog%2F2018%2F06%2Funderstanding-building-object-detection-model-python%2F&psig=AOvVaw2vYlBn3UbRSaxf-nD0xRAM&ust=1560453658635232 "objdect")

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

4. Convert XML Image Labels to CSV
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
