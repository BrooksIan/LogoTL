# Cloudera and Hortonworks Logo - Object Detection
## Data Science
### Object Detection Using Tensorflow
#### Retrain Existing Tensorflow Models

![objectdetection](https://github.com/BrooksIan/LogoTL/blob/master/Images/both.jpg "objdect")

**Level**: Moderate

**Language**: Python

**Requirements**: 
- Python 

**Author**: Ian R Brooks

**Follow**: [LinkedIn - Ian Brooks PhD](https://www.linkedin.com/in/ianrbrooksphd/)

**Additional Links**:
- [Object Detection Tutorial Link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html "link1")
- [Another Object Detection Tutorial]( https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9 "link4")
- [Yet Another Object Detection Tutorial](https://3sidedcube.com/guide-retraining-object-detection-models-tensorflow/ "link5")
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection "link2")
- [Free Object Labeling Tool](https://github.com/tzutalin/labelImg "link3")
- [Exporting a trained model for inference](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md "link7")
- [Convert Tensorflow Model for TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tpu_exporters.md "link6")
- [Convert Tensorflow Model to TFLite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/cmdline_reference.md "link8")
- [Online Edge TPU Compiler](https://coral.withgoogle.com/web-compiler "link9")

## Corporate Logo Object Detection

This Github repo is designed to be optmized for Cloudera Data Science Workbench (CDSW), but it's not required.  

In this project, the included scripts and images will create an Cloudera and Hortonworks logo Object Detection Tensorflow model  

## CDSW Run Instructions

1.  In CSDW, download the project using the git url for [here](https://github.com/BrooksIan/LogoTL.git) 
2.  Open a new session, run the CDSW-build.sh script at the terminal prompt. 

## Command Line Instructions

1. Download the project using the git url for [here](https://github.com/BrooksIan/LogoTL.git) 

2. [Install Tensorflow](https://www.tensorflow.org/install/pip "link")
```bash
pip3 install tensorflow
pip install tensorflow
```
3.  Download Tensorflow Models Git Repo and Build Project
```bash
#Clone Tensorflow Model Git Repo
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git
cd /home/cdsw/tensorflow/models/research
python setup.py build
python setup.py install
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
# From tensorflow/models/research/
cd ~/tensorflow/models/research/
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
# From Home Directory
cd

python3 train.py --logtostderr --train_dir=training/ \
--pipeline_config_path=training/ssd_inception_v2_coco.config
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
12.  Copy Export Inference Script To Home Dir
```bash
cd
cp ~/tensorflow/models/research/object_detection/export_inference_graph.py .
```
13. Find the Highest Ranked Checkpoint File. Make a note of the file’s name, as it will be passed as an argument when we call the export_inference_graph.py script.
```bash
ls -t ~/training/model.ckpt*
```
If training was sucessful, then results will be displayed.  Please keep in mind, the numeric values will be different. 
```bash
$ ls -t ~/training/model.ckpt*
/home/cdsw/training/model.ckpt-4041.meta
/home/cdsw/training/model.ckpt-4041.index
/home/cdsw/training/model.ckpt-4041.data-00000-of-00001
/home/cdsw/training/model.ckpt-3769.meta
/home/cdsw/training/model.ckpt-3769.index
/home/cdsw/training/model.ckpt-3769.data-00000-of-00001
/home/cdsw/training/model.ckpt-3497.meta
/home/cdsw/training/model.ckpt-3497.index
/home/cdsw/training/model.ckpt-3497.data-00000-of-00001
/home/cdsw/training/model.ckpt-3225.meta
/home/cdsw/training/model.ckpt-3225.index
/home/cdsw/training/model.ckpt-3225.data-00000-of-00001
/home/cdsw/training/model.ckpt-2954.meta
/home/cdsw/training/model.ckpt-2954.index
/home/cdsw/training/model.ckpt-2954.data-00000-of-00001
```

14. Export Inference Graph Into Home Directory
```bash
cd
python3 export_inference_graph.py --input_type image_tensor \
--pipeline_config_path training/ssd_inception_v2_coco.config \
--trained_checkpoint_prefix training/model.ckpt-4041 \
--output_directory trained-inference-graphs/output_inference_graph_v1.pb
```

If this command is successful, then the trained inference graph will be created. 
```bash
ls trained-inference-graphs/
output_inference_graph_v1.pb
```

15.  Convert Tensorflow Model to Tensorflow Lite Model
```bash
python3 tensorflow/models/research/object_detection/export_tflite_ssd_graph.py \
    --input_type=image_tensor \
    --input_shape={"image_tensor":[1,600,600,3]} \
    --pipeline_config_path=trained-inference-graphs/output_inference_graph_v1/pipeline.config \
    --trained_checkpoint_prefix=trained-inference-graphs/output_inference_graph_v1/model.ckpt \
    --output_directory=trainedTFLite \
    --add_postprocessing_op=true \
    --max_detections=10
```

16. Check Saved Model 
```bash
saved_model_cli show --dir trained-inference-graphs/output_inference_graph_v1/saved_model --all
```
17. Use Convert Tensorflow Model to Tensorflow Lite Model with TOCO
```bash
#Convert TF Graphs to TFLite Model
toco --output_file=LogoObjD.tflite \
  --graph_def_file=trainedTFLite/tflite_graph.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --inference_input_type=QUANTIZED_UINT8 \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
  --output_format=TFLITE \
  --input_shape=1,300,300,3 \
  --input_data_type=QUANTIZED_UINT8 \
  --mean_values=128 \
  --std_dev_values=128 \
  --default_ranges_min=0.0 \
  --default_ranges_max=300 \
  --allow_custom_ops 
```

18.  Use [Online Edge TPU Compiler](https://coral.withgoogle.com/web-compiler) to prepare .TFLite model for TPU 


