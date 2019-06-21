# Transfer Learning on Tensorflow and Tensorflow Lite Models 
## Data Science
### Object Detection on Logos Using Tensorflow

![objectdetection](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/both.jpg "objdect")

**Language**: Python

**Requirements**: 
- Python 3.6
- Tensorflow 1.13

**Author**: Ian R Brooks

**Follow**: [LinkedIn - Ian Brooks PhD](https://www.linkedin.com/in/ianrbrooksphd/)

**Object Detection Links**:
- [Object Detection Tutorial Link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html "link1")
- [Another Object Detection Tutorial]( https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9 "link4")
- [Yet Another Object Detection Tutorial](https://3sidedcube.com/guide-retraining-object-detection-models-tensorflow/ "link5")
- [Logo Object Detection Article](https://towardsdatascience.com/google-object-detection-api-to-detect-brand-logos-fd9e113725d8)
- [Logo Object Detection Article Using SSD](https://towardsdatascience.com/logo-detection-in-images-using-ssd-bcd3732e1776)

**Image Augmentation For Object Detection Links**:
- [Great Read on Data Augmentation for Object Detection](https://blog.paperspace.com/data-augmentation-for-bounding-boxes/)
- [Image Augmentation Examples in Python](https://towardsdatascience.com/image-augmentation-examples-in-python-d552c26f2873)
- [Image Augmentation Using Keras](https://machinelearningmastery.com/image-augmentation-deep-learning-keras/)

**Tool Links**:
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection "link2")
- [Online Edge TPU Compiler](https://coral.withgoogle.com/web-compiler "link9")
- [Free Object Labeling Tool](https://github.com/tzutalin/labelImg "link3")
- [Data Augmentation for Object Detection GitHub](https://github.com/Paperspace/DataAugmentationForObjectDetection)

**Converting Tensorflow Models to Tensorflow Lite Models**:
- [Exporting Tained Model for Inference](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md "link7")
- [Convert Tensorflow Model for TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tpu_exporters.md "link6")
- [Convert Tensorflow Model to TFLite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/cmdline_reference.md "link8")

## Corporate Logo Object Detection

This Github repo is designed to be optmized for Cloudera Data Science Workbench (CDSW), but it's not required.  

**Please Note**: Any scripts that use '~/\' in the path are assuming this is the home directory of the downloaded project.  This is the default for CDSW. 

## CDSW Run Instructions

1.  In CSDW, download the project using the git url for [here.](https://github.com/BrooksIan/LogoTL.git) 

2.  Open a new session, run the CDSW-build.sh script at the terminal prompt, which contains the following operating code. 

## Getting Started Super Fast (If you feel lucky)
1. Download the project using the git url for [here.](https://github.com/BrooksIan/LogoTL.git) 

2. Run at terminal prompt.
```bash
./scripts/setup.sh
./scripts/buildModel.sh
```

## Step By Step Command Line Instructions

1. Download the project using the git url for [here.](https://github.com/BrooksIan/LogoTL.git) 

2. [Install Tensorflow](https://www.tensorflow.org/install/pip "link")
```bash
pip3 install tensorflow
pip install tensorflow
```
3.  Download Tensorflow models using their Git Repo and build the project.
```bash
#Clone Tensorflow Model Git Repo
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git
cd /home/cdsw/tensorflow/models/research
python setup.py build
python setup.py install
```

4. Convert XML image labels to CSV. (Optional - CSV files have been provided in annotations Dir)
```bash
#Convert XML Labels to CSV
python ~/scipts/xml_to_csv.py -i Images/train -o ~/annotations/train_labels.csv
python ~/sciptsxml_to_csv.py -i Images/test -o ~/annotations/test_labels.csv
```

5. Convert CSV labels to Tensorflow TF-Record type. 
```bash
#Convert CSV to TF-Record
python3 ~/scipts/generate_tfrecord.py --label0=Cloudera --label1=Hortonworks --csv_input=~/annotations/train_labels.csv --img_path=Images/train  --output_path=annotations/train.record
python3 ~/scipts/generate_tfrecord.py --label0=Cloudera --label1=Hortonworks --csv_input=~/annotations/test_labels.csv --img_path=Images/test  --output_path=~/annotations/test.record
```

6. Download original SSD Tensorflow model.
```bash
#Download Original SSD Tensorflow Model
cd
mkdir pre-trained-model
cd pre-trained-model
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar -xzf ssd_inception_v2_coco_2018_01_28.tar.gz
```

7. Install COCO API.
```bash
#COCO API Install
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ~/tensorflow/models/research/
```

8. Download Google's protobuffer tools.
```bash
# From tensorflow/models/research/
cd ~/tensorflow/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
```

9. Create protobuffers for Object Dectection model.
```bash
# From tensorflow/models/research/
cd ~/tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.
```

10.  Export Path to the protobuffer library.
```bash
# From tensorflow/models/research/
cd ~/tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

11. Retrain Object Detection model to create new Object Detection model.

```bash
# From Home Directory
cd

#Update PATH and PYTHONPATH
cd
export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research/slim
export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research/object_detection
export PATH=$PATH:~/.local/bin 

python3 ~/scripts/train.py --logtostderr --train_dir=~/training/ \
--pipeline_config_path=~/training/ssd_inception_v2_coco.config
```
  If Everything goes to plan, then you should see this type of output with steps.  Please keep in mind this could take HOURS if using CPU(s) to complete:

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

You can also check the Tenorboard with this command
```bash
tensorboard --logdir=~/training --port=8080
```

12. Find the Highest Ranked Checkpoint File. Make a note of the file’s name, as it will be passed as an argument when we call the export_inference_graph.py script.
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

14. Export inference graph into Home directory.
```bash
cd
python3 ~/scripts/export_inference_graph.py --input_type image_tensor \
--pipeline_config_path ~/training/ssd_inception_v2_coco.config \
--trained_checkpoint_prefix ~/training/model.ckpt-4041 \
--output_directory ~/trained-inference-graphs/output_inference_graph_v1.pb
```

If this command is successful, then the trained inference graph will be created. 
```bash
ls ~/trained-inference-graphs/output_inference_graph_v1.pb
```

15.  Convert Tensorflow model to Tensorflow Lite model.
```bash
python3 ~/tensorflow/models/research/object_detection/export_tflite_ssd_graph.py \
    --input_type=image_tensor \
    --input_shape={"image_tensor":[1,300,300,3]} \
    --pipeline_config_path=~/trained-inference-graphs/output_inference_graph_v1/pipeline.config \
    --trained_checkpoint_prefix=~/trained-inference-graphs/output_inference_graph_v1/model.ckpt \
    --output_directory=trainedTFLite \
    --add_postprocessing_op=true \
    --max_detections=10
```

16. Evaulated the Saved Model using CLI tools.  
```bash
saved_model_cli show --dir ~/trained-inference-graphs/output_inference_graph_v1/saved_model --all
```
17. Convert Tensorflow model to Tensorflow Lite model with TOCO.
```bash
#Convert TF Graphs to TFLite Model
toco --output_file=~/LogoObjD.tflite \
  --graph_def_file=~/trainedTFLite/tflite_graph.pb \
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

18.  Use [Online Edge TPU Compiler](https://coral.withgoogle.com/web-compiler) to prepare LogoObjD.tflite model for TPU. 


