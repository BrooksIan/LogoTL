# Cloudera Logo Object Detection 
## Deep Learning
### Transfer Learning with Tensorflow

## Introduction - Corporate Logo Object Detection <a name="introduction"></a>
The goal of this project is to build a Tensorflow Object Detection model designed to detect the Cloudera and Hortonworks logos.  This project will include the scripts, photos, and instructions that are required to follow this project at home. This Github repo include annotated images and completed models for quick testing.    

![hwlogodetection](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/HWLogoDect.gif "hwlogodect")
![clogodetection](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/ClouderaLogoDect.gif "clogodect")
![COlogodetection](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/CLogoDect.gif "cologodect")

**Language**: Python
**Requirements**: 
- Python 3.8
- Tensorflow 2.2

**Author**: Ian R Brooks
**Follow**: [LinkedIn - Ian Brooks PhD](https://www.linkedin.com/in/ianrbrooksphd/)

# Table of Contents
1. [Introduction](#introduction)
2. [Links](#links)
    1. [Object Detection](#linksObjDect)
    2. [Image Augmentation For Object Detection](#linksImgAug)
    3. [Converting Tensorflow Models to Tensorflow Lite Models](#linksConvert)
    4. [Google Coral Dev Board](#linksCoral)
    5. [Tools](#linksTools)

3. [Image Preprocessing - Image and Label Preparation (Optional)](#ImgPrep)
4. [Data Augmentation - Synthetic Image Creation](#DataAug)
5. [Build Tensorflow Model Instructions](#ModelBuild)
    1. [CDSW/CML Run Instructions](#ModelBuildCDSW)
    2. [Quick Instructions](#ModelBuildLuck)
    3. [Step By Step Command Line Instructions](#ModelBuildStepBy)
        1. [Download and Install Libraries](#ModelBuildStepBy1)
        2. [Preprocess the Images](#ModelBuildStepBy2)
        3. [Transfer Learning - Retrain Model With New Detection Objects](#ModelBuildStepBy3)
            
6. [Convert Tensorflow Model to Tensorflow Lite Instructions](#ModelConvert)
7. [Compile Tensorflow Lite Model for Edge TPU](#EdgeTPU)
8. [Deploy Object Detection Model Coral Dev Board](#CoralDeploy)


## Links <a name="links"></a>
**Object Detection**: <a name="linksObjDect"></a>
- [TensorFlow 2 Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
- [Object Detection Tutorial Link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html "link1")
- [Another Object Detection Tutorial]( https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9 "link4")
- [Yet Another Object Detection Tutorial](https://3sidedcube.com/guide-retraining-object-detection-models-tensorflow/ "link5")
- [Logo Object Detection Article](https://towardsdatascience.com/google-object-detection-api-to-detect-brand-logos-fd9e113725d8)
- [Logo Object Detection Article Using SSD](https://towardsdatascience.com/logo-detection-in-images-using-ssd-bcd3732e1776)

**Image Augmentation For Object Detection**: <a name="linksImgAug"></a>
- [Great Read on Data Augmentation for Object Detection](https://blog.paperspace.com/data-augmentation-for-bounding-boxes/)
- [Image Augmentation Examples in Python](https://towardsdatascience.com/image-augmentation-examples-in-python-d552c26f2873)
- [Image Augmentation Using Keras](https://machinelearningmastery.com/image-augmentation-deep-learning-keras/)

**Converting Tensorflow Models**: <a name="linksConvert"></a>
- [Importing a TensorFlow GraphDef based Models into TensorFlow.js](https://www.tensorflow.org/js/tutorials/conversion/import_saved_model)
- [Exporting Tained Model for Inference](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md "link7")
- [Convert Tensorflow Model for TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tpu_exporters.md "link6")
- [Convert Tensorflow Model to TFLite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/cmdline_reference.md "link8")

**Google Coral Dev Board**: <a name="linksCoral"></a>
- [Coral Board Hardware](https://coral.withgoogle.com/)
- [Coral Board Getting Started](https://coral.withgoogle.com/docs/dev-board/get-started/)
- [Retrain Object Detection Model Tutorial](https://coral.withgoogle.com/docs/edgetpu/retrain-detection/)
- [Coral Dev Board - Hands On](https://medium.com/@aallan/hands-on-with-the-coral-dev-board-adbcc317b6af)


**Tools**: <a name="linksTools"></a>
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection "link2")
- [Online Edge TPU Compiler](https://coral.withgoogle.com/web-compiler "link9")
- [LabelImg - Free Object Labeling Tool](https://github.com/tzutalin/labelImg "link3")
- [Data Augmentation for Object Detection GitHub](https://github.com/Paperspace/DataAugmentationForObjectDetection)
- [ImageMagick - Free Image Processing Tool](https://imagemagick.org/)
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)
- [Gif Maker!!!](https://ezgif.com/video-to-gif)

## Image Preprocessing - Image and Label Preparation (Optional)<a name="ImgPrep"></a>

Since this is an Object Detection model, images and annotation labels, which x,y coordinate information on the location of the object in the image, are both required for training.  To create these label annotations, the software package LabelImg can create the object's label annotation into XML files.  Please note this step is optional, these files have been provided in this project, and they are avaiable in the Images/train/ or Images/test/ directories.  Below is a screenshot of the the application.

![LabelImg](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/labelObjects.png)

## Data Augmentation - Synthetic Image Creation <a name="DataAug"></a>

Considering this is a Deep Learning model, the training set should be in the 1000s of photographs, but this project only has 10s of photos. In order to create trainset that is proper size, Data Augmentation will be required to create synthetic images for training.  Using this [library](https://github.com/Paperspace/DataAugmentationForObjectDetection), this project will take the originally provided photos and create synthetic images for training to boost model performance.  This [article](https://blog.paperspace.com/data-augmentation-for-bounding-boxes/) on the subject is a must read to fully understand this project.

This process is automated by provided scripts, but the user will need to determine the amount of synthetic training examples that will be created. 

Below are a few different examples, please note the object labels are updated for the image.

![AugImg1](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/imgAug1.png)
![AugImg2](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/imgAug2.png)
![AugImg3](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/imgAug3.png)
![AugImg4](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/imgAug4.png)


## Build Tensorflow Object Detection Model Instructions <a name="ModelBuild"></a>

**Please Note**: Any scripts that use '~/\' in the path are assuming this is the home directory of the downloaded project.  This is the default for CDSW. 

### CDSW/CML Run Instructions <a name="ModelBuildCDSW"></a>

1.  In CSDW/CML, download the project using the git url for [here.](https://github.com/BrooksIan/LogoTL.git) 
2.  Open a new session, run the CDSW-build.sh script at the terminal prompt, which contains the following operating code. 

### Manual Instructions <a name="ModelBuildStepBy"></a>

#### Download and Install Libraries <a name="ModelBuildStepBy1"></a>
1. Download the project using the git url for [here.](https://github.com/BrooksIan/LogoTL.git) 

2. Project Setup
```bash
./scripts/setup.sh
```

#### Preprocessing Images <a name="ModelBuildStepBy2"></a>
1. Convert CSV labels to Tensorflow TF-Record type. 
```bash
./scripts/imagePrep.sh
```
2. (Optional) Validate images for sizing errors
```bash
 python3 scripts/checkImages.py 
```

#### Build Model Using Transfer Learning - Retrain Model With New Detection Objects <a name="ModelBuildStepBy3"></a>
1. Start model training process. 
```bash
./scripts/buildModel.sh
```

####  Export TF Model to Saved Model Format <a name="ModelBuildStepBy4"></a>
1. Export model 
```bash
./scripts/exportModel.sh
```

####  Export Saved Model To Java Script Format <a name="ModelBuildStepBy5"></a>
1. Export model 
```bash
./scripts/exportModelJS.sh
```

## Convert Tensorflow Model to Tensorflow Lite Instructions <a name="ModelConvert"></a>

1. Export inference graph into Home directory.
```bash
cd

#Optional at this point if PATH and PYTHONPATH are already set
export PYTHONPATH=$PYTHONPATH:tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:tensorflow/models/research/slim
export PYTHONPATH=$PYTHONPATH:tensorflow/models/research/object_detection
export PATH=$PATH:~/.local/bin 

python3 scripts/export_inference_graph.py --input_type image_tensor \
--pipeline_config_path training/ssd_inception_v2_coco.config \
--trained_checkpoint_prefix training/model.ckpt-<***Check Point Number Here***> \
--output_directory trained-inference-graphs/output_inference_graph_v1
```

If this command is successful, then the trained inference graph will be created. 
```bash
ls trained-inference-graphs/output_inference_graph_v1.pb
```

2.  Convert Tensorflow model to Tensorflow Lite model.
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

3. Evaulated the Saved Model using CLI tools.  
```bash
saved_model_cli show --dir trained-inference-graphs/output_inference_graph_v1/saved_model --all
```
4. Convert Tensorflow model to Tensorflow Lite model with TOCO.
```bash
#Convert TF Graphs to TFLite Model
toco --output_file=trainedModels/LogoObjD.tflite \
  --graph_def_file=trainedTFLite/tflite_graph.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --inference_input_type=QUANTIZED_UINT8 \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays=normalized_input_image_tensor \
  --input_shape=1,300,300,3 \
  --input_data_type=QUANTIZED_UINT8 \
  --output_format=TFLITE \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
  --mean_values=128 \
  --std_dev_values=128 \
  --default_ranges_min=0 \
  --default_ranges_max=300 \
  --allow_custom_ops 
```

## Compile Tensorflow Lite Model for Edge TPU <a name="EdgeTPU"></a>
 Use [Online Edge TPU Compiler](https://coral.withgoogle.com/web-compiler) to prepare LogoObjD.tflite model for TPU.  If the compiler finishes, then you should see the screen below.  Please note, this project includes the resulting edgetpu.tflite model, which are located in the trainedModel directory. 

![OnlineCompile](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/OnlineCompiler.png)

## Deploy Object Detection Model to a Coral Dev Board <a name="CoralDeploy"></a>
1. Copy the the edgetpu.tflite file to the Coral Dev board or the model from this project can be downloaded using the following command. This is assuming MDT has been setup: [Coral Board Getting Started.](https://coral.withgoogle.com/docs/dev-board/get-started/)  

```bash
#On the Coral Dev Board - Copy Model
wget https://github.com/BrooksIan/LogoTL/raw/master/trainedModels/LogoObjD_<***ID Number Here***>_edgetpu.tflite
```

2. Deploy the model to Edge TPU
```bash
edgetpu_detect_server --model LogoObjD_<***ID Number Here***>_edgetpu.tflite --label label.txt --threshold=0.51
```

3. Open a web browser to the address: http:// (Coral Dev Board Host Address):(Default Port)
    
    Here is an example: http://192.168.1.245:4664/

![logodetection](https://github.com/BrooksIan/LogoTL/blob/master/Images/project/LogoDect.gif "logodect")

