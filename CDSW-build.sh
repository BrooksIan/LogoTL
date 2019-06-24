########## Project Build ######
./scripts/setup.sh
./scripts/imagePrep.sh
./scripts/buildModel.sh
##############################


#Install Tensorflow
#pip3 install tensorflow
#pip install tensorflow

#Prerequisite for Tensorflow Models
#pip3 install pillow
#pip3 install lxml
#pip3 install jupyter
#pip3 install matplotlib
#pip3 install opencv

#Clone Tensorflow Model Git Repo and Build Project
#mkdir tensorflow
#cd tensorflow
#git clone https://github.com/tensorflow/models.git
#cd /home/cdsw/tensorflow/models/research
#python setup.py build
#python setup.py install

#Convert XML Labels to CSV
# From Home Directory
#cd
#python xml_to_csv.py -i Images/train -o annotations/train_labels.csv
#python xml_to_csv.py -i Images/test -o annotations/test_labels.csv

#Convert CSV to TF-Record
# From Home Directory
#cd
#python3 generate_tfrecord.py --label0=Cloudera --label1=Hortonworks --csv_input=annotations/train_labels.csv --img_path=Images/train  --output_path=annotations/train.record
#python3 generate_tfrecord.py --label0=Cloudera --label1=Hortonworks --csv_input=annotations/test_labels.csv --img_path=Images/test  --output_path=annotations/test.record

#Download Original Tensorflow Model
#cd
#mkdir pre-trained-model
#cd pre-trained-model
#wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
#tar -xzf ssd_inception_v2_coco_2018_01_28.tar.gz

#Install Tensorflow - Object Detection Tools

#COCO API Install
#git clone https://github.com/cocodataset/cocoapi.git
#cd cocoapi/PythonAPI
#make
#cp -r pycocotools ~/tensorflow/models/research/
#cd ~/tensorflow/models/research/
##protoc object_detection/protos/*.proto --python_out=.

#Download Protobuffer Writers
# From tensorflow/models/research/
#cd ~/tensorflow/models/research
#wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
#unzip protobuf.zip

# From tensorflow/models/research/
#cd ~/tensorflow/models/research
#./bin/protoc object_detection/protos/*.proto --python_out=.

# From tensorflow/models/research/
#cd ~/tensorflow/models/research
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#Train the Model
#python3 tensorflow/models/research/object_detection/legacy/train.py \
#--logtostderr --train_dir=training/ \
#--pipeline_config_path=training/ssd_inception_v2_coco.config

# From Home Directory
#cd
#python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config

# Copy Export Inference Script To Home Dir
#cd
#cp ~/tensorflow/models/research/object_detection/export_inference_graph.py . 

#Find the Highest Ranked Checkpoint File
#ls -t ~/training/model.ckpt*
#Make a note of the file’s name, as it will be passed as an argument when we call the export_inference_graph.py script.

#Export Inference Graph From Home Directory
#cd
#export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research/
#export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research/slim
#export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research/object_detection


#python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-2139 --output_directory trained-inference-graphs/output_inference_graph_v1.pb
#python3 export_inference_graph.py \
#--input_type=image_tensor \
#--input_shape= {"image_tensor" : [1,600,600,3]} \
#--pipeline_config_path=training/ssd_inception_v2_coco.config \
#--trained_checkpoint_prefix=training/model.ckpt-4041 \
#--output_directory=trained-inference-graphs/output_inference_graph_v1 \
#--add_postprocessing_op=true \
#--write_inference_graph=true

#Export Covert Graph and Checkpoint into TPU Model
#cd #Home directory 

#python3 tensorflow/models/research/object_detection/tpu_exporters/export_saved_model_tpu.py \
#    --pipeline_config_file= trained-inference-graphs/output_inference_graph_v1/pipeline.config \
#    --ckpt_path= trained-inference-graphs/output_inference_graph_v1/model.ckpt \
#    --export_dir= TFLite \
#    --input_placeholder_name= input \
#    --input_type= image_tensor \
#    --input_shape={"image_tensor" : [1,600,600,3]} \
#    --use_bfloat16=false
    

#tflite_convert \
#  --output_file=tfliteModel/LogoObjDetection.tflite \
#  --saved_model_dir=trained-inference-graphs/output_inference_graph_v1/saved_model \
#  --inference_type=QUANTIZED_UINT8 \
#  --input_arrays=["image_tensor"] \
#  --input_shape=1,600,600,3 \
#  --output_arrays=["detection_scores","detection_boxes","detection_classes"]


#Convert to TF Lite Format

#export PYTHONPATH=$PYTHONPATH:/home/cdsw/.local/lib/python3.6/site-packages/tensorflow/contrib/lite/python/convert.py
#pip3 install tf-nightly
#export PATH=$PATH:~/.local/bin 

#This is the winner!!   *** No White SPACEEEEE!!! 
#python3 tensorflow/models/research/object_detection/export_tflite_ssd_graph.py \
#    --input_type = image_tensor \
#    --input_shape={"image_tensor" : [1,600,600,3]} \
#    --pipeline_config_path=trained-inference-graphs/output_inference_graph_v1/pipeline.config \
#    --trained_checkpoint_prefix=trained-inference-graphs/output_inference_graph_v1/model.ckpt \
#    --output_directory=trainedTFLite \
#    --add_postprocessing_op=true \
#    --max_detections=10

#Check Model 
#saved_model_cli show --dir trained-inference-graphs/output_inference_graph_v1/saved_model \
#--tag_set serve \
#--signature_def serving_default

#saved_model_cli show --dir trained-inference-graphs/output_inference_graph_v1/saved_model --all

#Convert with TOCO

#This is Part 2 of WINNER!!! 
#Convert TF Graphs to TFLite Model
#toco --output_file=LogoObjD.tflite \
#  --graph_def_file=trainedTFLite/tflite_graph.pb \
#  --input_format=TENSORFLOW_GRAPHDEF \
#  --inference_input_type=QUANTIZED_UINT8 \
#  --inference_type=QUANTIZED_UINT8 \
#  --input_arrays=normalized_input_image_tensor \
#  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
#  --output_format=TFLITE \
#  --input_shape=1,300,300,3 \
#  --input_data_type=QUANTIZED_UINT8 \
#  --mean_values=128 \
#  --std_dev_values=128 \
#  --default_ranges_min=0.0 \
#  --default_ranges_max=300 \
#  --allow_custom_ops 


#Run the Model Locally
#python3 tensorflow/models/research/object_detection/model_main.py \
#    --pipeline_config_path=trained-inference-graphs/output_inference_graph_v1/pipeline.config \
#    --model_dir=trained-inference-graphs/output_inference_graph_v1 \
#    --num_train_steps=30 \
#    --sample_1_of_n_eval_examples=1 \
#    --alsologtostderr
