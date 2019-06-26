# Build Tensorflow Model Script
#!/bin/bash 

echo '####### Start Model Build #######'

#Train Tensorflow Model
echo '####### Train Model #######'

#From Home Directory
cd
python3 ~/scripts/train.py --logtostderr --train_dir=~/training/ \
--pipeline_config_path=~/training/ssd_inception_v2_coco.config

#Find the Highest Ranked Checkpoint File
ls -t ~/training/model.ckpt* | head -1 | cut -d'-' -f 2 | cut -d'.' -f 1
#Make a note of the file’s name, as it will be passed as an argument when we call the export_inference_graph.py script.

#Save Check Point Value
CKPNUM=$(ls -t ~/training/model.ckpt* | head -1 | cut -d'-' -f 2 | cut -d'.' -f 1)
echo '####### The highest check point value is:' $CKPNUM

#Export Inference Graph For TFLite
echo '####### Export Inference Graph #######'

cd
python3 ~/scripts/export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=~/training/ssd_inception_v2_coco.config \
--trained_checkpoint_prefix=~/training/model.ckpt-$CKPNUM \
--output_directory=~/trained-inference-graphs/output_inference_graph_v4

#Check Saved Model
echo '####### Check Saved Model #######'

saved_model_cli show --dir trained-inference-graphs/output_inference_graph_v1/saved_model --all

#Export TFLite SSD Inference Graph
echo '####### Export TFLite SSD Inference Graph #######'

python3 ~/tensorflow/models/research/object_detection/export_tflite_ssd_graph.py \
    --input_type=image_tensor \
    --input_shape={"image_tensor":[1,600,600,3]} \
    --pipeline_config_path=~/trained-inference-graphs/output_inference_graph_v1/pipeline.config \
    --trained_checkpoint_prefix=~/trained-inference-graphs/output_inference_graph_v1/model.ckpt \
    --output_directory=~/trainedTFLite \
    --add_postprocessing_op=true \
    --max_detections=10
    
#Convert with TOCO
#Convert TF Graphs to TFLite Model
echo '####### Convert TF Graphs to TFLite Model #######'

toco --output_file=~/trainedModels/LogoObjD.tflite \
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
  --default_ranges_min=0 \
  --default_ranges_max=300 \
  --allow_custom_ops 
