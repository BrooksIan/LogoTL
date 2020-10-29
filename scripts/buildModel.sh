# Build Tensorflow Model Script
#!/bin/bash 
# conda activate tensorflow 

echo '####### Start Model Build #######'

#Train Tensorflow Model
echo '  &&&&& Train Model &&&&&'

#Download Model
cd
./scripts/downloadModel.sh


cd 
python3 tensorflow/models/research/object_detection/model_main_tf2.py \
--model_dir=pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8 \
--pipeline_config_path=training/pre_configed_ssd_mobilenet_v2_fpn.config \
--checkpoint_dir=pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\checkpoint