# Evaluate Tensorflow Model Script
#!/bin/bash 
#conda activate tensorflow 

#Eval Tensorflow Model
echo '  ***** Evaluate Model *****'

cd 
python3 tensorflow/models/research/object_detection/model_main_tf2.py \
--model_dir=pre-trained-models/ssd_mobilenet_v2_CLDR \
--pipeline_config_path=training/pre_configed_ssd_mobilenet_v2_fpn.config \
--checkpoint_dir=pre-trained-models/ssd_mobilenet_v2_CLDR