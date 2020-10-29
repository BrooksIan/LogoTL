#Export Tensorflow Model's Graph for TF Lite

echo '   ^v^v^ Export TF Model to TF-Lite ^v^v^'
#conda activate tensorflow

cd

#Export TF Model's Graphs to Saved Model for TF Lite
python3 tensorflow/models/research/object_detection/export_tflite_graph_tf2.py \
    --pipeline_config_path training/pre_configed_ssd_mobilenet_v2_fpn.config \
    --trained_checkpoint_dir pre-trained-models/ssd_mobilenet_v2_CLDR \
    --output_directory exported-models/cldr_logo_ssd_modilenet/tflite/
    
#Export Saved Model to TF Lite Model
python3 scripts/convert2TFL.py

