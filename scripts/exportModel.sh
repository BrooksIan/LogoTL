# Export TF Model

#conda activate tensorflow

cd
mkdir exported-models

python3 tensorflow/models/research/object_detection/exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path training/pre_configed_ssd_mobilenet_v2_fpn.config \
--trained_checkpoint_dir pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8checkpoint \
--output_directory exported-models/cldr_logo_ssd_modilenet/