# Export TF Model To Java Script
echo '+++++ Export Model to JS +++++'

#conda activate tensorflow

pip install tensorflowjs

cd

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    exported-models/cldr_logo_ssd_modilenet/saved_model \
    exported-models/cldr_logo_ssd_modilenet/web_model