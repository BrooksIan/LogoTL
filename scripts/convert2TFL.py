import tensorflow as tf

savedModelDIR = "exported-models/cldr_logo_ssd_modilenet/tflite/saved_model"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(savedModelDIR) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model.
with open('exported-models/cldr_logo_ssd_modilenet/tflite/cldr_model.tflite', 'wb') as f:
  f.write(tflite_model)
  
#  tflite_convert \
#  --saved_model_dir=exported-models/cldr_logo_ssd_modilenet/saved_model/ \
#  --output_file=exported-models/cldr_logo_ssd_modilenet/cldr_mobilenet.tflite