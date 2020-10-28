#Download Original SSD Tensorflow Model

echo '  @@@@ Download Pre-Trained Model @@@@'

# Create Folder for model
cd
mkdir pre-trained-model

cd pre-trained-model

# Download Model
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar -xzf ssd_inception_v2_coco_2018_01_28.tar.gz