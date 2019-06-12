#Install Tensorflow
pip3 install tensorflow

#Prerequisite for Tensorflow Models
pip3 install pillow
pip3 install lxml
pip3 install jupyter
pip3 install matplotlib
pip3 install opencv

#Clone Tensorflow Model Git Repo
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git

#Convert XML Labels to CSV
python xml_to_csv.py -i Images/train -o annotations/train_labels.csv
python xml_to_csv.py -i Images/test -o annotations/test_labels.csv

#Convert CSV to TF-Record

python3 generate_tfrecord.py --label0=Cloudera --label1=Hortonworks --csv_input=annotations/train_labels.csv --img_path=Images/train  --output_path=annotations/train.record
python3 generate_tfrecord.py --label0=Cloudera --label1=Hortonworks --csv_input=annotations/test_labels.csv --img_path=Images/test  --output_path=annotations/test.record

