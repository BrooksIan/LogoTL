#Image Preprocessing
echo '####### Start Image Preprocessing #######'
conda activate tensorflow

##Convert CSV to TF-Record
echo '  $$$$$$ Convert CSV to TF-Record $$$$$'

# From Home Directory
cd

python3 scripts/generate_tfrecord.py \
--c=annotations/train_labels.csv \
--x=images/train  \
--i=images/train  \
--o=annotations/train.record \
--l=label_map.pbtxt

#Post Data Augmentation - Test Set
python3 scripts/generate_tfrecord.py \
--c=annotations/test_labels.csv \
--i=images/test  \
--x=images/test  \
--o=annotations/test.record \
--l=label_map.pbtxt