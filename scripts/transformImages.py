# Copyright 2019 Shadow ML
#
# ==============================================================================

r"""Tool To create Synthetic Images Using Data Augmentation

Example Usage:
--------------
python transformImages.py \
    --input_dir path/to/images \
    --numIters NumberOfIterations \
    --image_label_file path/to/image_labels_csv_file \
    --output_path path/to/out_put_cvs_file \
    --label0 Cloudera \
    --label1 Hortonworks \
    --label2 ClouderaOrange\
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import tensorflow as tf

from DataAugmentationForObjectDetection.data_aug.data_aug import *
from DataAugmentationForObjectDetection.data_aug.bbox_util import *
import cv2 
import pickle as pkl

#"""
#Define Application Flags
flags = tf.app.flags
flags.DEFINE_string('img_input_dir', '~/Images/train/', 'Path to image director')
flags.DEFINE_string('image_label_file', '~/annotations/train_labels.csv', 'Path to the image label CSV file')
flags.DEFINE_string('numIters', '100', 'Number of iterations for each image')
flags.DEFINE_string('output_path', '~/annotations/train_labels_DA.csv', 'Path to output image label CSV file')
flags.DEFINE_string('label0', 'Cloudera', 'Name of class[0] label')
flags.DEFINE_string('label1', 'Hortonworks', 'Name of class[1] label')
flags.DEFINE_string('label2', 'ClouderaOrange', 'Name of class[2] label')
                              
FLAGS = flags.FLAGS
#"""

#=============  Method - Write Pickle Files ======================
def writePickleFile(image_fileName, objectAnnotation, ID, class_name):

  #Prepare to Write Annotations to Pickle File
  img_pickle_dir = FLAGS.img_input_dir + "pickle/"  
  fileName = img_pickle_dir + str(image_fileName).split(".")[0] + "_" + str(ID) + "_" + class_name + ".pkl"
  fileObject = open(fileName, 'wb')
  pkl.dump(objectAnnotation, fileObject)
  print('Pickle File Created @: ' + fileName)
  fileObject.close()
  return fileName
#=============  End of Method - Write Pickle Files ===============    


#=============  Method - createSyntheticImages =================
def createSyntheticImage(image_fileName, pickleFile, image_id):
  
  imagePath = FLAGS.img_input_dir +'/'+ image_fileName
  
  #Read Original Image
  img = cv2.imread(imagePath)[:,:,::-1]   
  bboxes = pkl.load(open(pickleFile, "rb"))
  #print(bboxes)
              
  #Sequence Multiple Data Augmentation Steps to a new image
  #seq = Sequence([RandomHSV(40, 40, 30), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear(), RandomResize(640)])
  seq = Sequence([RandomHSV(40, 40, 30), RandomScale(0.3), RandomTranslate(0.3), RandomShear(), Resize(600)])
  img_seq, bboxes_seq = seq(img.copy(), bboxes.copy())
  
  #Save File
  DAImageName = str(image_id) + "_da_"+ image_fileName
  saveDAImagePath =  FLAGS.img_input_dir + "DA/" + DAImageName
  matplotlib.image.imsave(saveDAImagePath, img_seq)
  
  #print('***Data Augmentation ** Synthetic Image Created @ ' + saveDAImagePath) 
  
  return DAImageName, bboxes_seq #Return Object Boundary Boxes
  
#=============  End of Method - createSyntheticImages =================


#=============  Method - writeToCSVOutFile =================
def writeToCSVOutFile(writeRow, writeCode):
  
  DA_label_file = FLAGS.output_path         # default value = annotations/train_labels_DA.csv
  
  with open(DA_label_file, writeCode) as csvOFile:
                writer = csv.writer(csvOFile)
                writer.writerow(writeRow)
                csvOFile.close()
          
#=============  End of Method - writeToCSVOutFile =================


#=============  Method - Main ==============================
def main(_):
    
    
#Define Class Paths 
  label_file= FLAGS.image_label_file        # default value = annotations/train_labels.csv
  DA_label_file = FLAGS.output_path         # default value = annotations/train_labels_DA.csv
  class_name0= FLAGS.label0                 # default value = Cloudera
  class_name1= FLAGS.label1                 # default value = Hortonworks
  class_name2= FLAGS.label2                 # default value = ClouderaOrange
  
  #Open CSV Label File
  with open(label_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
           
            # Output row for DA CSV File
            header_row = ['filename','width','height','class','xmin','ymin','xmax','ymax']
            writeToCSVOutFile(header_row, 'w')
            
        else:
            #print(f'\t{row[0]} w:{row[1]} h:{row[2]} with object: {row[3]} at  xmin:{row[4]} ymin:{row[5]} xmax:{row[6]} ymax:{row[7]} ')
            line_count += 1
            image_fileName = row[0]
            imageW = row[1]
            imageH = row[2]
            xmin = float(row[4])
            ymin = float(row[5])
            xmax = float(row[6])
            ymax = float(row[7])
            class_name = row[3]
            
            #Write Picke File for Image
            objectAnnotation = np.array([xmin, ymin, xmax, ymax, 0], ndmin=2)
            pickleFile = writePickleFile(image_fileName, objectAnnotation, line_count, class_name)
            
            #if line_count == 132:
              
              #Write Original Image to CSV File
              #O_row = [image_fileName, imageW, imageH, class_name, int(xmin), int(ymin), int(xmax), int(ymax)]
              #writeToCSVOutFile(O_row, 'a')
                           
              #Create a new syntethic image based on the number of user provided iterations
            for i in range( int(FLAGS.numIters) ):
                        
                #Create Syntethic Image and Return FileName and Object Boundaries 
                DAImageName, boundaryBoxes = createSyntheticImage(image_fileName, pickleFile, i)
                
                #print(boundaryBoxes)
                
                if boundaryBoxes.shape[0] == 1:
                  xminBB = int(boundaryBoxes[0][0])
                  yminBB = int(boundaryBoxes[0][1])
                  xmaxBB = int(boundaryBoxes[0][2])
                  ymaxBB = int(boundaryBoxes[0][3])
                  
                  DA_row = [ DAImageName , imageW, imageH, class_name, xminBB, yminBB, xmaxBB, ymaxBB]
                  writeToCSVOutFile(DA_row, 'a')
                  
                else:
                  xminBB = int(xmin)
                  yminBB = int(ymin)
                  xmaxBB = int(xmax)
                  ymaxBB = int(ymax)
              
                #print(boundaryBoxes)
              
                # Write Output Row to CSV File
                #DA_row = [ DAImageName , imageW, imageH, class_name, xminBB, yminBB, xmaxBB, ymaxBB]
                #writeToCSVOutFile(DA_row, 'a')
       
            print('***Data Augmentation ** {FLAGS.numIters} Synthetic Images Created For Image ' + image_fileName) 
                                                           
    print('Processed {line_count} images in CVS file: ' + label_file)  
  
    print('Successfully created the Output CSV Label File:{}'.format(output_path))

#=============  End of Method - Main ==============================


if __name__ == '__main__': tf.app.run()
