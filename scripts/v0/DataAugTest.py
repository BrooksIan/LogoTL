# Image Loading Code used for these examples
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv


from DataAugmentationForObjectDetection.data_aug.data_aug import *
from DataAugmentationForObjectDetection.data_aug.bbox_util import *
import cv2 
import pickle as pkl


### Test Code ##  
img2 = cv2.imread("Images/train/c19.jpeg")[:,:,::-1] 
bboxes2 = pkl.load(open("Images/project/Cloudera.pkl", "rb"))
print(bboxes2)

#Random Horizonal Flip
img_f, bboxes_f = RandomHorizontalFlip(1)(img2.copy(), bboxes2.copy())
plotted_img = draw_rect(img_f, bboxes_f)
plt.imshow(plotted_img)
plt.show()
print(bboxes_f)

#Random Scale
img_s, bboxes_s = RandomScale(0.3, diff = True)(img2.copy(), bboxes2.copy())
plotted_img = draw_rect(img_s, bboxes_s)
plt.imshow(plotted_img)
plt.show()
print(bboxes_s)

#Random Translate
img_t, bboxes_t = RandomTranslate(0.3, diff = True)(img2.copy(), bboxes2.copy())
plotted_img = draw_rect(img_t, bboxes_t)
plt.imshow(plotted_img)
plt.show()
print(bboxes_t)

#Random Rotate
img_r, bboxes_r = RandomRotate(20)(img2.copy(), bboxes2.copy())
plotted_img = draw_rect(img_r, bboxes_r)
plt.imshow(plotted_img)
plt.show()
print(bboxes_r)

#Random Shear
img_sh, bboxes_sh = RandomShear(0.7)(img2.copy(), bboxes2.copy())
plotted_img = draw_rect(img_sh, bboxes_sh)
plt.imshow(plotted_img)
plt.show()
print(bboxes_sh)

#Random Resize
img_rs, bboxes_rs = Resize(608)(img2.copy(), bboxes2.copy())
plotted_img = draw_rect(img_rs, bboxes_rs)
plt.imshow(plotted_img)
plt.show()
print(bboxes_rs)

#HVS
img_hvs, bboxes_hvs = RandomHSV(100, 100, 100)(img2.copy(), bboxes2.copy())
plotted_img = draw_rect(img_hvs, bboxes_hvs)
plt.imshow(plotted_img)
plt.show()
print(bboxes_hvs)

#Sequence stages
seq = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear()])
img_seq, bboxes_seq = seq(img2.copy(), bboxes2.copy())

plotted_img = draw_rect(img_seq, bboxes_seq)
plt.imshow(plotted_img)
plt.show()
print(bboxes_seq)