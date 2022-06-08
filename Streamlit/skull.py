from distutils.command.upload import upload
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage.filters import threshold_otsu
from skimage import measure
import matplotlib.image as imgṬ
from skimage.morphology import disk
import skimage.morphology as morph
from keras.models import model_from_json
import streamlit as st

# from mrcnn.config import Config
# from mrcnn import utils
# import mrcnn.model as modellib
# from mrcnn import visualize
# from mrcnn.model import log

# eNTER ROOT DIRECTORY
def skull_s(upload):
  #img = cv2.reshape(1, 512,512,3)
  img = np.ascontiguousarray(upload)
  #img = np.rot90(img,-1)
 
  mednoise = cv2.medianBlur(img, 3)

  img_array = mednoise
  ret,thresh = cv2.threshold(img_array,145,300,cv2.THRESH_TOZERO_INV)
  dimn=thresh.shape
  
  th = threshold_otsu(thresh)
  
  binim1 = thresh > th

  eroded_image = morph.erosion(binim1, disk(3))

  labelimg = measure.label(eroded_image, background=0)
  prop = measure.regionprops(labelimg)

  ncount = len(prop)
  
  argmax = 0
  maxarea = 0
  
  for i in range(ncount):
      if(prop[i].area > maxarea):
          maxarea = prop[i].area
          argmax = i
  

  bmask = np.zeros(thresh.shape, dtype=np.uint8)

  bmask[labelimg == (argmax+1)] = 1

  dilated_mask = morph.dilation(bmask, disk(6))
  brain = img_array*dilated_mask
  return brain

def detection(upload):
  json_file = open('/content/drive/MyDrive/dataset/detection.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  det_model = model_from_json(loaded_model_json)
  # load weights into new model
  det_model.load_weights("/content/drive/MyDrive/dataset/detection.h5")
  img = np.ascontiguousarray(upload)
  # img = cv2.imread(upload)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(gray,(150,150))
  img = img.reshape(1,150,150,3)
  p = det_model.predict(img)
  p = np.argmax(p,axis=1)[0]
  if p==1:
    st.success("The Model predicts that there is no tumor")
    # st.snow()
    return False
  else:
    st.warning("There is a Tumor")
    return True



def classification(upload):
  json_file = open('/content/drive/MyDrive/dataset/model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  class_model = model_from_json(loaded_model_json)
  # load weights into new model
  class_model.load_weights("/content/drive/MyDrive/dataset/model.h5")

  img = np.ascontiguousarray(upload)
  #img = cv2.imread(upload)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  img = cv2.resize(gray,(150,150))
  img = img.reshape(1,150,150,3)
  p = class_model.predict(img)
  p = np.argmax(p,axis=1)[0]

  if p==0:
      st.info("The Model predicts that it is a Glioma tumor")
  elif p==1:
      st.info("The model predicts that there is Meningioma tumor")
  elif p==2:
      st.info("The Model predicts that it is a Pituitary tumor")
  elif (p != 0) and (p != 1) and (p != 2):
      st.info("Model cant predict")


def segmentation(upload):
  json_file = open('/content/drive/MyDrive/dataset/seg_model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  seg_model = model_from_json(loaded_model_json)
  # load weights into new model
  seg_model.load_weights("/content/drive/MyDrive/dataset/mask_rcnn_tumor_detect_0030.h5")

  results = seg_model.detect([upload], verbose=0)
  r = results[0]

  visualize.display_instances(upload,r['rois'],r['masks'],r['class_ids'],'tumor')
  
