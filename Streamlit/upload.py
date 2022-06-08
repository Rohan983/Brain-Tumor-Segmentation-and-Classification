import streamlit as st 
from PIL import Image
#from classify import predict
from skull import skull_s, classification, segmentation, detection
import cv2

## Introduction
st.title("Brain Tumor Detection-Segmentation and Classification 🧠")
st.write("This project will help to identify the tumour and segment tumor region from the MRI image. When the tumour is detected this project will also classify the tumor into either of the three categories: ")
st.write("1. Meningioma tumour")
st.write("2. Pituitary tumour")
st.write("3. Glioma tumour")

## File Uploading
uploaded_file = st.file_uploader("Upload the MRI Image.", type=["jpg","png"])
if uploaded_file is not None:
    im = Image.open(uploaded_file)
    
    st.image(im, caption='Uploaded Image.', use_column_width=True, width=100)
    st.write("")
    st.write("Uploaded")
    

## Skull stripping
    st.header("Extracting the Brain Image by Skull Stripping")
   
    sk = skull_s(im)
    st.image(sk,caption='skull stripped image', use_column_width=True,width=100)


## Detection
    st.header("Brain tumour detection status:")
    det = detection(im)
    #result = st.button("Detect tumor", on_click=detection(im), disabled=False)
    
    if det:
      st.header("Type of tumour detected after Classification:")
      #result2 = st.button("Classify Tumor", on_click=classification(im), disabled=False)
      # st.write(result2)
      classification(im)

    ## Segmentation
    st.header("Segmentation of MRI Image (Under working)")
    # seg = st.button("Click for tumor Segmentation", on_click=segmentation(im))
    #seg = st.button("Segement tumor")
    # st.image(seg)
