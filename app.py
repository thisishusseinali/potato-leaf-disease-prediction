import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from predictor import predict_class

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.image('header.png')

imageLocation = st.empty()
imageLocation.image('image.jpg')
img = st.file_uploader(label="potato leaf test", type=['jpeg', 'jpg', 'png'], key="xray")

if img is not None:
    imageLocation.image(img)
    loading_msg = st.empty()
    loading_msg.text("Predicting...")
    result, confidence = predict_class(img)
    st.write('Prediction : {}'.format(result))
    st.write('Confidence : {}%'.format(confidence))

st.image('footer.png')
