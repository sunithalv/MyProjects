#Library imports
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import array_to_img


#Loading the Model
gen_model = load_model('gen_model_pix2pix.h5')
gen_model.load_weights('model_pix2pix.h5')

#Setting Title of App
st.title("Map Generation")
st.markdown("Upload a satellite image of the location")

#Uploading the image
image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Generate')
#On predict button click
if submit:


    if image is not None:

        # Displaying the image
        st.image(image)
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        #st.write(opencv_image.shape)
        #Convert jpg file to tensor
        image_tf=tf.convert_to_tensor(tf.cast(opencv_image, tf.uint8))

        #Resizing the image
        img = tf.image.resize(image_tf, [256, 256])
        #Convert image to 4 Dimension
        img=tf.expand_dims(img, 0)
        #st.write(img.shape)
        #Generate image
        generated_img = gen_model(img/255.0, training=True)

        st.write('Generated map :')
        gen_img_array=np.array(generated_img)
        gen_img=array_to_img(gen_img_array[0])
        st.image(gen_img)
