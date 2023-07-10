#Library imports
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.layers import Lambda, Input, GlobalAveragePooling2D
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import pickle
import cv2

# Load the breed names dictionary back from the pickle file.
breed_names = pickle.load(open("breed_names.pkl", "rb"))


#Loading the Models
#feature_extractor=load_model('model_dogbreedClassfn.h5')
classfn_model = load_model('model_dogbreedClassfn.h5')

#Setting Title of App
st.title("Dog Breed Classification")
st.markdown("Upload image of dog to predict its breed")

#Uploading the image
image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')
#On predict button click
if submit:
    #Extract test data features.
    img_size = (331,331)

    def get_features(model_name, model_preprocessor, input_size, data):

        input_layer = Input(input_size)
        preprocessor = Lambda(model_preprocessor)(input_layer)
        base_model = model_name(weights='imagenet', include_top=False,
                                input_shape=input_size)(preprocessor)
        avg = GlobalAveragePooling2D()(base_model)
        feature_extractor = Model(inputs = input_layer, outputs = avg)

        #Extract feature.
        feature_maps = feature_extractor.predict(data, verbose=1)
        print('Feature maps shape: ', feature_maps.shape)
        return feature_maps


    if image is not None:

        # Displaying the image
        st.image(image)
        img = Image.open(image) 
        # file_bytes  = np.asarray(bytearray(image.read()), dtype=np.uint8)
        # opencv_image = cv2.imdecode(file_bytes, 1)
        #st.write(opencv_image.shape)
        #Resizing the image
        img = img.resize(img_size)
        #img = np.resize(opencv_image, img_size)
        img_arr=np.asarray(img)
        #Convert image to 4 Dimension
        img_arr=np.expand_dims(img_arr, 0)
        #st.write(img_arr.shape)
        inception_features = get_features(InceptionV3,
                                        preprocess_input,
                                        (331,331,3), img_arr)
        
        #Make Prediction
        Y_pred = classfn_model.predict(inception_features)
        result = breed_names[np.argmax(Y_pred)]
        st.title(str("The dog is of breed " + result))

