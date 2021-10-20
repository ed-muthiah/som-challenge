import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from som import SOM
import time
from google_images_download import google_images_download
import glob
import cv2 as cv2


keyword = "downloads"
path = glob.glob(f"./google-image-download/{keyword}/*")
cv_img = None
count = 0
for img in path:
    if count == 0:
        new = cv2.imread(img)
        new_rgb = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
        new_rgb = cv2.resize(new_rgb, (300,300), interpolation = cv2.INTER_AREA)
        cv_img = new_rgb
    elif count > 4:
        break
    else:
        new = cv2.imread(img)
        new_rgb = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
        new_rgb = cv2.resize(new_rgb, (300,300), interpolation = cv2.INTER_AREA)
        cv_img = np.concatenate((cv_img, new_rgb), axis = 1)
    count += 1

plt_img = cv_img/255
plt.imshow(plt_img)
plt.show()


@st.cache
def run_model(img, size=100, number_iterations=1000, initial_learningrate=0.1):
    image_vector = img.reshape(-1,3)
    som = SOM(size,size)
    som.train(image_vector, number_iterations=number_iterations, initial_learningrate=initial_learningrate)
    output = som.get_image(som.weights, size, 3)
    return output


%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.imshow(output)
    np.random.seed(seed=42)
    input_data = np.random.random((100,3))
    som = SOM(size,size)
    som.train(input_data, number_iterations=number_iterations,initial_learningrate=initial_learningrate)
    img = som.get_image(som.weights, size, 3)
    return img
    
def get_image(keyword):
    response = google_images_download.googleimagesdownload()
    paths = response.download({"keywords":f"{keyword}","limit":5, "size":"medium", "format":"jpg"})

def show_image(keyword):
    keyword = "downloads"
    path = glob.glob(f"./google-image-download/downloads/{keyword}/*")
    cv_img = None
    count = 0
    for img in path:
        if count == 0:
            new = cv2.imread(img)
            new_rgb = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
            new_rgb = cv2.resize(new_rgb, (300,300), interpolation = cv2.INTER_AREA)
            cv_img = new_rgb
        elif count > 4:
            break
        else:
            new = cv2.imread(img)
            new_rgb = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
            new_rgb = cv2.resize(new_rgb, (300,300), interpolation = cv2.INTER_AREA)
            cv_img = np.concatenate((cv_img, new_rgb), axis = 1)
        count += 1

    return plt_img = cv_img/255


    

def app():
    st.title("SOM-as-a-Service")
    st.markdown("## What colour do we associate with different items?")
    
    keyword = st.text_input("What's the colour of...") 

    if keyword:
        get_image(keyword)
        img = show_image(keyword)
        fig = plt.figure()
        fig.suptitle("Sample Images", fontsize=20)
        plt.imshow(img)        
        st.pyplot(fig)
        st.markdown("### Your search's colour space!")
        output_img = run_model(img)
        fig2 = plt.figure(figsize=(14,8))
        plt.imshow(output_img)        
        st.pyplot(fig2)