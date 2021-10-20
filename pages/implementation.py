import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from som import SOM
import time

@st.cache
def run_model(size, number_iterations, initial_learningrate):
    np.random.seed(seed=42)
    input_data = np.random.random((100,3))
    som = SOM(size,size)
    som.train(input_data, number_iterations=number_iterations,initial_learningrate=initial_learningrate)
    img = som.get_image(som.weights, size, 3)
    return img

def app():
    st.title("Implemtation")
    st.markdown("## Have a go yourself!")
    
    size = st.slider(key='SIZE', label="Network Size", min_value=10.00, max_value=1000.00, step=10.00, value = 100.00)
    number_iterations = st.slider(key='ITERATIONS', label="Number of Iterations", min_value=100.00, max_value=1000.00, step=100.00, value = 100.00)
    initial_learningrate = st.slider(key='LR', label="Initial Learning rate", min_value=0.01, max_value=0.99, step=0.01, value = 0.1)
    
    img = run_model(int(size), int(number_iterations), initial_learningrate)
    input_img = np.random.random((int(size),int(size),3))
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(input_img)
    axarr[0].set_title('Input Array')
    axarr[1].imshow(img)
    axarr[1].set_title('Trained Array')       
    st.pyplot(f)