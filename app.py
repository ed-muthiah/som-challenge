import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from som import SOM
import streamlit as st
import time
import os

# Custom imports 
from multipage import MultiPage
from pages import background, implementation, service

# Create an instance of the app 
app = MultiPage()

# Add all your application here
app.add_page("Background", background.app)
app.add_page("Implementation", implementation.app)
app.add_page("SOM", service.app)

# The main app
app.run()