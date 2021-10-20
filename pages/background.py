import streamlit as st

def app():
    st.title("Self Organising Maps")
    st.markdown("## What's a Self Organising Map?")
    st.markdown("A self-organizing map (SOM) is a type of artificial neural network (ANN) that is trained using unsupervised learning to produce a two-dimensional, discretized representation of the data. It is a method to do dimensionality reduction. Compared to standard clustering alogrithms like k-means, SOMs use a neighborhood function to preserve the topological properties of the input space.")
    st.image("./assets/diagram.png")
    st.markdown("## So how does it work?")
    st.markdown("The SOM algorithm involves iterating through sample input vectors, identifying the best matching unit, then calculating a weight update. The strength and extent of the updates decay over time.")
    st.image("./assets/flowchart.png")
    st.markdown("""
    Given the simplicity and interpritability of the algorithm, SOMs have been applied to many problems. These include:
    - visualisation   
    - data mining
    - colour space mapping
    - colour space quantisation
    - class assignment
    - outlier detection
    """)
    st.markdown("## But enough theory! Let's see some code [Click here](https://www.gituhb.com/ed-muthiah) :point_left: to see my implementation of a simple Self Organising Map")