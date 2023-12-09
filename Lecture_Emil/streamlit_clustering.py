import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Set the title of the app
st.title('2D Data Clustering Application')
st.header("Welcome to super duper clustering app! The best of its kind")
st.text("Simple textqsdasdasdw")


# Button to generate random data
if st.button('Generate Random Data'):
    # Generate random data with 4 clusters
    data, _ = make_blobs(n_samples=300, 
                         centers=4, 
                         cluster_std=0.60)
    df = pd.DataFrame(data, columns=['X', 'Y'])

    # Save the generated data to the session state
    st.session_state['df'] = df

    # Visualize the data
    st.subheader('Random 2D Data')
    fig, ax = plt.subplots()
    ax.scatter(df['X'], df['Y'])
    st.pyplot(fig)

# Perform K-means clustering
if st.button('Perform Clustering') and 'df' in st.session_state:
    df = st.session_state['df']
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(df)
    df['Cluster'] = kmeans.predict(df)

    # Visualize clustered data
    st.subheader('Clustered Data')
    fig, ax = plt.subplots()
    colors = ['blue', 'green', 'red', 'purple']
    for i in range(4):
        d = df[df['Cluster'] == i]
        ax.scatter(d['X'], d['Y'], c=colors[i], label=f'Cluster {i}')
    ax.legend()
    st.pyplot(fig)


st.image("irisFlower.jfif", caption = "THE BEST IMAGE IN THE WORLD")
st.text("END OF THE APP")