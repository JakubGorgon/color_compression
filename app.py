import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from src.models.build_models import build_kmeans

st.set_page_config(layout="wide")

# Customizing the Header Section
st.markdown("""
    <style>
    .main-header {
        font-size:48px;
        color: #4B0082;
        text-align:center;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size:24px;
        color: #8A2BE2;
        text-align:center;
        font-weight: normal;
        margin-top: 0px;
    }
    .clustering-results {
        font-size:36px;
        color: #378200; /* Choose a bold and contrasting color */
        text-align:left;
        font-weight: bold;
        
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Machine Learning for Color Compression 🎨</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover, Use, Tune, and Compare Unsupervised Clustering Algorithms</p>', unsafe_allow_html=True)

st.markdown("---")

st.sidebar.title("⚙️ Control Panel")
st.sidebar.write("Upload your image, select clustering algorithm, tune parameters, and view results.")
st.sidebar.markdown("---")

st.write("""
### How it works:
1. **Upload your image**: Choose an image you'd like to compress.
2. **Choose a clustering algorithm**: Select an unsupervised clustering algorithm to compress the image's color palette.
3. **Tune the hyperparameters**: Adjust hyperparameters like the number of clusters or iterations.
4. **Compare Results**: View the compressed image and analyze the performance of your model.
""")

st.markdown("#### Ready to start? Upload your image and select your algorithm from the sidebar!")
st.markdown("---")
st.sidebar.markdown("#### 1. **Upload your image**")

uploaded_file = st.sidebar.file_uploader("Choose an image...", 
                                 type=["jpg", "jpeg", "png"])

def img_to_tabular(img):
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_np = np.array(img)
        img_tab = pd.DataFrame(img_np.reshape(-1, 3), columns=['r', 'g', 'b'])
        return img, img_tab


if uploaded_file is not None:
    img, img_tab = img_to_tabular(uploaded_file)
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 2. **Clustering algorithm**")
    clustering_chosen = st.sidebar.selectbox("Choose a clustering method...",
                                            ["K-means"],
                                            )

    if clustering_chosen is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"#### 3. **Tune {clustering_chosen} hyperparameters**")

    if clustering_chosen == 'K-means':
        n_number = st.sidebar.number_input("Number of clusters to form", 
                                        1, 100, value=8)
        max_iter_slider = st.sidebar.slider("Maximum number of iterations",
                                        1, 1000, value = 300)
        algo_radio = st.sidebar.radio("K-means algorithm to use",
                                    ("lloyd", 'elkan'))
        
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"#### 4. **See results of your {clustering_chosen} clustering!**")
    
    cluster_button = st.sidebar.button(label="Compress")

    if cluster_button:
        if clustering_chosen == 'K-means':
            build_kmeans(img=img_tab, n=n_number, 
                    max_iter=max_iter_slider, algo=algo_radio)
        
        st.markdown(f'<p class="clustering-results">Results of {clustering_chosen} Clustering</p>', unsafe_allow_html=True)
        st.markdown("""
                        * **Number of Clusters**: 5
                        * **Inertia**: 200.45
                        * **Iterations**: 10
                        * **Time taken**: 2.5 seconds
                        """)