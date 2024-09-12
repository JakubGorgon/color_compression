import pandas as pd
import numpy as np
import streamlit as st; import streamlit_ext as ste
import matplotlib.pyplot as plt
from PIL import Image
import io

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
        color: #378200; 
        text-align:left;
        font-weight: bold;
    }
    .compressed-img-title {
        font-size:24px;
        color: #378200; 
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
        img_flat = img_tab.values
       
        return img, img_np, img_flat


if uploaded_file is not None:
    img, img_np, img_flat = img_to_tabular(uploaded_file)
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
            elapsed, inertia, cluster_centers, iters, compressed_image, compressed_image_pil = build_kmeans(img_flat=img_flat,
                                                                img_np=img_np,
                                                                n=n_number, 
                                                                max_iter=max_iter_slider, 
                                                                algo=algo_radio)
        
        st.markdown(f'<p class="clustering-results">Results of {clustering_chosen} Clustering:</p>', unsafe_allow_html=True)
        st.markdown(f"""
                        * **Number of Clusters**: {n_number}
                        * **Inertia**: {inertia}
                        * **Iterations**: {iters}
                        * **Time taken**: {elapsed} seconds
                        """)
        
        buffer = io.BytesIO()
        compressed_image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        
        st.markdown(f'<p class="compressed-img-title">Your image after applying {clustering_chosen}</p>', unsafe_allow_html=True)
        st.image(buffer, width=750)
        ste.download_button("Download ", 
                           data=buffer,
                           file_name= f"{clustering_chosen}_color_compressed.png",
                           mime='image/png')
