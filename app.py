import pandas as pd
import numpy as np
import streamlit as st; import streamlit_ext as ste
import matplotlib.pyplot as plt
from PIL import Image
import io

from styles import styles
from src.models.build_models import build_kmeans, build_bisecting_kmeans, build_mini_batch_kmeans
from src.data.data_transformations import img_to_tabular

st.set_page_config(layout="wide")

st.markdown(styles, unsafe_allow_html=True)

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

if 'results' not in st.session_state:
    st.session_state.results = []

if uploaded_file is not None:
    img, img_np, img_flat, img_tab = img_to_tabular(img=uploaded_file)
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 2. **Clustering algorithm**")
    clustering_chosen = st.sidebar.selectbox("Choose a clustering method...",
                                            ["K-means", "Bisecting K-means", "Mini Batch K-means"]
                                            )

    if clustering_chosen is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"#### 3. **Tune {clustering_chosen} hyperparameters**")

    if clustering_chosen == 'K-means':
        n_number = st.sidebar.number_input("Number of clusters to form", 
                                        1, 100, value=4)
        max_iter_slider = st.sidebar.slider("Maximum number of iterations",
                                        1, 1000, value = 300)
        algo_radio = st.sidebar.radio("K-means algorithm to use",
                                    ("lloyd", 'elkan'))
    
    if clustering_chosen == 'Bisecting K-means':
        n_number = st.sidebar.number_input("Number of clusters to form", 
                                        1, 100, value=4)
        initialization_number = st.sidebar.number_input("Number of time the inner k-means algorithm will be run with different centroid seeds in each bisection.",
                                                 1,10, value=1)        
        initialization_method_radio = st.sidebar.radio(label = "Method for initialization",
                                       options =("k-means++", "random"),
                                        index=1)
        bisecting_strategy_radio = st.sidebar.radio("How bisection should be performed",
                                                    ("biggest_inertia", "largest_cluster"))
        
    if clustering_chosen == 'Mini Batch K-means':
        n_number = st.sidebar.number_input("Number of clusters to form", 
                                        1, 100, value=4)
        batch_size_number_input = st.sidebar.slider("Size of the mini batches", min_value=2, max_value=8192, value= 1024, step=1)
        reassignment_ratio_slider = st.sidebar.slider("Control the fraction of the maximum number of counts for a center to be reassigned",
                                                      min_value=0.00,max_value=1.00,step=0.01, value=0.01)
        
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"#### 4. **See results of your {clustering_chosen} color compression!**")
    
    original_image_checkbox = st.sidebar.checkbox("Display original image", True)
    
    cluster_button = st.sidebar.button(label="Compress Colors")

    if cluster_button:
        if clustering_chosen == 'K-means':
            elapsed, inertia, cluster_centers, iters, compressed_image, compressed_image_pil = build_kmeans(img_flat=img_flat,
                                                                                                            img_np=img_np,
                                                                                                            n=n_number, 
                                                                                                            max_iter=max_iter_slider, 
                                                                                                            algo=algo_radio)       

        if clustering_chosen == 'Bisecting K-means':
            elapsed, inertia, cluster_centers, compressed_image, compressed_image_pil = build_bisecting_kmeans(img_flat=img_flat,
                                                                                                           img_np=img_np,
                                                                                                           n=n_number,
                                                                                                           initialization=initialization_method_radio,
                                                                                                           n_initialization=initialization_number,                                                                                            
                                                                                                           bisecting_strat=bisecting_strategy_radio)
        
        if clustering_chosen == "Mini Batch K-means":
            elapsed, inertia, iters, cluster_centers, compressed_image, compressed_image_pil = build_mini_batch_kmeans(img_flat = img_flat,
                                                                                                                   img_np = img_np,
                                                                                                                   n=n_number,
                                                                                                                   batch_size=batch_size_number_input,
                                                                                                                   reassignment_ratio=reassignment_ratio_slider)
        
        buffer = io.BytesIO()
        compressed_image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        
        st.session_state.results.append({
            'clustering_chosen': clustering_chosen,
            'n_number': n_number,
            'elapsed': elapsed,
            'compressed_image': compressed_image_pil,
            'original_image': img if original_image_checkbox else None,
            'buffer': buffer,
            'metrics': {
                'inertia': inertia,
                'iterations': iters if clustering_chosen != 'Bisecting K-means' else None,
                'time': elapsed,
            }
        })
        
    hide_img_fs = '''
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        '''
    st.markdown(hide_img_fs, unsafe_allow_html=True)
        

    if 'results' in st.session_state and len(st.session_state.results) > 0:
        for idx, result in enumerate(st.session_state.results):
            st.markdown(f'<p class="clustering-results">Results of {result['clustering_chosen']} ({idx+1}):</p>', unsafe_allow_html=True)
            st.markdown(f"""
                <ul class="cluster-info">
                <li><strong>Number of Clusters:</strong> {result['n_number']}</li>
                <li><strong>Time taken:</strong> {result['metrics']['time']:.2f} s</li>
                {"<li><strong>Inertia:</strong> " + str(result['metrics']['inertia']) + "</li>" if result['metrics']['inertia'] is not None else ""}
                {"<li><strong>Iterations:</strong> " + str(result['metrics']['iterations']) + "</li>" if result['metrics']['iterations'] is not None else ""}
                </ul>
            """, unsafe_allow_html=True)

            if result['original_image'] is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<p class="compressed-img-title">Original Image</p>', unsafe_allow_html=True)
                    st.image(result['original_image'], width=600, caption=f"Before {result['clustering_chosen']}")  
                with col2:
                    st.markdown(f'<p class="compressed-img-title">Color Compressed Image</p>', unsafe_allow_html=True)
                    st.image(result['compressed_image'], width=600, caption=f"After {result['clustering_chosen']}")  # Display compressed image in the second column
                    ste.download_button("Download ", 
                            data=result['buffer'],
                            file_name= f"{result['clustering_chosen']}_color_compressed.png",
                            mime='image/png')
            else:
                st.markdown(f'<p class="compressed-img-title">Your image after applying {result['clustering_chosen']}</p>', unsafe_allow_html=True)
                st.image(result['compressed_image'], width=750)  # Display compressed
                ste.download_button("Download ", 
                            data=result['buffer'],
                            file_name= f"{result['clustering_chosen']}_color_compressed.png",
                            mime='image/png')
            
            st.markdown("---")


            
