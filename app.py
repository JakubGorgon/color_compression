import pandas as pd
import numpy as np
import streamlit as st; import streamlit_ext as ste
import matplotlib.pyplot as plt
from PIL import Image
import io

from styles import styles
from src.models.build_models import build_kmeans, build_bisecting_kmeans, build_mini_batch_kmeans, build_fuzzy_cmeans
from src.data.data_transformations import img_to_tabular, sample_img
from src.visualization.visualize import plot_3d_scatter, plot_3d_scatter_compressed

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
                                            ["K-means", "Bisecting K-means", "Mini Batch K-means", "Fuzzy C-means"]
                                            )
    if clustering_chosen is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"#### 3. **Tune {clustering_chosen} hyperparameters**")

    with st.sidebar.form(key='cluster_parameters_form', clear_on_submit=False):
        if clustering_chosen == 'K-means':
            n_number = st.number_input("Number of clusters to form", 
                                            1, 100, value=4)
            max_iter_slider = st.slider("Maximum number of iterations",
                                            1, 1000, value = 300)
            algo_radio = st.radio("K-means algorithm to use",
                                        ("lloyd", 'elkan'))
        
        if clustering_chosen == 'Bisecting K-means':
            n_number = st.number_input("Number of clusters to form", 
                                            1, 100, value=4)
            initialization_number = st.number_input("Number of time the inner k-means algorithm will be run with different centroid seeds in each bisection.",
                                                    1,10, value=1)        
            initialization_method_radio = st.radio(label = "Method for initialization",
                                        options =("k-means++", "random"),
                                            index=1)
            bisecting_strategy_radio = st.radio("How bisection should be performed",
                                                        ("biggest_inertia", "largest_cluster"))
            
        if clustering_chosen == 'Mini Batch K-means':
            n_number = st.number_input("Number of clusters to form", 
                                            1, 100, value=4)
            batch_size_number_input = st.slider("Size of the mini batches", min_value=2, max_value=8192, value= 1024, step=1)
            reassignment_ratio_slider = st.slider("Control the fraction of the maximum number of counts for a center to be reassigned",
                                                        min_value=0.00,max_value=1.00,step=0.01, value=0.01)
            
        if clustering_chosen == 'Fuzzy C-means':
            n_number = st.number_input("Number of clusters to form", 
                                            1, 100, value=4)
            fuzzifier_slider = st.slider(label = "Specify the degree of fuzziness in the fuzzy algorithm",
                                                            min_value=1.01, 
                                                            max_value=10.0, 
                                                            value=2.0, 
                                                            step=0.01)
            max_iter_slider = st.slider(label="Maximum number of iterations allowed",
                                                min_value=1,
                                                max_value=1500, 
                                                step=1, 
                                                value=500)
            
        st.markdown("---")
        st.markdown(f"#### 4. **See results of your {clustering_chosen} color compression!**")
        
        scatter_3d_checkbox = st.checkbox("Show 3d scatterplots", True)
        
        cluster_button = st.form_submit_button(label="Compress Colors")

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
        if clustering_chosen == 'Fuzzy C-means':
            elapsed, iters, cluster_centers, compressed_image, compressed_image_pil = build_fuzzy_cmeans(img_flat=img_flat,
                                                                                                              img_np=img_np,
                                                                                                              n=n_number,
                                                                                                              fuzzifier=fuzzifier_slider,
                                                                                                              max_iters=max_iter_slider)
        
        if scatter_3d_checkbox:
            img_tab_sample, img_tab_sample_compressed = sample_img(img_np_compressed=compressed_image, img_tab=img_tab)
            fig_before = plot_3d_scatter(df = img_tab_sample)
            fig_after = plot_3d_scatter_compressed(img_before=img_tab_sample, img_after= img_tab_sample_compressed)
        
        
        buffer = io.BytesIO()
        compressed_image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        
        st.session_state.results.append({
            'clustering_chosen': clustering_chosen,
            'n_number': n_number,
            'elapsed': elapsed,
            'compressed_image': compressed_image_pil,
            'original_image': img,
            'buffer': buffer,
            'fig_before': fig_before if scatter_3d_checkbox == True else None,
            'fig_after': fig_after if scatter_3d_checkbox == True else None,
            'metrics': {
                'inertia': inertia if clustering_chosen != 'Fuzzy C-means' else None,
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
            with st.expander(f"Results of {result['clustering_chosen']} ({idx+1}):", expanded=(idx == len(st.session_state.results) - 1), icon ="📋"):
                st.markdown(f"""
                    <ul class="cluster-info">
                    <li><strong>Number of Clusters:</strong> {result['n_number']}</li>
                    <li><strong>Time taken:</strong> {result['metrics']['time']:.2f} s</li>
                    {"<li><strong>Inertia:</strong> " + str(result['metrics']['inertia']) + "</li>" if result['metrics']['inertia'] is not None else ""}
                    {"<li><strong>Iterations:</strong> " + str(result['metrics']['iterations']) + "</li>" if result['metrics']['iterations'] is not None else ""}
                    </ul>
                """, unsafe_allow_html=True)

                if result['fig_before'] is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'<p class="compressed-img-title">Original Image</p>', unsafe_allow_html=True)
                        st.image(result['original_image'], width=600, caption=f"Before {result['clustering_chosen']}", use_column_width=True)  
                        
                        st.plotly_chart(result['fig_before'], use_container_width=True)
                    with col2:
                        st.markdown(f'<p class="compressed-img-title">Color Compressed Image</p>', unsafe_allow_html=True)
                        st.image(result['compressed_image'], width=600, caption=f"After {result['clustering_chosen']}", use_column_width=True)  # Display compressed image in the second column
                        
                        st.plotly_chart(result['fig_after'], use_container_width=True)
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'<p class="compressed-img-title">Original Image</p>', unsafe_allow_html=True)
                        st.image(result['original_image'], width=600, caption=f"Before {result['clustering_chosen']}", use_column_width=True)  
                    with col2:
                        st.markdown(f'<p class="compressed-img-title">Color Compressed Image</p>', unsafe_allow_html=True)
                        st.image(result['compressed_image'], width=600, caption=f"After {result['clustering_chosen']}", use_column_width=True)  # Display compressed image in the second column
                        ste.download_button("Download", 
                                            data=result['buffer'],
                                            file_name=f"{result['clustering_chosen']}_color_compressed.png",
                                            mime='image/png')


