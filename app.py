import streamlit as st

# Setting up the wide mode
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

    </style>
""", unsafe_allow_html=True)

# Adding Header and Subheader with a Sleek Design
st.markdown('<p class="main-header">Machine Learning for Color Compression 🎨</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover, Use, Tune, and Compare Unsupervised Clustering Algorithms</p>', unsafe_allow_html=True)

# Optional: Add a line divider for separation of sections
st.markdown("---")

st.sidebar.title("⚙️ Control Panel")
st.sidebar.write("Upload your image, select clustering algorithm, tune parameters, and view results.")

st.write("""
### How it works:
1. **Upload your image**: Choose an image you'd like to compress.
2. **Choose a clustering algorithm**: Select an unsupervised clustering algorithm to compress the image's color palette.
3. **Tune the parameters**: Adjust hyperparameters like the number of clusters or iterations.
4. **Compare Results**: View the compressed image and analyze the performance of your model.
""")

st.markdown("#### Ready to start? Upload your image and select your algorithm from the sidebar!")
st.markdown("---")
st.sidebar.markdown("#### 1. **Upload your image**")

uploaded_file = st.sidebar.file_uploader("Choose an image...", 
                                 type=["jpg", "jpeg", "png"])

    
