styles = """
    <style>
    .main-header {
        font-size: 50px;
        color: #4B0082;
        text-align: center;
        font-weight: bold;
        margin-bottom: 15px;
        text-shadow: 3px 3px 10px rgba(0, 0, 0, 0.3);
        letter-spacing: 1.5px;
    }
    .sub-header {
        font-size: 26px;
        color: #8A2BE2;
        text-align: center;
        font-weight: 500;
        margin-top: 0px;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    .clustering-results {
        font-size: 38px;
        color: #378200;
        text-align: left;
        font-weight: bold;
        margin-bottom: 10px;
        border-left: 5px solid #378200;
        padding-left: 15px;
        opacity: 0; /* Start hidden */
        animation: fadeIn 1.5s ease-in-out forwards; /* Fade-in effect */
    }
    .cluster-info {
        font-size: 22px;
        color: #333;
        text-align: left;
        margin-bottom: 5px;
        line-height: 1.6;
        opacity: 0; /* Start hidden */
        animation: fadeIn 1.5s ease-in-out forwards; /* Fade-in effect */
    }
    .compressed-img-title {
        font-size: 30px;
        color: #378200;
        text-align: left;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
        padding-bottom: 5px;
        opacity: 0; /* Start hidden */
        animation: fadeIn 1.5s ease-in-out forwards; /* Fade-in effect */
    }
    
    /* Style for Streamlit's download buttons */
    .stDownloadButton {
        margin-top: 0px; 
        margin-bottom: 0px; 
        display: inline-block; 
        color: white; 
        border-radius: 4px; 
        opacity: 0; /* Start hidden */
        animation: fadeIn 1.5s ease-in-out forwards; /* Fade-in effect */
    }
        
    img {
        border: 2px solid #378200;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 5px;
        opacity: 0; /* Start hidden */
        max-height: 400px;
        aspect-ratio: auto;
        animation: fadeIn 1.5s ease-in-out forwards; /* Fade-in effect */
    }
    
    /* Specific to Streamlit image containers */
    .stImage img {
        width: 100%;
        height: auto;
        max-height: 10px;
        border: 2px solid #378200;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        opacity: 0; /* Start hidden */
        animation: fadeIn 1.5s ease-in-out forwards; /* Fade-in effect */
    }

    /* Keyframes for fade-in animation */
    @keyframes fadeIn {
        0% {
            opacity: 0; /* Fully transparent */
            transform: translateY(10px); /* Slight upward movement */
        }
        100% {
            opacity: 1; /* Fully visible */
            transform: translateY(0); /* Normal position */
        }
    }
    
    html {
        scroll-behavior: smooth;
    }
    
    </style>

"""
