from PIL import Image
import numpy as np
import pandas as pd

def img_to_tabular(img):
    # Check if the image is already a PIL image
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    
    # Convert to RGB mode if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert image to numpy array
    img_np = np.array(img)
    
    # Flatten the image and convert to a tabular format
    img_tab = pd.DataFrame(img_np.reshape(-1, 3), columns=['r', 'g', 'b'])
    img_flat = img_tab.values
    
    return img, img_np, img_flat, img_tab
    
def downsample_image(img, target_pixels = 10000):
    original_pixels = img.width * img.height
    
    # Check if downsampling is needed
    if original_pixels <= target_pixels:
        return img
    
    # Calculate downsample factor based on target size
    downsample_factor = int(np.sqrt(original_pixels / target_pixels))
    
    downsampled_img_pil = img.resize(
        (img.width // downsample_factor, img.height // downsample_factor)
    )
    
    return downsampled_img_pil
    
def sample_img(img_np_compressed, img_tab, n = 1000):
    img_tab_sample = img_tab.sample(n=n, random_state=42).sort_index()
    compressed_img_tab = pd.DataFrame(img_np_compressed.reshape(-1,3))
    img_tab_sample_compressed = compressed_img_tab.iloc[img_tab_sample.index]
    img_tab_sample_compressed.columns = ['r', 'g', 'b']
    return img_tab_sample, img_tab_sample_compressed # Returns a sample of n rows from the image and that same sample with compressed colors