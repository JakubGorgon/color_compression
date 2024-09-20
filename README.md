# Machine Learning for Color Compression 🎨

This Streamlit app allows you to compress the color palette of images using several unsupervised clustering algorithms, such as K-means, Bisecting K-means, Mini Batch K-means, and Fuzzy C-means. Users can upload an image, select an algorithm, tune its hyperparameters, and compare the original and compressed images in 3D visualizations.

## File Structure
- `app.py`: Main Streamlit app file.
- `src/`: Contains core logic for clustering algorithms and data transformations.
  - `src/models/`: Implements the different clustering algorithms.
  - `src/data/`: Handles image transformation from pixel format to tabular data and other data manipulation operations.
  - `src/visualization/`: Manages the 3D scatter plots.
- `styles.py`: Custom CSS to style the app.