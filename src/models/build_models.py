from sklearn.cluster import KMeans, AgglomerativeClustering
import time
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin


def build_kmeans(img_flat, img_np, n, max_iter, algo):
    start = time.time()
    model = KMeans(n_clusters=n, max_iter=max_iter, algorithm=algo)
    model.fit(img_flat)
    end = time.time()
    elapsed = round(end-start,4)

    inertia = round(model.inertia_,2)
    cluster_centers = model.cluster_centers_
    iters = model.n_iter_ 

    labels = model.predict(img_flat)
    compressed_img = cluster_centers[labels]
    compressed_img = compressed_img.reshape(img_np.shape).astype(np.uint8)
    compressed_img_pil = Image.fromarray(compressed_img)

    return elapsed, inertia, cluster_centers, iters, compressed_img, compressed_img_pil
    
def build_agglomerative(img_np, img_tab, n):
    if img_tab.shape[0] > 10000:
        img_flat = img_tab.sample(n=10000, random_state=1).sort_index().to_numpy()
    else:
        img_flat = img_tab.values
    
    start = time.time()
    model = AgglomerativeClustering(n_clusters=n)
    model.fit(img_flat)
    end = time.time()
    elapsed = round(end-start,4)

    labels = model.labels_

    df = pd.DataFrame(img_flat)
    df = pd.concat([df, pd.DataFrame(labels)], axis = 1)
    df.columns = ['r', 'g', 'b', 'label']

    cluster_centers = np.array(df.groupby(by='label')[['r', 'g', 'b']].mean())
    
    full_labels = pairwise_distances_argmin(img_tab.values, cluster_centers, metric='euclidean')
    compressed_image_data = cluster_centers[full_labels] # REPLACE EACH OBSERVATION WITH VALUES OF THE NEAREST CLUSTER CENTROID

    image_data_np = compressed_image_data.reshape(img_np.shape)
    
    compressed_img_pil = Image.fromarray(image_data_np.astype('uint8'))
    
    return compressed_img_pil, elapsed