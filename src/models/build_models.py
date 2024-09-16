import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans
import time
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin

from src.data.data_transformations import downsample_image, img_to_tabular

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
    
def build_agglomerative(img, n):
    downsampled_img = downsample_image(img, target_pixels=10000)
    img, img_np, img_flat, img_tab = img_to_tabular(downsampled_img)
    
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
    
    color_compressed_img_flat = cluster_centers[labels] # REPLACE EACH OBSERVATION WITH VALUES OF THE NEAREST CLUSTER CENTROID

    color_compressed_img_np = color_compressed_img_flat.reshape(img_np.shape)
    
    color_compressed_img_pil = Image.fromarray(color_compressed_img_np.astype('uint8'))
    
    return color_compressed_img_pil, elapsed

def build_bisecting_kmeans(img_flat, img_np,n, initialization, n_initialization, bisecting_strat):
    start = time.time()
    model = BisectingKMeans(n_clusters=n, init=initialization,n_init=n_initialization, bisecting_strategy=bisecting_strat)
    model.fit(img_flat)
    end = time.time()
    elapsed = round(end-start,4)

    inertia = round(model.inertia_,2)
    cluster_centers = model.cluster_centers_

    labels = model.predict(img_flat)
    compressed_img = cluster_centers[labels]
    compressed_img = compressed_img.reshape(img_np.shape).astype(np.uint8)
    compressed_img_pil = Image.fromarray(compressed_img)

    return elapsed, inertia, cluster_centers, compressed_img, compressed_img_pil