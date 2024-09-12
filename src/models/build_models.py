from sklearn.cluster import KMeans, AgglomerativeClustering
import time
import numpy as np
from PIL import Image

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
    
def build_agglomerative(img_flat, img_np, n, linkage):
    start = time.time()
    model = AgglomerativeClustering(n_clusters=n, linkage=linkage)
    model.fit(img_flat)
    end = time.time()
    elapsed = round(end-start,4)

    

    labels = model.predict(img_flat)
