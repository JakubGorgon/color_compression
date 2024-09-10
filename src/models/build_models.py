from sklearn.cluster import KMeans

def build_kmeans(img, n, max_iter, algo):
    model = KMeans(n_clusters=n, max_iter=max_iter, algorithm=algo)
    model.fit(img)
    return model.cluster_centers_
    