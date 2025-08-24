import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import time

data = pd.read_csv('iris.data', header=None)
X = data.iloc[:, :-1].values  # all columns except label


initial_centroids = np.loadtxt('centroids.txt', delimiter=',')

start = time.time()
kmeans = KMeans(
    n_clusters=3,
    init=initial_centroids,
    n_init=1  
)


kmeans.fit(X)
end = time.time()
np.set_printoptions(precision=4, suppress=True)

print("Cluster Centers:")
print(kmeans.cluster_centers_)
print("Runtime in seconds:", end - start)
