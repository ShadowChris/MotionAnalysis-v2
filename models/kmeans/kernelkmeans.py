import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import pairwise, pairwise_kernels


def mypuri(classes):
    classesone=list(set(classes))
    dict={}
    sum=0
    for i in range(3):
        for j in classesone:
            dict[j]=0
        for j in range(i*50,i*50+50):
            for cl in classesone:
                if classes[j] == cl:
                    dict[cl] += 1
        sum+=max(dict.values())
        print(max(dict.values()))
    return sum/150
def get_keys(d, value):
    return [k for k,v in d.items() if v == value]
def ytrue(classes):
    classesone=list(set(classes))
    dict={}
    sum=0
    ytruels=[]
    for i in range(3):
        for j in classesone:
            dict[j]=0
        for j in range(i*50,i*50+50):
            for cl in classesone:
                if classes[j] == cl:
                    dict[cl] += 1
        key=get_keys(dict,max(dict.values()))
        #print('keys:')
        #print(key)
        key=key[0]
        for _ in range(50):
            ytruels.append(key)
    return ytruels


class Kmeans():
    def __init__(self, k=3, max_iterations=5000, varepsilon=0.0001,kernel=pairwise.linear_kernel):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
        self.kernel=kernel

    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def closest_centroid(self, sample, centroids):
        one_sample = sample.reshape(1, -1)
        X = centroids.reshape(centroids.shape[0], -1)
        distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
        closest_i = np.argmin(distances)
        return closest_i

    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for i in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self.closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def predict(self, X):
        centroids = self.init_random_centroids(X)
        for _ in range(self.max_iterations):
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids
            centroids = self.update_centroids(clusters, X)
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break
        return self.get_cluster_labels(clusters, X)





if __name__ == '__main__':
    idata = pd.read_csv('iris.data', header=None)
    idata.columns = ['s l', 's w', 'p l', 'p w', 'label']
    del idata['label']
    x = idata.values
    kernel = lambda X: pairwise_kernels(X, metric='additive_chi2')#
    z=kernel(x)
    print(z)
    kmeans=Kmeans()
    pred=kmeans.predict(z)
    print(pred)
    print('purity:')
    print(mypuri(pred))
    ytrue(pred)
    print('silhouette_score:')
    print(metrics.silhouette_score(idata, pred))
    print('ARI:')
    ari = metrics.adjusted_rand_score(ytrue(pred), pred)
    print(ari)
    x0 = x[pred == 0]
    plt.scatter(x0[:, 0], x0[:, 1])
    x1 = x[pred == 1]
    plt.scatter(x1[:, 0], x1[:, 1])
    x2 = x[pred == 2]
    plt.scatter(x2[:, 0], x2[:, 1])
    plt.ylabel('s w')
    plt.xlabel('s l')
    plt.show()



