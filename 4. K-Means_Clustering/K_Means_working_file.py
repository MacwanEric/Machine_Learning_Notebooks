# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#Importing Libraries
import numpy as np 
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics


# %%
digits = load_digits()
data = scale(digits.data)

Y = digits.target

k = 10
#k = len(np.unique(Y))
samples, features = data.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(Y, estimator.labels_),
             metrics.completeness_score(Y, estimator.labels_),
             metrics.v_measure_score(Y, estimator.labels_),
             metrics.adjusted_rand_score(Y, estimator.labels_),
             metrics.adjusted_mutual_info_score(Y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))
clf = KMeans(n_clusters=k,init="k-means++",n_init=10)
bench_k_means(clf,"1",data)


# %%
digits = load_digits()
data = scale(digits.data)


# %%
data


# %%
from sklearn.datasets import load_digits  
digits = load_digits()  
print(digits.data.shape)  
(1797, 64)

import matplotlib.pyplot as plt #doctest: +SKIP  
for i in [0,1,2,3,4,5,6,7,8,9]:
    plt.gray() #doctest: +SKIP  
    plt.matshow(digits.images[i]) #doctest: +SKIP  
    plt.show() #doctest: +SKIP 


# %%
Y = digits.target
print(Y)

#k = 10
k = len(np.unique(Y))
print(k)
samples, features = data.shape
print(samples)
print(features)


# %%
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(Y, estimator.labels_),
             metrics.completeness_score(Y, estimator.labels_),
             metrics.v_measure_score(Y, estimator.labels_),
             metrics.adjusted_rand_score(Y, estimator.labels_),
             metrics.adjusted_mutual_info_score(Y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))


# %%
print(digits)

