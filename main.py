from statistics import mode
from typing_extensions import Self
import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pandas import DataFrame

X_train, _ = make_blobs(n_samples=500, centers=3,
                        n_features=2, random_state=20)

df = DataFrame(dict(x=X_train[:, 0], y=X_train[:, 1]))
fig, ax = plt.subplots(figsize=(8, 8))
df.plot(ax=ax, kind='scatter', x='x', y='y')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show()


class Kmeans:
    def __init__(self, k =3):
        self.k = k
    
    def mes_dis(self,a,b):
        return np.sqrt(sum(np.square(a-b)))

    def assign(self,k, X, cg):
      
       cluster = [-1]*len(X)
       for i in range(len(X)):
           dist_arr = []
           for j in range(k):
              dist_arr.append(self.mes_dis(X[i], cg[j]))
           idx = np.argmin(dist_arr)
           cluster[i] = idx
         
       return np.asarray(cluster)


    def compute(self ,k, X, cluster):
 
        cg_arr = []
        for i in range(k):
            arr = []
            for j in range(len(X)):
                if cluster[j]==i:
                    arr.append(X[j])
            cg_arr.append(np.mean(arr, axis=0))
            
            
        return np.asarray(cg_arr)

    def changes(self,cg_prev, cg_new):
        res = 0
        for a,b in zip(cg_prev,cg_new):
          res+= self.mes_dis(a,b)
        return res   

    def predict(self,X):
        arr = []
        for i in range(self.k):
           cx1 = np.random.uniform(min(X[:,0]), max(X[:,0]))
           cx2 = np.random.uniform(min(X[:,1]), max(X[:,1]))
           arr.append([cx1, cx2])     
        cg_prev = np.asarray(arr)  
        

        cluster = [0]*len(X)
        cg_change = 100
        while cg_change>.001:
           cluster = self.assign(self.k, X, cg_prev)
           cg_new =  self.compute(self.k, X, cluster)
           cg_change = self.changes(cg_new, cg_prev)
           cg_prev = cg_new
        return cluster ,cg_prev



model = Kmeans()        
a= model.predict(X_train)

print(a)




   
def show_clusters(X, cluster, cg):
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=cluster))
    colors = {0:'blue', 1:'orange', 2:'green'}
    fig, ax = plt.subplots(figsize=(8, 8))
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    ax.scatter(cg[:, 0], cg[:, 1], marker='*', s=150, c='#ff2222')
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.show()

show_clusters(X_train,a[0] ,a[1])
        