from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
   
def plot_decision_regions(X,y,classifier,resolution = 0.02):
    #初始化 markers 和 color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # 绘制决策边界
    x1_min ,x1_max = X[:,0].min() -1,X[:,0].max() +1
    x2_min, x2_max = X[:,1].min() -1,X[:,1].max() +1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                          np.arange(x2_min,x2_max,resolution))
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T) 
    z = z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z, alpha = 0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    print(cmap)
    #绘制样本点
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl,0],y = X[y == cl,1], alpha = 0.8,
                    c = colors[idx], marker = markers[idx], label = cl)
                    