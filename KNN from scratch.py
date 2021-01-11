import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
dataset=datasets.load_iris()
x=dataset.data
y=dataset.target

from sklearn.model_selection import *
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3)

def distance(instance1,instance2):
    instance1=np.array(instance1)
    instance2=np.array(instance2)
    
    return np.sqrt(np.sum((instance1-instance2)**2))
    

def get_neighbors(training_x,labels,test_instance,k,distance=distance):
    m=len(training_x)
    neighbors=[]
    for index in range(m):
        dis=distance(training_x[index],test_instance)
        neighbors.append((training_x[index],dis,labels[index]))
    neighbors.sort(key=lambda x:x[1])
    return neighbors[:k]


from collections import Counter
def vote(neighbors):
    class_counter=Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]]+=1
    return class_counter.most_common(1)[0][0]


def vote_harmonic_weights(neighbors,all_results=True):
    class_counter=Counter()
    neighbor_index=len(neighbors)
    for index in range(neighbor_index):
        class_counter[neighbors[index][2]]+=1/(index+1)
    labels,votes=zip(*class_counter.most_common())
    winner=class_counter.most_common(1)[0][0]
    votes4winner=class_counter.most_common(1)[0][1]
    total=sum(class_counter.values(),0.0)
    
    if all_results:
        for key in class_counter:
            class_counter[key]/=total
        return winner,class_counter.most_common()
    else:
        return winner,votes4winner/sum(votes)
    
def vote_distance_weights(neighbors,all_results=True):
    class_counter=Counter()
    neighbor_index=len(neighbors)
    for index in range(neighbor_index):
        class_counter[neighbors[index][2]]+=(1/(class_counter[neighbors[index][1]]**2+1))
    labels,votes=zip(*class_counter.most_common())
    winner=class_counter.most_common(1)[0][0]
    votes4winner=class_counter.most_common(1)[0][1]
    total=sum(class_counter.values(),0.0)
    
    if all_results:
        for key in class_counter:
            class_counter[key]/=total
        return winner,class_counter.most_common()
    else:
        return winner,votes4winner/sum(votes)
    
  


def run_KNN(x,y,test_x,test_y):
    y_pred=[] 
    y_pd=[]  
    y_d=[]
    for i in range(len(test_x)):
        nb=get_neighbors(x,y,test_x[i],5)
        y_pred.append(vote(nb))
        h_w,_=vote_harmonic_weights(nb)
        h_d,_=vote_distance_weights(nb)
        y_pd.append(h_w)
        y_d.append(h_d)
    
    acc1=(sum(np.array(y_pred)==test_y)/len(test_y))*100    
    acc2=(sum(np.array(y_pd)==test_y)/len(test_y))*100    
    acc3=(sum(np.array(y_d)==test_y)/len(test_y))*100  
    
    return acc1,acc2,acc3,y_pd

acc1,acc2,acc3,ppp=run_KNN(x,y,test_x,test_y)

    
print('Accuracy1:',acc1)
print('Accuracy2:',acc2)    
print('Accuracy3:',acc3)    
    
    
    
    





