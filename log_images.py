

from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import copy 
from tqdm import tqdm
import random
import tensorflow as tf
import wandb

wandb login --relogin
entity_name="siddharth-s"
project_name="FODL_Assignment_1"
def load_dataset():
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    return {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test
        }
class_labels={0:'T-shirt',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle_Boot'}

def label_to_name(label):
  if enumerate(label):
    l_names=[]
    for l in label:
      l_names.append(class_labels[l])
    return l_names
  else:
    return class_labels[label]
def log_images():
  images=[]
  labels=[]
  dataset=load_dataset()
  X_train=dataset['X_train']
  y_train=dataset['Y_train']
  wandb.init(entity=entity_name,project=project_name, name="log_images")
  for i in range(100):
    if len(labels)==10:
      break
    if class_labels[y_train[i]] not in labels:
      images.append(X_train[i])
      labels.append(class_labels[y_train[i]])
  wandb.log({"Images": [wandb.Image(img, caption=caption) for img, caption in zip(images,labels)]})