import wandb
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
from activations import sigmoid, tanh, relu, softmax, derivative
from weights import weight_init
from loss import CrossEntropy, squarederror

'''Network class: performs forward and backward pass 

Arguments: X,y,params,active,layers,loss_type 

forward_prop
backward_prop
'''

class network():
  def __init__(self,X,y,params,active,layers,loss_type):
    self.X=X
    self.y=y
    self.params=params
    self.active=active
    self.layers=layers
    self.loss_type=loss_type

  def forward_prop(self):
   out=copy.deepcopy(self.X)
   out=out.reshape(-1,1)
   h=[out]
   a=[out] 

   act=activation(a)

   if(self.active=="sigmoid"):
     for i in range(1,len(self.layers)-1):
       weights = self.params["w"+str(i)]
       biases = self.params["b"+str(i)]

       out = np.dot(weights,h[i-1])+biases
       a.append(out)
       post_a = act.sigmoid(out)
       h.append(post_a)
  
   elif(self.active=="tanh"):
     for i in range(1,len(self.layers)-1):
       weights=self.params["w"+str(i)]
       biases=self.params["b"+str(i)]
      
       out=np.dot(weights,h[i-1])+biases
       a.append(out)
       post_a=act.tanh(out)
       h.append(post_a)
  
   elif(self.active=="relu"):
     for i in range(1,len(self.layers)-1):
       weights=self.params["w"+str(i)]
       biases=self.params["b"+str(i)]
      
       out=np.dot(weights,h[i-1])+biases
       a.append(out)
       post_a=act.relu(out)
       h.append(post_a)       

   else:
     print("Invalid activation function") 
   weights=self.params["w"+str(len(self.layers)-1)]
   biases=self.params["b"+str(len(self.layers)-1)]
  
   out=np.dot(weights,h[len(self.layers)-2])+biases
   a.append(out)
   y_hat=act.softmax(out)
   h.append(y_hat)
   return h,a,y_hat

  def backward_prop(self,y,y_hat,h,a,params,layers):
    grad = {}
    act=activation(self.active)
    if self.loss_type == "squared_loss":
      grad["dh"+str(len(layers)-1)] = (y_hat - y)
      grad["da"+str(len(layers)-1)] = (y_hat - y) * act.softmax_derivative(a[len(layers)-1])

    elif self.loss_type == 'cross_entropy':
      grad["da"+str(len(layers)-1)] = -(y-y_hat)
      grad["dh"+str(len(layers)-1)] = -(y/y_hat)

    for i in range(len(layers)-1, 0, -1 ):
      grad["dw"+str(i)] = np.dot(grad["da"+str(i)], np.transpose(h[i-1]))
      grad["db"+str(i)] = grad["da"+str(i)]
      if i > 1:
        grad["dh"+str(i-1)] = np.dot(np.transpose(params["w"+str(i)]), grad["da"+str(i)])
        grad["da"+str(i-1)] = np.multiply(grad["dh" + str(i-1)], act.derivative(a[i-1],self.active))
    return grad