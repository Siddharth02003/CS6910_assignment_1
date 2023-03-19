
import numpy as np


''' The activation class houses all the activation functions and their corresponding derivatives
Functions:

Sigmoid
Relu
Tanh
Softmax (final layer activation)

Sigmoid derivative
Relu derivative
Tanh derivative
Softmax derivative

Derivative function to calculate gradient 
'''

class activation(): 
  def __init__(self,a):
    self.a=a

  def sigmoid(self,a):
    try:
      return (1.0/(1.0+np.exp(-a)))
    except:
      print("error")

  def relu(self,a):
    return (np.maximum(0,a))

  def tanh(self,a):
    return np.tanh(a)

  def softmax(self,a):
    try:
      return(np.exp(a)/np.sum(np.exp(a)))
    except:
      print("error")

  def sigmoid_derivative(self,x):
    return self.sigmoid(x)*(1-self.sigmoid(x))

  def tanh_derivative(self,x):
    return 1.0 -self.tanh(x)**2

  def relu_derivative(self,x):
    return 1. * (x>0)
     
  def softmax_derivative(self,x):
    return self.softmax(x) * (1-self.softmax(x))

  def derivative(self,x,activation):
    if activation == "sigmoid":
      return self.sigmoid_derivative(x)
    elif activation == "tanh":
      return self.tanh_derivative(x)
    elif activation == "relu":
      return self.relu_derivative(x)