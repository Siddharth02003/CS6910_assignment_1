import random
import numpy as np

class weights():
  def __init__(self,layers):
    self.layers=layers

  def Xavier(self,layers):
    params = {}
    for i in range(1,len(layers)):
       norm_xav=np.sqrt(6)/np.sqrt(layers[i]+layers[i-1])
       params["w"+str(i)]=np.random.randn(layers[i],layers[i-1])*norm_xav
       params["b"+str(i)]=np.zeros((layers[i],1))
    return params

  def Random(self,layers):
    params = {}
    for i in range(1,len(layers)):
       params["w"+str(i)]=0.01*np.random.randn(layers[i],layers[i-1])
       params["b"+str(i)]=0.01*np.random.randn(layers[i],1)
    return params

  def weight_init(self,init_type = "random"):
    params={}
    if(init_type=="xavier"):
      params = self.Xavier(self.layers)
    elif(init_type=="random"):
      params = self.Random(self.layers)
    else:
      print("invalid activation function")
    return params