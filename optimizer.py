import wandb
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
from activations import sigmoid, tanh, relu, softmax, derivative
from weights import weight_init
from network import forward_prop, backward_prop
from loss import CrossEntropy, squarederror

class stochastic_gd():
  def __init__(self,grads, eta, max_epochs,layers,mini_batch_size,lambd,parameters,i,t):
    self.grads=grads
    self.eta=eta
    self.layers=layers
    self.mini_batch_size=mini_batch_size
    self.parameters=parameters
    self.lambd=lambd
    self.i=i
    self.t=t

  def get_params(self):
    for j in range(len(self.layers)-1,0,-1):
        self.parameters["w"+str(j)] = (1-((self.eta*self.lambd)/self.mini_batch_size))*self.parameters["w"+str(j)] - (self.eta * self.grads["dw"+str(j)])
        self.parameters["b"+str(j)] = self.parameters["b"+str(j)] - (self.eta * self.grads["db"+str(j)])
    return self.parameters
class momentum_gd():
  def __init__(self,grads, eta, max_epochs,layers,mini_batch_size,lambd,parameters,i,t):
    self.grads=grads
    self.eta=eta
    self.layers=layers
    self.mini_batch_size=mini_batch_size
    self.parameters=parameters
    self.lambd=lambd
    self.gamma=0.9
    self.i=i
    self.t=t

  def get_update_history(self):
    update_history={}
    for j in range(len(self.layers)-1, 0, -1):
          update_history["w"+str(j)] = self.eta*self.grads["dw"+str(j)]
          update_history["b"+str(j)] = self.eta*self.grads["db"+str(j)]
    for j in range(len(self.layers)-1, 0, -1):
          update_history["w"+str(j)] = (self.gamma*update_history["w"+str(j)]) + (self.eta*self.grads["dw"+str(j)])
          update_history["b"+str(j)] = (self.gamma*update_history["b"+str(j)]) + (self.eta*self.grads["db"+str(j)])
    return update_history

  def get_params(self):
    update_history=self.get_update_history()
    for j in range(len(self.layers)-1,0,-1):
        self.parameters["w"+str(j)] = (1-((self.eta*self.lambd)/self.mini_batch_size))*self.parameters["w"+str(j)] - update_history["w"+str(j)]
        self.parameters["b"+str(j)] = self.parameters["b"+str(j)] - update_history["b"+str(j)]
    return self.parameters
class nesterovacc_gd():
  def __init__(self,grads, eta, max_epochs,layers,mini_batch_size,lambd,parameters,i,t):
    self.grads=grads
    self.eta=eta
    self.layers=layers
    self.mini_batch_size=mini_batch_size
    self.parameters=parameters
    self.lambd=lambd
    self.i=i
    self.t=t

  def paramlookahead(self):
    update_history={}
    if self.i==0:
        param_lookahead = copy.deepcopy(self.parameters)
    else:
        for j in range(len(self.layers)-1, 0, -1):
          param_lookahead['w'+str(j)] = self.parameters['w'+str(j)] + (self.gamma*update_history["w"+str(j)])
    return param_lookahead,update_history

  def get_params(self,update_history):
    param_lookahead,update_history=self.paramlookahead()
    if self.i == 0 :
        for j in range(len(self.layers)-1, 0, -1):
          update_history["w"+str(j)] = self.eta*self.grads["dw"+str(j)]
          update_history["b"+str(j)] = self.eta*self.grads["db"+str(j)]
    else:
        for j in range(len(self.layers)-1, 0, -1):
          update_history["w"+str(j)] = (self.gamma*update_history["w"+str(j)]) + (self.eta*self.grads["dw"+str(j)])
          update_history["b"+str(j)] = (self.gamma*update_history["b"+str(j)]) + (self.eta*self.grads["db"+str(j)])
    for j in range(len(self.layers)-1,0,-1):
        self.parameters["w"+str(j)] = (1-((self.eta*self.lambd)/self.mini_batch_size))*self.parameters["w"+str(j)] - update_history["w"+str(j)]
        self.parameters["b"+str(j)] = self.parameters["b"+str(j)] - update_history["b"+str(j)]
    return self.parameters

class rmsprop():
  def __init__(self,grads, eta, max_epochs,layers,mini_batch_size,lambd,parameters,i,t):
    self.grads=grads
    self.eta=eta
    self.layers=layers
    self.mini_batch_size=mini_batch_size
    self.parameters=parameters
    self.lambd=lambd
    self.i=i
    self.beta = 0.9 
    self.epsilon=1e-8
    self.t=t

  def momenta(self):
    update_history={}
    v={}
    for i in range(len(self.layers)-1,0,-1):
      update_history["w"+str(i)]=np.zeros((self.layers[i],self.layers[i-1]))
      update_history["b"+str(i)]=np.zeros((self.layers[i],1))
    for i in range(len(self.layers)-1,0,-1):
      v["w"+str(i)]=np.zeros((self.layers[i],self.layers[i-1]))
      v["b"+str(i)]=np.zeros((self.layers[i],1))
    return v,update_history
     
  def get_params(self):
    v,update_history=self.momenta()
    for iq in range(len(self.layers)-1,0,-1):
        v["w"+str(iq)]=self.beta*v["w"+str(iq)]+(1-self.beta)*self.grads["dw"+str(iq)]**2
        v["b"+str(iq)]=self.beta*v["b"+str(iq)]+(1-self.beta)*self.grads["db"+str(iq)]**2     
        update_history["w"+str(iq)]=self.eta*np.multiply(np.reciprocal(np.sqrt(v["w"+str(iq)]+self.epsilon)),self.grads["dw"+str(iq)])
        update_history["b"+str(iq)]=self.eta*np.multiply(np.reciprocal(np.sqrt(v["b"+str(iq)]+self.epsilon)),self.grads["db"+str(iq)])
    for j in range(len(self.layers)-1,0,-1):
        self.parameters["w"+str(j)] = (1-((self.eta*self.lambd)/self.mini_batch_size))*self.parameters["w"+str(j)] - update_history["w"+str(j)]
        self.parameters["b"+str(j)] = self.parameters["b"+str(j)] - update_history["b"+str(j)]
    return self.parameters

class adam():
  def __init__(self,grads, eta, max_epochs,layers,mini_batch_size,lambd,parameters,i,t):
    self.grads=grads
    self.eta=eta
    self.layers=layers
    self.mini_batch_size=mini_batch_size
    self.parameters=parameters
    self.lambd=lambd
    self.i=i
    self.beta1=0.9 
    self.epsilon=1e-8
    self.beta2=0.999
    self.t=t
  
  def momenta(self):
    update_history={}
    v={}
    m={}
    for i in range(len(self.layers)-1,0,-1):
      update_history["w"+str(i)]=np.zeros((self.layers[i],self.layers[i-1]))
      update_history["b"+str(i)]=np.zeros((self.layers[i],1))
    for i in range(len(self.layers)-1,0,-1):
      v["w"+str(i)]=np.zeros((self.layers[i],self.layers[i-1]))
      v["b"+str(i)]=np.zeros((self.layers[i],1))
    for i in range(len(self.layers)-1,0,-1):
      m["w"+str(i)]=np.zeros((self.layers[i],self.layers[i-1]))
      m["b"+str(i)]=np.zeros((self.layers[i],1))
    return m,v,update_history
     
  def get_params(self):
    m,v,update_history=self.momenta()
    for iq in range(len(self.layers)-1,0,-1):
          m["w"+str(iq)]=self.beta1*m["w"+str(iq)]+(1-self.beta1)*self.grads["dw"+str(iq)]
          m["b"+str(iq)]=self.beta1*m["b"+str(iq)]+(1-self.beta1)*self.grads["db"+str(iq)]    
          v["w"+str(iq)]=self.beta2*v["w"+str(iq)]+(1-self.beta2)*(self.grads["dw"+str(iq)])**2
          v["b"+str(iq)]=self.beta2*v["b"+str(iq)]+(1-self.beta2)*(self.grads["db"+str(iq)])**2
          mw_hat=m["w"+str(iq)]/(1-np.power(self.beta1,self.t+1))
          mb_hat=m["b"+str(iq)]/(1-np.power(self.beta1,self.t+1))
          vw_hat=v["w"+str(iq)]/(1-np.power(self.beta2,self.t+1))
          vb_hat=v["b"+str(iq)]/(1-np.power(self.beta2,self.t+1))
          update_history["w"+str(iq)]=self.eta*np.multiply(np.reciprocal(np.sqrt(vw_hat+self.epsilon)),mw_hat)
          update_history["b"+str(iq)]=self.eta*np.multiply(np.reciprocal(np.sqrt(vb_hat+self.epsilon)),mb_hat)

    for j in range(len(self.layers)-1,0,-1):
          self.parameters["w"+str(j)] = (1-((self.eta*self.lambd)/self.mini_batch_size))*self.parameters["w"+str(j)] - update_history["w"+str(j)]
          self.parameters["b"+str(j)] = self.parameters["b"+str(j)] - update_history["b"+str(j)]
    return self.parameters

class nadam():
  def __init__(self,grads, eta, max_epochs,layers,mini_batch_size,lambd,parameters,i,t):
    self.grads=grads
    self.eta=eta
    self.layers=layers
    self.mini_batch_size=mini_batch_size
    self.parameters=parameters
    self.lambd=lambd
    self.i=i
    self.beta1=0.9 
    self.epsilon=1e-8
    self.beta2=0.999
    self.t=t
  
  def momenta(self):
    update_history={}
    v={}
    m={}
    for i in range(len(self.layers)-1,0,-1):
      update_history["w"+str(i)]=np.zeros((self.layers[i],self.layers[i-1]))
      update_history["b"+str(i)]=np.zeros((self.layers[i],1))
    for i in range(len(self.layers)-1,0,-1):
      v["w"+str(i)]=np.zeros((self.layers[i],self.layers[i-1]))
      v["b"+str(i)]=np.zeros((self.layers[i],1))
    for i in range(len(self.layers)-1,0,-1):
      m["w"+str(i)]=np.zeros((self.layers[i],self.layers[i-1]))
      m["b"+str(i)]=np.zeros((self.layers[i],1))
    return m,v,update_history
     
  def get_params(self):
    m,v,update_history=self.momenta()
    for iq in range(len(self.layers)-1,0,-1):
          m["w"+str(iq)]=self.beta1*m["w"+str(iq)]+(1-self.beta1)*self.grads["dw"+str(iq)]
          m["b"+str(iq)]=self.beta1*m["b"+str(iq)]+(1-self.beta1)*self.grads["db"+str(iq)]    
          v["w"+str(iq)]=self.beta2*v["w"+str(iq)]+(1-self.beta2)*(self.grads["dw"+str(iq)])**2
          v["b"+str(iq)]=self.beta2*v["b"+str(iq)]+(1-self.beta2)*(self.grads["db"+str(iq)])**2
          mw_hat=m["w"+str(iq)]/(1-np.power(self.beta1,self.t+1))
          mb_hat=m["b"+str(iq)]/(1-np.power(self.beta1,self.t+1))
          vw_hat=v["w"+str(iq)]/(1-np.power(self.beta2,self.t+1))
          vb_hat=v["b"+str(iq)]/(1-np.power(self.beta2,self.t+1))
          update_history["w"+str(iq)]=self.eta*np.multiply(np.reciprocal(np.sqrt(vw_hat+self.epsilon)),(self.beta1*mw_hat+(1-self.beta1)*self.grads["dw"+str(iq)]))*(1/(1-np.power(self.beta1,self.t+1)))
          update_history["b"+str(iq)]=self.eta*np.multiply(np.reciprocal(np.sqrt(vb_hat+self.epsilon)),(self.beta1*mb_hat+(1-self.beta1)*self.grads["db"+str(iq)]))*(1/(1-np.power(self.beta1,self.t+1)))

    for j in range(len(self.layers)-1,0,-1):
          self.parameters["w"+str(j)] = (1-((self.eta*self.lambd)/self.mini_batch_size))*self.parameters["w"+str(j)] - update_history["w"+str(j)]
          self.parameters["b"+str(j)] = self.parameters["b"+str(j)] - update_history["b"+str(j)]

    return self.parameters