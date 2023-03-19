import wandb
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
from activations import sigmoid, tanh, relu, softmax, derivative
from weights import weight_init
from optimizers import stochastic_gd, momentum_gd, nesterovacc_gd, rmsprop, adam, nadam
from loss import CrossEntropy, squarederror

''' Loss functions 
Squared loss
CrossEntropy

loss_calc(loss_name, y, y_hat, lambd, layers, parameters)

L2 regularization is added, it is added to the existing loss function chosen 
The value of lambda is again a hyperparameter and can be changed accordingly 
'''

def squared_loss(y, y_hat):
  error = np.sum(((y - y_hat)**2) / (2 * len(y)))
  return error

def CrossEntropy(y, y_hat):
  error = - np.sum( np.multiply(y , np.log(y_hat)))/len(y)
  return error

def loss_calc(loss_name, y, y_hat, lambd, layers, parameters):
  loss=0
  if(loss_name == "squared_loss"):
    loss=squared_loss(y, y_hat)
  elif(loss_name == "cross_entropy"):
    loss= CrossEntropy(y, y_hat)

  reg_loss = 0.0
  for i in range(len(layers)-1, 0, -1):
    reg_loss = reg_loss + (np.sum(parameters["w"+str(i)]))**2
  reg_loss = loss + ((lambd/(2*len(y)))*(reg_loss))
  return reg_loss

def calculate_grad(X, Y, parameters, activation, layers, loss_function):
  grads={}
  grads.clear() 
  for j in range(len(X)):
    y = np.reshape(Y[j], (-1,1))

    nn=network(X[j], y, parameters, activation, layers, loss_function)
    h,a,y_hat = nn.forward_prop()
    new_grads = nn.backward_prop(y,y_hat,h,a,parameters,layers)

    if j == 0:
      grads = copy.deepcopy(new_grads)
    else:
      for k in range(len(layers)-1,0,-1):
        grads["dw"+str(k)] += new_grads["dw"+str(k)]
        grads["db"+str(k)] += new_grads["db"+str(k)]
  return grads

''' Following function performs gradient descent for all the layers.
Arguments: 
X_train, y_train, eta, max_epochs, layers, mini_batch_size, lambd,loss_function, activation, parameters,optimiser,wandb_log

The function finds derivatives per layer and updates the weights and biases accordingly 

Optimisers supported:
SGD
NAG
Momentumgd
adam
nadam
rmsprop
'''

def gradient_descent(X_train, y_train, eta, max_epochs, layers, mini_batch_size, lambd,loss_function, activation, parameters,optimiser,wandb_log=False):
  grads={}
  train_loss = []
  val_loss = []
  train_acc = []
  val_acc = []

  for t in tqdm(range(max_epochs)):
    for i in range(0, len(X_train), mini_batch_size):

      grads.clear()

      if str(optimiser) == "nesterovacc_gd":
        opt=optimiser(grads, eta, max_epochs,layers,mini_batch_size,lambd,parameters,i,t)
        param_lookahead,update_history=opt.paramlookahead()

      X = X_train[i:i + mini_batch_size]
      Y = y_train[i:i + mini_batch_size]
      
      if str(optimiser) == "nesterovacc_gd":
        grads = calculate_grad(X,Y,param_lookahead,activation,layers,loss_function)
      else: 
        grads = calculate_grad(X,Y,parameters,activation,layers,loss_function)

      opt=optimiser(grads, eta, max_epochs,layers,mini_batch_size,lambd,parameters,i,t)
      parameters=opt.get_params()
    
    #Calculating train loss 
    res = run_inference(X_train,y_train,parameters, activation, layers)
    train_err = loss_calc(loss_function,y_train,res,lambd,layers,parameters) 
    train_ac=accuracy_calc(res, y_train)
    train_loss.append(train_err)
    train_acc.append(train_ac)

    #Calculating validation loss
    res = run_inference(X_val, y_val, parameters, activation, layers)
    val_err = loss_calc(loss_function, y_val, res, lambd, layers, parameters )
    val_ac=accuracy_calc(res, y_val)
    val_loss.append(val_err)
    val_acc.append(val_ac)

    if(wandb_log==True):
      log_dict = {"Train_Accuracy": train_ac, "Validation_Accuracy": val_ac, \
                  "Train_Loss": train_err, "Validation_loss": val_err, "epoch": t}
                  
      wandb.log(log_dict)

  return parameters, train_acc, val_acc