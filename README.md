# CS6910_assignment-1
1. Clone this repository
   ```bash
   git clone https://github.com/Siddharth02003/CS6910_assignment_1.git
   ```
2. Install the required packages
   ```bash
   pip3 install -r requirements.txt
   ```
Peek inside the requirements file if you have everything already installed. Most of the dependencies are common libraries.

## Training the Neural Network 

```python
def train(X_train=X_train, y_train=y_train, layers=[784,16,10],wandb_log=True, learning_rate = 0.0001, initialization_type = "random", activation_function = "sigmoid", loss_function = "cross_entropy", mini_batch_Size = 32, max_epochs = 5, lambd = 0,optimization_function = adam)
```

The following model supports the following arguments (with their default values) 

- X_train=X_train
- y_train=y_train
- layers=[784,16,10]
- wandb_log=True
- learning_rate = 0.0001
- initialization_type = "random"
- activation_function = "sigmoid"
- loss_function = "cross_entropy"
- mini_batch_Size = 32
- max_epochs = 5
- lambd = 0
- optimization_function = adam 

It prints the validation and train accuracy upon completion of training.

Available functions to customize the network 

1. initialization_type 

```
Xavier()
Random()
```

2. activation_function

```
sigmoid()
tanh()
relu()
```

3. loss_function

```
CrossEntropy()
squaredloss()
```
4. Optimisers

```
stochastic_gd()
momentum_gd()
nesterovacc_gd()
rmsprop()
adam()
nadam()
```





