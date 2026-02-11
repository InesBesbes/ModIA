#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trying to build a practical work

@author: pierre
"""
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim

#%% GPU or CPU ?
use_cuda=torch.cuda.is_available()
use_cuda=0
device = torch.device("cuda" if use_cuda else "cpu") 
if use_cuda :
    dtype = torch.cuda.FloatTensor
    dtype = torch.float64
else:
    dtype = torch.FloatTensor
    dtype = torch.float64
print("GPU: ", use_cuda)

#%% Define a 1-layer neural net
class one_layer_NN(nn.Module):
    def __init__(self, n_hidden = 10):
        super(one_layer_NN, self).__init__()
        self.b = 2*(torch.rand(n_hidden)-0.5).to(device)
        self.w = torch.nn.Parameter(torch.zeros(n_hidden)) #nn.parameters allows to define trainable parameters
    def forward(self, x):
        dif = x - self.b[None,None,:]
        hidden = torch.maximum(dif,torch.zeros_like(dif))
        return torch.sum(self.w[None,None,:]*hidden, keepdim=True, dim=2)

        # self.linear1 = nn.Linear(in_features=1, out_features=n_hidden, bias=False)
        # self.activation = nn.ReLU()
        # self.linear2 = nn.Linear(in_features=n_hidden, out_features=1, bias=True)    
        # hidden = self.activation(self.linear1(x))
        # return torch.sum(hidden,keepdim=True, dim=2)

#%% Function to be found
def f(X) : 
    return (torch.abs(X)*torch.sin(2*2*torch.pi*X)).type(dtype)

#%% Parameters
N = 10
n_hidden = 1000 # Defining the model     
learning_rate = 1e-2
batch_size = 10
niter_train = 100000

#%% Generating training data
sigma = 0
M = 1000 # number of sampling points to display the function
T = torch.linspace(-1,1,M).to(device)
fT = f(T) 
plt.figure(1)
plt.plot(T.cpu(),fT.cpu())
plt.title("True function to estimate")
plt.show()

X = torch.linspace(-1,1,N).to(device)
E = sigma * torch.randn_like(X).to(device)
Y = f(X) + E
plt.plot(X.cpu(),Y.cpu())
plt.title("Data")

#%% Setting up the model
model = one_layer_NN(n_hidden = n_hidden).to(device) 
plt.figure(2)
plt.plot(T.cpu(),model(T[:,None,None])[:,0,0].cpu().detach())
plt.title("Initial guess")
plt.show()

#%% Seting up the training procedure
#optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-10)
#optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0)
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay=0)

k = 0
avg_loss = 0
train = []
test = []
model.requires_grad_(True)
model.train()
while k<niter_train:
    #%% Training procedure
    k+=1  
    
    if batch_size>=N:
        xx = X[:,None,None] # For full batch
        yy = Y[:,None,None]
    else:
        # Taking a random batch (size batch x 1 x 1)
        Ik = torch.sort(torch.randint(N,(batch_size,)))[0].to(device)
        yy = Y[Ik][:,None,None]
        xx = X[Ik][:,None,None]
  
    optimizer.zero_grad()
    prediction = model(xx)
    loss = torch.sum((prediction - yy)**2)/batch_size # Defining the loss function
    loss.backward() # Computing the adjoint state (backprop)
    optimizer.step() # One step of a gradient descent
    
    train_loss = torch.sum((model(X[:,None,None]) - Y[:,None,None])**2)/N
    train.append(train_loss)

    #optimizer.param_groups[0]['lr'] = learning_rate/(np.sqrt(k)+1)
    
    #%% Display the stuff
    if k%1000==0:
        model.eval()
        prediction = model(T[:,None,None])[:,0,0]
        avg_error = torch.sum((prediction-fT)**2)/M
        test.append(avg_error.detach().cpu().numpy())
        plt.figure(2)
        plt.plot(T.cpu(),prediction.detach().cpu(),'-r')
        plt.plot(X.cpu(),Y.cpu(),'-ko')
        #plt.plot(T.cpu(),fT.cpu(),'-b')
        plt.axis([-1,1,-1,1])
        plt.title("Iteration %i/%i -- Step: %1.2e -- train:%1.2e -- test: %1.2e " % (k,niter_train, optimizer.param_groups[0]['lr'] , train_loss, avg_error))
        plt.show()
        print("Iteration %i/%i -- Step: %1.2e -- train:%1.2e -- test: %1.2e " % (k,niter_train, optimizer.param_groups[0]['lr'] , train_loss, avg_error))
        model.train()
        
plt.figure(3)
plt.plot(test,'g')
plt.plot(train,'b')