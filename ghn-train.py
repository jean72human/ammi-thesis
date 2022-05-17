import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from simgd.deepnets1m.graph import Graph, GraphBatch
from simgd.ghn.nn import GHN
from simgd.ghn.data_store import DataStore


from nasbench_pytorch.model import Network
from nasbench_pytorch.model import ModelSpec
import nasbench_pytorch

import random

import higher

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
sns.set(rc={'figure.figsize':(15,5)})

import wandb

import sys
import os

PATH = sys.argv[5]
LEARNING_RATE = 1e-3
META_LEARNING_RATE = 1e-4
INTERLEAVE = 0
LIMITS=[1]

n_iter = int(sys.argv[1])
meta_batch = int(sys.argv[2])
print_freq = int(sys.argv[3])
run_name = sys.argv[4]

# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

def random_spec():
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = ModelSpec(matrix=matrix, ops=ops)
        if spec.valid_spec:
            return spec

def correct_model(model):
    for layer in model.layers:
        if isinstance(layer, nasbench_pytorch.model.model.Cell):
            for op_i,op in enumerate(layer.vertex_op):
                if op is None: layer.vertex_op[op_i] = nn.Identity()
            for op_i,op in enumerate(layer.input_op):
                if op is None: layer.input_op[op_i] = nn.Identity()

def get_optimizer(model,model_lr=LEARNING_RATE):
    return optim.Adam(model.parameters(), lr=model_lr)

def weights_to_dict(model,graph):
    out = DataStore.empty_dict(model,graph)
    for (key,param) in model.named_parameters():
        if key in out.keys():
            out[key] = param.data.detach().clone()       
    return out

def dict_to_weights(model,weights):
    for (key,param) in model.named_parameters():
        if key in weights.keys():
            param.data = weights[key].detach()  
            
def grad_to_dict(model,graph):
    out = DataStore.empty_dict(model,graph)
    for (key,param) in model.named_parameters():
        if key in out.keys():
            out[key] = param.grad.data.detach().clone()       
    return out
            
def dict_to_grad(model,preds):
    for (key,param) in model.named_parameters():
        if key in preds.keys():
            param.grad = preds[key].detach()
            
def added_noise(data,spread=1e-3):
    for key in data.keys():
        data[key] += torch.normal(mean=0,std=spread,size=data[key].shape, device=data[key].device)
    return data

def store_zeros_like(data):
    new_data = {}
    for key in data.keys():
        new_data[key] = torch.zeros_like(data[key])
    return DataStore(new_data)

def relevant(store,model):
    to_return = []
    for (key,param) in model.named_parameters():
        if key in store.keys():
            to_return.append(store[key])
        else:
            to_return.append(param.detach().clone())
    
    return to_return

def get_models(n_models=8, viz=False):
    paths = []
    graphs = []
    for d in range(n_models):
        model = Network(random_spec(), num_labels=10, in_channels=3, stem_out_channels=128, num_stacks=3, num_modules_per_stack=3)
        m_path = f"d-{d}"
        model.expected_image_sz = (3,32,32)
        g = Graph(model.eval()).to(device)
        graphs.append(g)
        paths.append(m_path)
        correct_model(model)
        model.to(device)
        torch.save(model, m_path+'.pt')
        if viz: g.visualize(figsize=(6,2))
    return paths,graphs

def main():

    run = wandb.init(reinit=True, name=run_name, project="ammi-thesis")
    run.config.update({
        "n_iter":n_iter,
        "meta_batch":meta_batch,
        "schedule_step":print_freq
    })

    ## Graph Hyper Network
    ghn = GHN([512,512,3,3],10, hid=128, ve=True, hypernet='gatedgnn', backmul=False, passes=1, layernorm=True, weightnorm=False, device=device).to(device)

    # MNIST Data 
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64 #11254
    batch_size_test = 32

    trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                            download=True, transform=data_transform)

    #trainset,_ = torch.utils.data.random_split(dataset, [10000, 40000])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                        download=True, transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                            shuffle=True)

    hyper_optimizer = optim.AdamW(ghn.parameters(), lr=META_LEARNING_RATE)
    scheduler = optim.StepLR(hyper_optimizer, step_size=print_freq, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for it in range(n_iter):
        if it%print_freq==0:
            INTERLEAVE+=1
            
        ## TRAIN
        ghn.train()
        paths,graphs = get_models(n_models=meta_batch)
        for e in range(INTERLEAVE):
            pl=0
            hyper_optimizer.zero_grad()
            for idx in range(meta_batch):
                model =  torch.load(paths[idx]+'.pt')
                g = graphs[idx]
                model.to(device)
                model.train()
                opt = optim.Adam(model.parameters())

                weights = weights_to_dict(model,g)
                
                weights = weights.detach()
                new_weights = ghn(g,weights.data)

                with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
                    for data_it, (inputs, labels) in enumerate(trainloader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = fmodel(inputs, params=relevant(new_weights,model))

                        loss = criterion(outputs, labels) #+ 1e-5*new_weights.norm().sum()
                        loss.backward(retain_graph=True)
                        pl+=loss.item()
                        if data_it == LIMITS[-1]:break
                
                
                dict_to_weights(model,new_weights)
                
                torch.save(model, paths[idx]+'.pt')

                if e+1==INTERLEAVE: run.log({"loss":pl/LIMITS[-1]}, step=it+1)
                
            #torch.nn.utils.clip_grad_norm_(ghn.parameters(), 1)
            hyper_optimizer.step() 
            print(f"Iteration {it+1} Leaf {e+1} Loss is {pl/(meta_batch*LIMITS[-1])}")
        scheduler.step()
        
            
        ## TEST    
        if it==0 or (it+1)==n_iter or (it+1)%print_freq==0:
            ghn.eval()
            test_spec = random_spec()
            test_model = Network(test_spec, num_labels=10, in_channels=3, stem_out_channels=128, num_stacks=3, num_modules_per_stack=3)
            test_model.expected_image_sz = (3,32,32)
            test_graph = Graph(test_model.eval()).to(device)
            test_graph.visualize(figsize=(6,6))
            test_model.to(device)
            test_model.train()

            losses = []
            acc = []

            for e in range(5):
                weights = weights_to_dict(test_model,g)

                new_weights = ghn(g,weights.data)

                dict_to_weights(test_model,new_weights)

                running_loss = 0.0
                correct = 0
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # forward
                    outputs = test_model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    pred = outputs.max(1, keepdim=True)[1] # get the index of the max log-probability
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                print(f"Loss: {running_loss/len(testloader)}")
                losses.append(running_loss/len(testloader))
                acc.append(correct/len(testset))

            plt.plot(losses)
            plt.xlabel("epochs")
            plt.ylabel("loss at test time")
            plt.show()
            plt.plot(acc)
            plt.xlabel("epoch")
            plt.ylabel("accuracy at test time")
            plt.show()
            print("\n\n")

    torch.save(ghn.state_dict(), PATH)
    print('Finished Training')

if __name__=="__main__":
    main()