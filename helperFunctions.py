from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import pickle, pdb
import numpy as np

########################################################################################
## Initialize: test and validate data loaders
## Initialize: train set and train sampler for
## test data loader is initialized every epoc using the particular batch size chosen
########################################################################################
def obtain_dataloader(train_size,test_size,kwargs):

    ## Tranformation to normaliize data
    trans = transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    train_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)
    validate_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)
    test_set = dset.MNIST(root='./data', train=False, transform=trans, download=True)

    ## Description from the paper
    ## "commonly used for image classification. Each sample is a black and white
    ##  image and 28  28 in size. The MNIST is split into three parts: 55,000
    ##  samples for training, 5,000 samples for validation, and 10,000 samples for test."

    indices = list(range(len(train_set)))
    split = int(5000);

    ## Keep validation set deterministic across runs
    random_seed=12345
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    ## Partition train into train and validate
    train_idx, validate_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    validate_sampler = SubsetRandomSampler(validate_idx)

    validate_loader = torch.utils.data.DataLoader(dataset=validate_set, batch_size=test_size, sampler=validate_sampler,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_size, shuffle=True, **kwargs)

    return train_set, train_sampler, validate_loader, test_loader


########################################################################################
## Initialize: The train data loader using the particular batch size chosen
## Train the model for 1 epoch
########################################################################################
def train(model, device, train_set, train_sampler, optimizer, epoch, batch_size, kwargs):
    model.train()

    ## Get train_loader for current batch_size
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size = int(batch_size), sampler=train_sampler, **kwargs)

    ite = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        ## For storing data
        train_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 55000,
                100. * batch_idx / len(train_loader), loss.item()))
        ite += len(output)

    return train_loss

########################################################################################
## Evaluate: Validation loss
########################################################################################
def validate(model, device, validate_loader):
    model.eval()

    validate_loss = 0
    correct = 0
    print("Validating...")
    ite=0
    with torch.no_grad():
        for data, target in validate_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validate_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            ite+=len(output)

    validate_loss /= ite #len(validate_loader.dataset) # Should this be batch_size?
    print('\nValidation set --> Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validate_loss, correct,ite,
        100. * correct / (1.0 * ite) ))

    return validate_loss, 100. * correct / (1.0 * ite)

########################################################################################
## Evaluate: Test loss
########################################################################################
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    print("Testing...")
    ite=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            ite+=len(output)

    test_loss /= ite
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, ite,         100. * correct / (1.0 * ite) ))
    return test_loss, 100. * correct / (1.0 * ite)


## Different loss functions for the bandit algorithm
def loss01(current,prev):
    return prev <= current
def lossHinge(current,prev):
    return max(0,current-prev)
def lossRatio(current,prev):
    return max(0,(current-prev)/np.abs(prev+1e-15))


def store_data(epoch, time, batch_size, probability, rmgd_loss, test_loss, test_accuracy, train_loss,  prev_loss, prev_accuracy, cur_loss, cur_accuracy):

    # get single run result from memory
    single_run_result = pickle.load(open("./data/single_run.pickle", "rb"))

    # update result
    single_run_result["epoch"].append(epoch)
    single_run_result["time"].append(time)
    single_run_result["batch_size"].append(batch_size)
    single_run_result["probability"].append(probability)
    single_run_result["rmgd_loss"] = rmgd_loss

    single_run_result["test_loss"].append(test_loss)
    single_run_result["test_accuracy"].append(test_accuracy)

    single_run_result["train_loss"].append(train_loss)

    single_run_result["validation_loss_prev"].append(prev_loss)
    single_run_result["validation_accuracy_prev"].append(prev_accuracy)

    single_run_result["validation_loss_cur"].append(cur_loss)
    single_run_result["validation_accuracy_cur"].append(cur_accuracy)

    # dump single run result to memory
    pickle.dump(single_run_result, open("./data/single_run.pickle", "wb"))
