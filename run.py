from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os, time
import itertools as iter
import numpy as np
from neuralNetwork import MNIST_NET
from helperFunctions import *

import argparse
import sys


### Major part of the code was adapted from
def single_run(lr, optim, epochs, rmgd, rmgd_loss, rmgd_batch_set, mgd_batch_size, torch_seed=1):
    ########################################################################################
    ## Set up data logger and Parallelization
    ########################################################################################
    torch.manual_seed(torch_seed)

    ## Parallelization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    ## Saving data
    os.system("rm ./data/single_run.pickle")
    single_run = {"epoch": [],  "time":[],
                "test_loss": [], "test_accuracy": [],
                "train_loss": [], "train_accuracy": [],
                "validation_loss_cur": [], "validation_accuracy_cur": [],
                "validation_loss_prev": [], "validation_accuracy_prev": [],
                "batch_size": [], "probability": [], "batch_set": -1 , "rmgd_loss": -1}
    pickle.dump(single_run, open("./data/single_run.pickle", "wb"))



    ########################################################################################
    ## Set up neural network and optimizer
    ########################################################################################
    ## Get the neural network
    model = MNIST_NET().to(device)

    ## Generate data loaders
    test_batch_size = 1000
    train_set, train_sampler, validate_loader, test_loader = obtain_dataloader(train_size=mgd_batch_size,test_size=test_batch_size, kwargs=kwargs)

    ## Choose optimizer
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == "AdaGrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=0)


    ########################################################################################
    ## Set up bandit algorithm
    ########################################################################################
    beta = np.sqrt(np.log(6)/600.0)
    ##                ## Basic                      ## Sub        ## Super
    batch_set_list = [[16, 32, 64, 128, 256, 512], [16, 64, 256], [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]]


    ## Choose the batch set
    if rmgd == 0:
        batch_set = []
        batch_size = mgd_batch_size
        print(mgd_batch_size,flush=True)
        prob = [ -1 ]
    else:
        print("Running with bandits",flush=True)
        batch_set = batch_set_list[rmgd_batch_set]
        index_list = [i for i in range(len(batch_set))]

        # Prior distribution
        prob = [ 1.0/len(batch_set) for i in batch_set]


    ########################################################################################
    ## Start training
    ########################################################################################
    start_time = time.time()

    ## Tracks the testing and additional validation time not inherent to the algorithm
    testing_overhead = 0

    for epoch in range(1, epochs + 1):
        # Select batch size
        if rmgd == 1:
            index = np.random.choice(index_list,1,p=prob)[0]
            batch_size = batch_set[index]
            print("Batch size:",batch_size,flush=True)


        # Store previous loss
        validate_start = time.time()
        prev_loss, prev_accuracy = validate(model=model, device=device, validate_loader=validate_loader)
        testing_overhead += time.time() - validate_start

        # Train
        train_loss = train(model=model, device=device, train_set=train_set, train_sampler=train_sampler, optimizer=optimizer, epoch=epoch, batch_size=batch_size,kwargs=kwargs);

        # Validate and test
        cur_loss, cur_accuracy = validate(model=model, device=device, validate_loader=validate_loader)

        test_start = time.time()
        test_loss, test_accuracy = test(model=model, device=device, test_loader=test_loader)
        testing_overhead += time.time() - test_start

        # Update bandit distribution
        if rmgd == 1:
            if rmgd_loss == 0: y = loss01(current=cur_loss,prev=prev_loss);
            if rmgd_loss == 1: y = lossHinge(current=cur_loss,prev=prev_loss);
            if rmgd_loss == 2: y = lossRatio(current=cur_loss,prev=prev_loss);
            prob[index] = prob[index] * np.exp( -beta * y / prob[index] )
            prob /= np.sum(prob)

        # Store the data
        store_data(epoch=epoch, batch_size=batch_size, probability=prob, rmgd_loss=rmgd_loss, time=time.time()-start_time-testing_overhead,prev_loss=prev_loss, prev_accuracy=prev_accuracy, cur_loss=cur_loss, cur_accuracy=cur_accuracy,train_loss=train_loss, test_loss=test_loss, test_accuracy=test_accuracy)

LOSS_LIST = ['01_loss', 'hinge_loss', 'ratio_loss']


def initialize_data():
    # Contains list with ["name", batch-set, rmgd_loss]
    # batch-set 0: basic
    #           1: sub
    #           2: super

    # rmgd_loss  0: 01 loss
    #           1: hinge loss
    #           2: ratio loss

    CONFIGS = [["basic", 0, 0], ["sub", 1, 0], ["super", 2, 0], ["hinge", 0, 1],["ratio", 0, 2] ]
    experiment_result = {"RMGD": {CONFIGS[0][0]: {}, CONFIGS[1][0]: {}, CONFIGS[2][0]: {}, CONFIGS[3][0]: {}, CONFIGS[4][0]: {}},
                         "MGD": {"16": {}, "32": {}, "64": {}, "128": {}, "256": {}, "512": {}}}
    pickle.dump(experiment_result, open("./data/experiment_result.pickle", "wb"))

    return CONFIGS,experiment_result

def main():
    parser = argparse.ArgumentParser(description='ICLR Reproducibility')
    parser.add_argument('--optim', default="Adam", help='optimizer [\"Adam\", \"AdaGrad\"], default: Adam')
    parser.add_argument('--epochs', default=100, help='Number of Epochs, default: 100')
    parser.add_argument('--reruns', default=10, help='Number of reruns, default: 10')
    args = parser.parse_args()

    CONFIGS,experiment_result = initialize_data()

    ########################################################################################
    ## Run RMGD simulation
    ########################################################################################
    for rmgd_config, runs in iter.product(CONFIGS,range(args.reruns)):
        print("RMGD config:", rmgd_config, "Optimizer:", optim, "Run #",runs,flush=True)
    
        single_run(lr=1e-4, optim=args.optim, epochs=args.epochs, rmgd=1, rmgd_loss=rmgd_config[2], rmgd_batch_set=rmgd_config[1], mgd_batch_size=-1)
    
        # Get result from single run
        single_run_result = pickle.load(open("./data/single_run.pickle", "rb"))

        # Put single run result in experiment_result
        experiment_result["RMGD"][rmgd_config[0]][runs] = single_run_result
        pickle.dump(experiment_result,open("./data/experiment_result.pickle", "wb"))

    return
    ########################################################################################
    ## Run MGD simulation
    ########################################################################################
    mdg_batches = ["512", "256", "128","64", "32", "16"]
    for mgd_batch, runs in iter.product(mdg_batches,range(args.reruns)):
        print("MGD batch size:", mgd_batch, "Optimizer:", optim, "Run #",runs,flush=True)

        single_run(lr=1e-4, optim=args.optim, epochs=args.epochs, rmgd=0, mgd_batch_size=mgd_batch , rmgd_loss=-1, rmgd_batch_set=-1)

        # Get result from single run
        single_run_result = pickle.load(open("./data/single_run.pickle", "rb"))

        # Put single run result in experiment_result
        experiment_result["MGD"][mgd_batch][runs] = single_run_result
        pickle.dump(experiment_result,open("./data/experiment_result.pickle", "wb"))

if __name__ == '__main__':
    main()
