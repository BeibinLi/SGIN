"""
RNA Splicing Experiment
"""

from sgin_utils import classification_metric, sgin_experiment, linear_experiment

import sgin_model

import torch
import pandas as pd

import numpy as np
import pdb

import datetime

import sys

import argparse


# %%  Parse the Argument
parser = argparse.ArgumentParser(description='Run RNA Experiment.')
parser.add_argument('--models', metavar='N', type=str, nargs='+', required=True,
                    help='the model that will be trained in this experiment. It can contain SGIN, Lasso, or group_lasso')



parser.add_argument('--sampling', type=str, default="balance",
                    help='Sampling method, it should be either "balance" or "all"')


args = parser.parse_args()
args.models = [_.lower() for _ in args.models] # convert to lowercase

# %% Load all data as global constant for the "balance" sampling experiments
df_train = pd.read_csv("data_prepare/mit_gene/train_short_5_one_hot.csv")
df_val = pd.read_csv("data_prepare/mit_gene/validate_short_5_one_hot.csv")
df_test = pd.read_csv("data_prepare/mit_gene/test_5_one_hot.csv")

# %% Load data for "all" sampling with "weighted loss" experiments
df_train_big = pd.read_csv("data_prepare/mit_gene/train_5_one_hot.csv") 

feature_groups = np.loadtxt("data_prepare/mit_gene/groups_definition_one_hot.txt")

train_features_big = df_train_big.drop("Label", axis=1).values
train_labels_big = df_train_big["Label"].values.reshape(-1)
train_labels_big = train_labels_big.astype(np.long)

train_features = df_train.drop("Label", axis=1).values
train_labels = df_train["Label"].values.reshape(-1)
train_labels = train_labels.astype(np.long)

val_features = df_val.drop("Label", axis=1).values
val_labels = df_val["Label"].values.reshape(-1)
val_labels = val_labels.astype(np.long)

test_features = df_test.drop("Label", axis=1).values
test_labels = df_test["Label"].values.reshape(-1)
test_labels = test_labels.astype(np.long)


# %%
def run_rna_experiment(method, lambda_term, weighted_loss=False, **kwargs):
    print("begin method:".upper(), method, "with \\lambda", lambda_term, datetime.datetime.now(), "#" * 20)

    method = method.lower()

    if "num_epochs" in kwargs:
        num_epochs = kwargs["num_epochs"]
    else:
        num_epochs = 5
        
        
    if "layers" in kwargs:
        layers = kwargs["layers"]
    else:
        layers = [30, 20, 10, 2]
    
    
    if weighted_loss:
        pos_ratio = np.sum(train_labels_big == 1) / train_labels_big.shape[0]
        weights_sample = torch.Tensor([pos_ratio, 1 - pos_ratio])
        if sgin_model.USE_CUDA:
            weights_sample = weights_sample.cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights_sample)
        
        train_features_ = train_features_big
        train_labels_ = train_labels_big
    else:
        criterion = torch.nn.CrossEntropyLoss()                
        train_features_ = train_features
        train_labels_ = train_labels

    
    if method == "lasso" or method == "group lasso":
        if method == "group lasso":
            print("Please use GLLR model in R code with grplasso package for the RNA Splicing Experiment")

        val_rst, test_rst, n_sparse_features, n_sparse_groups, sparse_groups =\
        linear_experiment(train_features, train_labels, val_features, val_labels,\
                          test_features, test_labels, \
                          feature_groups, lambda_term=lambda_term, model_to_use=method)
 
    
    if method == "sgin" or method == "sgin_sgd" or method == "nn" or method == "theory":
        if method == "sgin":
            opt_method = "sbcgd"
            lam = lambda_term
        elif method == "sgin_sgd":      
            opt_method = "sgd"
            lam = lambda_term
        elif method == "nn":
            opt_method = "sgd"
            lam = 0
        elif method == "theory":
            opt_method = method
            lam = lambda_term


        val_rst, test_rst, n_sparse_features, n_sparse_groups, sparse_groups =\
            sgin_experiment(train_features_, train_labels_, 
                            val_features, val_labels,
                            test_features, test_labels,
                            feature_groups, input_dim=None, cv_id=-1, criterion=criterion, 
                            optmizer_method=opt_method, lam=lam, layers=layers,
                            num_epochs=num_epochs, train_batch_size=100, 
                            verbose=False)

        
    print("Final Sparsity %d features from %d groups in this Cross Validation:" % (n_sparse_features, n_sparse_groups))
    
    
    val_rst = classification_metric(val_rst[0], val_rst[1], val_rst[2])
    test_rst = classification_metric(test_rst[0], test_rst[1], test_rst[2])

    # Cache Result
    of = open("rst/rst_rna_exp.txt", "a")
    rst = [method, lambda_term, 
           n_sparse_features, n_sparse_groups,
           0, 0,
           *val_rst, *test_rst, str(kwargs), sparse_groups]
    
    rst = [str(_) for _ in rst]
    of.write("\t".join(rst).replace("\n", " ") + "\n")
    of.close()
    print("#" * 200)
          
# %%
if __name__ == "__main__":
    lambdas =  np.logspace(np.log(0.1), np.log(1e-15), 30, base=np.exp(1))
    
    nn_layers = [30, 20, 10, 2]    
    
    
    if args.sampling == "balance":
        weighted_loss = False
    elif args.sampling == "all":
        weighted_loss = True
    else:
        print("Unknown sampling method. We will use default balance sampling")


    lam1 = np.random.uniform(1e-4, 0.1, size=[20])
    lam2 =  np.random.uniform(0.1, 1, size=[20])  
    lam3 =  np.random.uniform(0.03, 0.1, size=[20])  
    nn_lambdas = np.concatenate([lam1, lam2, lam3])
        
        
    if "sgin" in args.models:
        # SGIN
        for lam in nn_lambdas:
            run_rna_experiment("sgin", lambda_term=lam, layers=nn_layers, num_epochs=5, 
                               weighted_loss=weighted_loss)
         
    if "theory" in args.models:
        for lam in nn_lambdas:
            run_rna_experiment("theory", lambda_term=lam, layers=nn_layers, num_epochs=5, 
                               weighted_loss=weighted_loss)        

    if "nn" in args.models:
        run_rna_experiment("nn", lambda_term=0, layers=nn_layers, num_epochs=5, weighted_loss=weighted_loss)


    if "sgd" in args.models:
        # SGIN Loss with SGD algorithm
        for lam in nn_lambdas:
            run_rna_experiment("sgin_sgd", lambda_term=lam, layers=nn_layers, num_epochs=5, weighted_loss=weighted_loss)
                
         
    # Note: For lasso, we can only use "balance" sampling

    if "lasso" in args.models:
        # Let's use a wider range for Lasso here so that it can find better solution
        lambdas1 =  np.random.uniform(1e-4, 0.1, size=[10])
        lambdas2 =  np.random.uniform(0., 1, size=[10])
        lambdas3 =  np.random.uniform(0.001, 0.01, size=[10])  
        lambdas4 =  np.logspace(np.log(0.02), np.log(0.000001), 10, base=np.exp(1))
        lambdas5 =  np.logspace(np.log(1), np.log(0.02), 10, base=np.exp(1))
        lambdas6 =  np.logspace(np.log(10000), np.log(10), 10, base=np.exp(1))
        lambdas7 =  np.logspace(np.log(100000), np.log(10), 40, base=np.exp(1))
        lambdas = np.concatenate([lambdas1, lambdas2, lambdas3, lambdas4, lambdas5,
                                  lambdas6, lambdas7], axis=0)
        for lam in lambdas:
            run_rna_experiment("lasso", lambda_term=lam)
        run_rna_experiment("lasso", lambda_term=0)
 