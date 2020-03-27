from sgin_utils import classification_metric, sgin_experiment, linear_experiment


import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import random
import datetime
import time
import re
import argparse


# %%  Parse the Argument
parser = argparse.ArgumentParser(description='Run ASD Classification Experiment.')
parser.add_argument('--models', metavar='N', type=str, nargs='+', required=True,
                    help='the model that will be trained in this experiment. It can contain SGIN, Lasso, or group_lasso')

args = parser.parse_args()
args.models = [_.lower() for _ in args.models] # convert to lowercase

# %%
infname = "data_prepare/dummy_et_asd_classification.csv"
normalize_features = True

df = pd.read_csv(infname)

# The label column, "isASD", contains bool values
df.isASD = df.isASD.astype(int)


features = df.drop(["isASD"], axis=1)
assert(features.shape[1] == 9647)

# each stimulus is a group. Total 109 groups
stimulus_types = [re.findall("^[a-z]+_\d+", _)[0] for _ in features.keys()] 

unique_type = np.unique(stimulus_types).tolist()
open("rst/et_group_names.txt", "w").write("\n".join(unique_type))

feature_groups = [unique_type.index(_) for _ in stimulus_types]

feature_groups = np.array(feature_groups)

indices = list(df.index)
random.seed("ASD")
random.shuffle(indices)
cross_validation_ids = np.array_split(np.array(indices), 10)

random.seed(time.time()) # reset the seed

# Save Cross Validation Split
f = open("rst/asd_cv_split.txt", "w")
for cv_ in cross_validation_ids:
    f.write("\t".join([str(_) for _ in cv_]) + "\n")
f.close()

# Save group definition to file
f = open("rst/et_group_definition.txt", "w")
f.write("\t".join([str(_ + 1) for _ in feature_groups]) + "\n")
f.close()


# %%
def run_asd_experiment(method, lambda_term, **kwargs):
    print("begin method:", method, "with \\lambda", lambda_term, datetime.datetime.now(), "#" * 20)
    method = method.lower()
    
    all_real = []
    all_pred = []
    all_prob = []
    
    sparse_features_counts = []
    sparse_groups_counts = []
    sparse_groups_all = []
    
    if "num_epochs" in kwargs:
        num_epochs = kwargs["num_epochs"]
    else:
        num_epochs = 5

    # we will use cross entropy
    layers = [3000, 500, 2]
    criterion = torch.nn.CrossEntropyLoss()    


    for cv_id, val_indices in enumerate(cross_validation_ids):
    
        num_val = len(val_indices)
        train_features = features.drop(val_indices).values
        train_labels = df.isASD.drop(val_indices).values.reshape(-1)
    
        val_features = features.ix[val_indices].values.reshape(num_val, -1)
        val_labels = np.array(df.isASD[val_indices]).reshape(-1)
    
        # Normalize Features
        if normalize_features:
            scaler = StandardScaler().fit(train_features)
            train_features = scaler.transform(train_features)
            val_features = scaler.transform(val_features)

        print("CV:", cv_id, "Shape Verification:", str(datetime.datetime.now()))  
        
        
        if method == "lasso" or method == "linear regression" or  \
                method == "logistic regression" or method == "group lasso":
            val_rst, test_rst, n_sparse_features, n_sparse_groups, sparse_groups =\
            linear_experiment(train_features, train_labels, 
                              val_features, val_labels,  
                              None, None, # nothing for testing
                              feature_groups,
                              lambda_term=lambda_term, model_to_use=method)

        if method == "sgin" or method == "sgin_sgd" or method == "nn" or method == "theory":
            if method == "sgin":
                opt_method = "sbcgd"
                lam = lambda_term
            elif method == "sgin_sgd":      
                opt_method = "sgd"
                lam = lambda_term
            elif method == "nn":
                opt_method = "sgd"
                lam = 0 # ignore and override the lambda for standard NN method
            elif method == "theory":
                opt_method = "theory"
                lam = lambda_term
    
            val_rst, test_rst, n_sparse_features, n_sparse_groups, sparse_groups =\
                    sgin_experiment(
                            train_features, train_labels, 
                            val_features, val_labels, 
                            None, None, # no testing set. We use cross validation here
                            feature_groups, cv_id=cv_id, criterion=criterion, 
                            optmizer_method=opt_method, lam=lam, layers=layers,
                            num_epochs=num_epochs, train_batch_size=100, 
                            verbose=False)
    
        real, pred, prob = val_rst
        all_real += real
        all_pred += pred
        all_prob += prob
        sparse_features_counts.append(n_sparse_features)
        sparse_groups_counts.append(n_sparse_groups)
        sparse_groups_all.append(sparse_groups)
                
        classification_metric(all_real, all_pred, all_prob)    
                
        print("Final Sparsity %d features from %d groups in this Cross Validation:" % (n_sparse_features, n_sparse_groups))
    
    print("#" * 10, "SUMMARY for", method)
    print("avg sparse features: %.2f; avg sparse groups: %.2f" % (np.mean(sparse_features_counts), 
                                                                  np.mean(sparse_groups_counts)))
    
    
    acc, f1, auc, cm, precision, recall, sensitivity, specificity, _ = classification_metric(all_real, all_pred, all_prob)

    # Cache Result
    of = open("rst/et_asd_classification_rst.tsv", "a") # with > 9000 features
    rst = [method, lambda_term, 
           np.mean(sparse_features_counts), np.mean(sparse_groups_counts),
           np.std(sparse_features_counts), np.std(sparse_groups_counts),
           acc, f1, auc, cm, precision, recall, sensitivity, specificity, kwargs, sparse_groups_all]
    rst = [str(_) for _ in rst]
    of.write("\t".join(rst).replace("\n", " ") + "\n")
    of.close()
    print("#" * 200)

# %%
if __name__ == "__main__":
    # SGIN
    lam1 =  np.random.uniform(1e-5, 1e-3, size=[20])  
    lam2 =  np.random.uniform(1e-6, 1e-4, size=[10])  
    lam3 =  np.random.uniform(1e-8, 1e-4, size=[10])  
    nn_lambdas = np.concatenate([lam1, lam2, lam3])
    random.shuffle(nn_lambdas)

    if "sgin" in args.models:
        for lam in nn_lambdas:
            run_asd_experiment("SGIN", lambda_term=lam,)            
            
    # Theory
    if "theory" in args.models:
        for lam in nn_lambdas:
            run_asd_experiment("theory", lambda_term=lam,)
    
    # NN
    if "nn" in args.models:
        run_asd_experiment("nn", lambda_term=0)
    
    # SGIN SGD
    if "sgd" in args.models:
        for lam in nn_lambdas:
            run_asd_experiment("SGIN_sgd", lambda_term=lam)

  
    lambdas1 =  np.random.uniform(0.01, 0.1, size=[10])
    lambdas2 =  np.random.uniform(0., 1, size=[10])
    lambdas3 =  np.random.uniform(0.001, 0.01, size=[10])  
    lambdas4 =  np.logspace(np.log(0.02), np.log(0.000001), 10, base=np.exp(1))
    lambdas5 =  np.logspace(np.log(1), np.log(0.02), 10, base=np.exp(1))
    lambdas6 =  np.logspace(np.log(10000), np.log(10), 10, base=np.exp(1))
    lambdas7 =  np.logspace(np.log(100000), np.log(10), 40, base=np.exp(1))
    linear_lambdas = np.concatenate([lambdas1, lambdas2, lambdas3, lambdas4, lambdas5,
                              lambdas6, lambdas7], axis=0)
    random.shuffle(linear_lambdas)

    # LASSO
    if "lasso" in args.models:
        for lam in linear_lambdas:
            run_asd_experiment("lasso", lambda_term=lam)
        
    # Group LASSO
    if "group_lasso" in args.models:
        for lam in linear_lambdas:
            run_asd_experiment("group lasso", lambda_term=lam)
