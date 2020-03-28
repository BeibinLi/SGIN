import sgin_model
sgin_model.INTERMEDIATE_ACTIVATION = None

from sgin_utils import sgin_experiment, linear_experiment


import pandas as pd
import numpy as np

import torch

import scipy
import scipy.stats



from sklearn.preprocessing import StandardScaler


import random
import datetime
import time
import re
import argparse


# %%  Parse the Argument
parser = argparse.ArgumentParser(description='Run ET Regression Experiment.')
parser.add_argument('--models', metavar='N', type=str, nargs='+', required=True,
                    help='the model that will be trained in this experiment. It can contain SGIN, Lasso, or group_lasso')


parser.add_argument('--task', type=str, default="iq",
                    help='The regression task, either "ados", "iq", "srs", or "vineland"')


args = parser.parse_args()
args.models = [_.lower() for _ in args.models] # convert to lowercase
args.task = args.task

possible_tasks = ["ados", "iq", "srs", "vineland"]

if args.task not in possible_tasks:
    raise(ValueError('Unknown regression taskeither "ados", "iq", "srs", or "vineland"'))


# %% Read the data
infname = "data_prepare/dummy_et_%s.csv" % args.task
normalize_features = True

df = pd.read_csv(infname)

label_col_name = df.keys()[-1] # the last column is the Label's column name

# drop the label column
features = df.drop([label_col_name], axis=1)

# each stimulus is a group. Total 109 groups
stimulus_types = [re.findall("^[a-z]+_\d+", _)[0] for _ in features.keys()] 

unique_type = np.unique(stimulus_types).tolist()

feature_groups = [unique_type.index(_) for _ in stimulus_types]

feature_groups = np.array(feature_groups)

indices = list(df.index)
random.seed("regression")
random.shuffle(indices)
cross_validation_ids = np.array_split(np.array(indices), 10)

random.seed(time.time()) # reset the seed

# Save Cross Validation Split
f = open("rst/et_%s_cv_split.txt" % label_col_name, "w")
for cv_ in cross_validation_ids:
    f.write("\t".join([str(_) for _ in cv_]) + "\n")
f.close()

# Save group definition to file
f = open("rst/et_%s_group_definition.txt" % label_col_name, "w")
f.write("\t".join([str(_ + 1) for _ in feature_groups]) + "\n")
f.close()


# Normalize the labels to [0, 1] range
df[label_col_name] = (df[label_col_name] - df[label_col_name].min())/ (df[label_col_name].max() - df[label_col_name].min())



# %%
def run_ados_experiment(method, lambda_term, permutation_test=False, **kwargs):
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
        
    if "train_batch_size" in kwargs:
        train_batch_size = kwargs["train_batch_size"]
    else:
        train_batch_size = 1
        
        
    if "lr" in kwargs:
        lr = float(kwargs["lr"])
    else:
        lr = 0.01
    print("Init Learning Rate:", lr)

    if "layers" in kwargs:
        layers = kwargs["layers"]
    else:
        layers = [3000, 'R', 500, 'R', 1, 'S']
        
        
        
    print("Layers:", layers, "Train_batch_size:", train_batch_size)
    
    
    criterion = torch.nn.MSELoss()    

    for cv_id, val_indices in enumerate(cross_validation_ids):
    
        num_val = len(val_indices)
        train_features = features.drop(val_indices).values
        train_labels = df[label_col_name].drop(val_indices).values.reshape(-1)
    
        val_features = features.iloc[val_indices].values.reshape(num_val, -1)
        val_labels = np.array(df.loc[list(val_indices), label_col_name]).reshape(-1)
    
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
                # We can set lambda to zero, so sgin_sgd becomes normal NN
                opt_method = "sgd"
                lam = 0
            elif method == "theory":
                opt_method = "theory"
                lam = lambda_term
    
    
            val_rst, test_rst, n_sparse_features, n_sparse_groups, sparse_groups =\
                sgin_experiment(
                        train_features, train_labels, 
                        val_features, val_labels, 
                        None, None, # no testing set
                        feature_groups, cv_id=cv_id, criterion=criterion, 
                        optmizer_method=opt_method, lam=lam, layers=layers,
                        num_epochs=num_epochs, train_batch_size=train_batch_size, 
                        verbose=False, learning_rate=lr
                )


    
        real, pred, prob = val_rst
        all_real += real
        all_pred += pred
        all_prob += prob
        sparse_features_counts.append(n_sparse_features)
        sparse_groups_counts.append(n_sparse_groups)
        sparse_groups_all.append(sparse_groups)
                
        try:
            print("Curr Results:", scipy.stats.linregress(all_prob, all_real))
        except:
            pass # Not enough data to evaluate correlation
                
        print("Final Sparsity %d features from %d groups in this Cross Validation:" % (n_sparse_features, n_sparse_groups))
    
    print("#" * 10, "SUMMARY for", method)
    print("avg sparse features: %.2f; avg sparse groups: %.2f" % (np.mean(sparse_features_counts), 
                                                                  np.mean(sparse_groups_counts)))
    
    
    all_prob = np.array(all_prob)
    all_prob[np.isnan(all_prob)] = np.mean(df[label_col_name])

    
    mse = ((np.array(all_real) - np.array(all_prob))**2).mean() 
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_prob, all_real)
    
    
    # Cache Result
    of = open("rst/et_%s_rst.txt" % label_col_name, "a") # with > 9000 features
    rst = [method, lambda_term, 
           np.mean(sparse_features_counts), np.mean(sparse_groups_counts),
           np.std(sparse_features_counts), np.std(sparse_groups_counts),
           mse, slope, intercept, r_value, p_value, std_err, kwargs, sparse_groups_all]
    rst = [str(_) for _ in rst]
    of.write("\t".join(rst).replace("\n", " ") + "\n")
    of.close()
    print(label_col_name, "Final Result:", "slope:", slope, "intercept:", intercept, 
          "r_value:", r_value, "p_value:", p_value, "std_err:", std_err, "mse:", mse)
    print("#" * 200)

    return r_value
    

# %%
if __name__ == "__main__":
    lambdas =  np.logspace(np.log(0.1), np.log(1e-15), 30, base=np.exp(1))        
    
    # SGIN
    lam1 =  np.random.uniform(1e-5, 1e-4, size=[10])  
    lam2 =  np.random.uniform(1e-5, 1e-3, size=[10])  
    lam3 =  np.random.uniform(1e-5, 1e-3, size=[10])  
    lam4 =  np.random.uniform(1e-8, 1e-4, size=[10])  
    nn_lambdas = np.concatenate([lam1, lam2, lam3, lam4])
    random.shuffle(nn_lambdas)
        
    if "sgin" in args.models:
        for lam in nn_lambdas:
            run_ados_experiment("SGIN", lambda_term=lam)
            
    # Theory
    if "theory" in args.models:
        for lam in nn_lambdas:
            run_ados_experiment("theory", lambda_term=lam,)
    
    # NN
    if "nn" in args.models:
        run_ados_experiment("nn", lambda_term=0)
        
    # SGIN SGD
    if "sgin_sgd" in args.models:
        for lam in nn_lambdas:
            run_ados_experiment("SGIN_sgd", lambda_term=lam)


    # LASSO
    if "lasso" in args.models:
        lambdas1 =  np.random.uniform(1e-4, 0.1, size=[10])
        lambdas2 =  np.random.uniform(0., 1, size=[10])
        lambdas3 =  np.random.uniform(1e-6, 1e-4, size=[10])
        lambdas4 =  np.logspace(np.log(10000), np.log(2), 10, base=np.exp(1))
        lasso_lambdas = np.concatenate([lambdas1, lambdas2, lambdas3, lambdas4], axis=0)
    
        for lam in lasso_lambdas:
            run_ados_experiment("lasso", lambda_term=lam)
       
    # LR
    if "linear_regression" in args.models:
        run_ados_experiment("linear regression", lambda_term=0)
        
