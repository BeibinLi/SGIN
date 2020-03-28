from sgin_utils import sgin_experiment, linear_experiment

import sgin_model


import numpy as np

import torch


import pickle
import random
import datetime
import argparse

# %%  Parse the Argument
parser = argparse.ArgumentParser(description='Run MNIST Experiment.')
parser.add_argument('--models', metavar='N', type=str, nargs='+', required=True,
                    help='the model that will be trained in this experiment. It can contain SGIN, Lasso, or group_lasso')


parser.add_argument('--representation', type=str, default="raw",
                    help='The data representation, which should either be "raw" or "wavelet"')


args = parser.parse_args()
args.models = [_.lower() for _ in args.models] # convert to lowercase
args.representation = args.representation.lower()

assert args.representation == "raw" or args.representation == "wavelet", "Unknown data represntation! It should be raw or wavelet."

# %%
def run_mnist_experiment(method, lambda_term, **kwargs):
    print("begin method:".upper(), method, "with \\lambda", lambda_term, datetime.datetime.now(), "#" * 20)
    method = method.lower()

    if "num_epochs" in kwargs:
        num_epochs = kwargs["num_epochs"]
    else:
        num_epochs = 5
        
    if "random_group_order" in kwargs and kwargs["random_group_order"] == True:
        sgin_model.RANDOM_GROUP_ORDER = True

    if "momentum_value" in kwargs and kwargs["momentum_value"] == True:
        sgin_model.MOMENTUM_VALUE = kwargs["momentum_value"]        
        
    if "layers" in kwargs:
        layers = kwargs["layers"]
    
    if "learning_rate" in kwargs:
        learning_rate = kwargs["learning_rate"]
    else:
        learning_rate = 0.1
        
        
    if "batch_size" in kwargs:
        batch_size = kwargs["batch_size"]
    else:
        batch_size = 100
        
    criterion = torch.nn.CrossEntropyLoss()
    
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


    if method in ["sgin", "sgin_sgd", "nn", "theory"]:
        val_rst, test_rst, n_sparse_features, n_sparse_groups, sparse_groups = sgin_experiment(train_features, train_labels, val_features, val_labels, test_features, test_labels,
               group_definition, input_dim=len(group_definition), cv_id=-1, criterion=criterion, optmizer_method=opt_method, lam=lam, layers=layers,
               num_epochs=num_epochs, train_batch_size=batch_size, learning_rate=learning_rate, verbose=False)  
    
        print("Final Sparsity %d features from %d groups in this Cross Validation:" % (n_sparse_features, n_sparse_groups))
    
    
        val_real, val_pred, val_prob = val_rst
        test_real, test_pred, test_prob = test_rst
        
        
        val_acc = np.sum(np.array(val_real) == np.array(val_pred)) / len(val_real)
        test_acc = np.sum(np.array(test_real) == np.array(test_pred)) / len(test_real)
        sparse_groups_found = sparse_groups # set this variable to be compatible with linear method
        
    elif method in ["lasso", "group lasso"]:
        sparse_groups_found = []
        val_rsts = []
        test_rsts = []
        ns_sparse_features = []
        ns_sparse_groups = []
        
        val_pred_matrix = [] # it will become (n, 10), matrix where n = number of testing data
        test_pred_matrix = []
        for label_name in range(10):
            binary_train_labels = train_labels == label_name
            binary_val_labels = val_labels == label_name
            binary_test_labels = test_labels == label_name
        
            val_rst, test_rst, n_sparse_features, n_sparse_groups, sparse_groups = linear_experiment(train_features, binary_train_labels, val_features, binary_val_labels, test_features, binary_test_labels,
               group_definition, lambda_term=lambda_term, 
                          model_to_use=method)        

            # record result for this 1-vs-rest run
            val_rsts.append(val_rst)
            test_rsts.append(test_rst)
            ns_sparse_features.append(n_sparse_features)
            ns_sparse_groups.append(n_sparse_groups)
            sparse_groups_found.append(sparse_groups)
            
            val_pred_matrix.append(val_rst[2]) # probability
            test_pred_matrix.append(test_rst[2]) # probability
        
        val_final_pred = np.array(val_pred_matrix).argmax(axis=0)
        test_final_pred = np.array(test_pred_matrix).argmax(axis=0)
                
        sparse_groups_found = [set(_) for _  in sparse_groups_found]
        
        # final sparse groups
        sparse_groups = set.intersection(*sparse_groups_found)
        
  
        val_acc = np.sum(np.array(val_labels) == np.array(val_final_pred)) / len(val_labels)
        test_acc = np.sum(np.array(test_labels) == np.array(test_final_pred)) / len(test_labels)
    
    
    
    print("!!! Validation Accuracy !!!!", val_acc, "!" * 20)
    print("!!! Testing Accuracy !!!!", test_acc, "!" * 20)
    
    num_features_in_sparse_groups = 0
    for g in sparse_groups:
        num_features_in_sparse_groups += np.sum(np.array(group_definition) == g)
    
    print("Total %d/%d (%.2f%%) features are in the sparse group" % 
              (num_features_in_sparse_groups, len(group_definition),
               num_features_in_sparse_groups / len(group_definition) * 100))

    print("The final sparsified groups are:", sparse_groups)
    
    # Cache Result
    of = open("rst/rst_mnist.tsv", "a")
    # this n_features_used_for_final_product will make the visualization eaiser
    # and it is compatible to the older version of the saved txt file format.
    kwargs["n_features_used_for_final_product"] = train_features.shape[1] - num_features_in_sparse_groups
    rst = [method, lambda_term, 
           n_sparse_features, n_sparse_groups,
           num_features_in_sparse_groups, args.representation, 
           val_acc, test_acc, sparse_groups_found, str(kwargs), str(sparse_groups)]
    
    rst = [str(_) for _ in rst]
    of.write("\t".join(rst).replace("\n", " ") + "\n")
    of.close()
    print("#" * 200)
    
    
# %%
def run_linear_mnist_experiment(method, lambda_term, **kwargs):
    print("begin method:".upper(), method, "with \\lambda", lambda_term, datetime.datetime.now(), "#" * 20)
    method = method.lower()
        
    sparse_groups_found = []
    val_rsts = []
    test_rsts = []
    ns_sparse_features = []
    ns_sparse_groups = []
    
    val_pred_matrix = [] # it will become (n, 10), matrix where n = number of testing data
    test_pred_matrix = []
    for label_name in range(10):
        binary_train_labels = train_labels == label_name
        binary_val_labels = val_labels == label_name
        binary_test_labels = test_labels == label_name
    
        
        
        val_rst, test_rst, n_sparse_features, n_sparse_groups, sparse_groups = linear_experiment(train_features, binary_train_labels, val_features, binary_val_labels, test_features, binary_test_labels,
           group_definition, lambda_term=lambda_term, 
                      model_to_use=method)        
        
        # record result for this 1-vs-rest run
        val_rsts.append(val_rst)
        test_rsts.append(test_rst)
        ns_sparse_features.append(n_sparse_features)
        ns_sparse_groups.append(n_sparse_groups)
        sparse_groups_found.append(sparse_groups)
        
        val_pred_matrix.append(val_rst[2]) # probability
        test_pred_matrix.append(test_rst[2]) # probability

        
    val_final_pred = np.array(val_pred_matrix).argmax(axis=0)
    test_final_pred = np.array(test_pred_matrix).argmax(axis=0)
    
    sparse_groups_found = [set(_) for _  in sparse_groups_found]
    
    final_sparse_group = set.intersection(*sparse_groups_found)
    
    
    val_acc = np.sum(np.array(val_labels) == np.array(val_final_pred)) / len(val_labels)
    test_acc = np.sum(np.array(test_labels) == np.array(test_final_pred)) / len(test_labels)
    
    print("!!! Validation Accuracy !!!!", val_acc, "!" * 20)
    print("!!! Testing Accuracy !!!!", test_acc, "!" * 20)
    print("Final Sparsity %d groups" % (len(final_sparse_group)), final_sparse_group)

    
    # Cache Result
    of = open("rst/linear_mnist_exp_rst.tsv", "a")
    rst = [method, lambda_term, 
           np.mean(ns_sparse_features), np.mean(ns_sparse_groups),
           0, 0, 
           val_acc, test_acc, sparse_groups_found, str(kwargs), str(final_sparse_group)]
    
    rst = [str(_) for _ in rst]
    of.write("\t".join(rst).replace("\n", " ") + "\n")
    of.close()
    print("#" * 200)

# %%
if __name__ == "__main__":
    # default use wavelets
    nn_layers = [1000, 100, 50, 10]



    train_features, train_labels, \
        val_features, val_labels,\
        test_features, test_labels, group_definition \
        = pickle.load(open("data_prepare/mnist_%s.pickle" % args.representation, "rb"))



    lam1 =  np.random.uniform(1e-10, 0.1, size=[20])  
    lam2 =  np.random.uniform(1e-15, 1e-3, size=[20])  
    lam3 =  np.random.uniform(1e-10, 0.02, size=[20])   
    lambdas = np.concatenate([lam1, lam2, lam3])
    random.shuffle(lambdas)
    
    
    batch_size = 1000    
   
    if "sgin" in args.models:
        for lam in lambdas:
            run_mnist_experiment("sgin", lambda_term=lam, layers=nn_layers, num_epochs=10, 
                             momentum_value=0, batch_size=batch_size, learning_rate=0.1)
            
    if "theory" in args.models:
        for lam in lambdas:
            run_mnist_experiment("theory", lambda_term=lam, layers=nn_layers, num_epochs=10, 
                             momentum_value=0, batch_size=batch_size, learning_rate=0.1)

    if "nn" in args.models:
        # NN
        run_mnist_experiment("nn", lambda_term=0, layers=nn_layers, num_epochs=10, 
                             momentum_value=0, batch_size=batch_size, learning_rate=0.1)
    
    if "sgd" in args.models or "sgin_sgd" in args.models:
        # sgin SGD
        for lam in lambdas:
            run_mnist_experiment("sgin_sgd", lambda_term=lam, layers=nn_layers, num_epochs=10, 
                             momentum_value=0, batch_size=batch_size, learning_rate=0.1)
            
            
    if "lasso" in args.models:
        lam1 =  np.random.uniform(1e-3, 1e-1, size=[20])  
        lam2 =  np.random.uniform(1e-6, 1e-2, size=[20])  
        lambdas = np.concatenate([lam1, lam2])
        for lam in lambdas:
            run_mnist_experiment("lasso", lambda_term=lam)

    if "group_lasso" in args.models:
        lam1 =  np.random.uniform(1e-20, 1e-5, size=[20])  
        lam2 =  np.random.uniform(1e-6, 1e-2, size=[20])  
        lambdas = np.concatenate([lam1, lam2])
        for lam in lambdas:
            run_mnist_experiment("group lasso", lambda_term=lam)   


    
    if "lasso" in args.models:
        lam1 =  np.random.uniform(1e-3, 1e-1, size=[20])  
        lam2 =  np.random.uniform(1e-6, 1e-2, size=[20])  
        lambdas = np.concatenate([lam1, lam2])
        for lam in lambdas:
            run_linear_mnist_experiment("lasso", lambda_term=lam)

    if "group_lasso" in args.models:
        lam1 =  np.random.uniform(1e-20, 1e-5, size=[20])  
        lam2 =  np.random.uniform(1e-6, 1e-2, size=[20])  
        lambdas = np.concatenate([lam1, lam2])
        for lam in lambdas:
            run_linear_mnist_experiment("group lasso", lambda_term=lam)            
