from sgin_model import SGIN, sbcgd_train, theory_sbcgd_train, sgd_train, USE_CUDA, TOL

from pyglmnet.pyglmnet  import GLM
import numpy as np

import torch
import torch.utils.data

from sklearn.linear_model import Lasso

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score

from collections import defaultdict

    
import os

import time

# %%

if not os.path.exists("rst"): os.mkdir("rst")

# %%
def sparsity_count_for_linear_model(coefs, feature_groups, verbose=True):
    """ Count number of sparse parameters and sparse groups of parameters
     for a one-layer machine learning model 
     
     
     Args:
         coefs (np.array): [d, 1] array
         feature_groups (np.array): [d, 1] array
    
    Returns:
        num_sparse_features (int): number of sparse features in the model
        num_sparse_group (int): number of sparse groups in the model
        sparse_groups (list): a list that contains group IDs that are identified as sparse.
    """    
    coefs = np.array(coefs).reshape(-1)
    feature_groups = np.array(feature_groups).reshape(-1)
    
    unique_type = np.unique(feature_groups)
    
    num_sparse_features = np.sum(coefs==0)
    
        
    num_sparse_group = 0
    sparse_groups = []
    for group_id in range(len(unique_type)):
        parameters_for_the_group = np.abs(coefs[feature_groups == group_id])
        
        if np.sum(parameters_for_the_group < TOL) == len(parameters_for_the_group):
            num_sparse_group += 1
            sparse_groups.append(group_id)
        
    if verbose:
        print("%d/%d parameters are sparse" %(num_sparse_features, len(coefs)))
        print("Total %d/%d groups are sparse" % (num_sparse_group, len(unique_type)))
    
    if len(sparse_groups) > 0:
        print("Final Sparsed Groups:", sparse_groups)

    return num_sparse_features, num_sparse_group, sparse_groups

# %%
def classification_metric(all_real, all_pred, all_prob):
    """ Metric used for experiments
    
    Args:
        all_real (list): real labels (ground truth) with n values
        all_pred (list): predictions for n predictions
        all_prob (list or np.array): probabilities (confidence). 
                If it is a numpy matrix, the size should be [n, c]
    
    Returns:
        accuracy, f1 score, auc score, confusion matrix, precision, recall, 
        sensitivity, specificity, best_cc
    """
    all_real = np.array(all_real)
    all_pred = np.array(all_pred)
    
    if type(all_prob[0]) is np.ndarray:
        all_prob = np.concatenate(all_prob)
    else:
        all_prob = np.array(all_prob)
    
    acc = np.mean( np.array(all_real) == np.array(all_pred))
        
    n = len(all_real)
    
    if len(np.unique(all_real)) == 2:
        avg_method = "binary"
    else:
        avg_method = "weighted"
        num_class = len(np.unique(all_real))
        try:
            all_real_one_hot = np.zeros((n, num_class))
            all_real_one_hot[np.arange(n), all_real.astype(int)] = 1
        except:
            all_real_one_hot = None


    try:
        f1 = f1_score(all_real, all_pred, average=avg_method)
    except:
        f1 = -1
        
        
    try:
        if avg_method == "binary":
            auc = roc_auc_score(all_real, all_prob)
        else:
            auc = roc_auc_score(all_real_one_hot, all_prob, average=avg_method)
    except:
        auc = -1
        

    print("Confusion Matrix")
    cm = confusion_matrix(all_real, all_pred)
    print(cm)

    try:
        precision = precision_score(all_real, all_pred, average=avg_method)
    except:
        precision = -1
        
    try:
        recall = recall_score(all_real, all_pred, average=avg_method)
    except:
        recall = -1
    sensitivity = recall

    try:
        tn = cm[0][0]  # True Negative
        negative = cm[0][0] + cm[0][1]
        specificity = tn / negative  # 1 minus false positive rate
    except:
        specificity = -1

    if avg_method == "binary":
        TN, FP, FN, TP = cm.ravel()
        
        cc = ((TP * TN) - (FN * FP)) / (np.sqrt(TP + FN)* np.sqrt(TN + FP) * np.sqrt(TP + FP)  * np.sqrt(TN + FN) )
        print("Standard CC:", cc)
        
        best_cc = 0    
        for threshold in np.arange(np.min(all_prob), np.max(all_prob), 0.01):
            cc = np.corrcoef(all_real, all_prob > threshold)[0][1]
            best_cc = max(cc, best_cc)
        print("!" * 30, "CC = ", best_cc, "!" * 30)
             
    else:
        best_cc = 0
    
    print("Acc:", acc, "F1:", f1, "AUC:", auc)
    print("Sensitivity (recall):", sensitivity, "Specificity:", specificity, "Precision:", precision)
        
    return acc, f1, auc, cm, precision, recall, sensitivity, specificity, best_cc


# %% Helper Functions to train, valid, test the SGIN model
    
def sgin_predict(model, dataloader, lam=0):
        # Use these container to store validation/testing result
    all_real = []
    all_prob = []
    all_pred = []
    wrong_rst = defaultdict(list)
    wrong_rst["sparse_groups"] = model.sparse_groups
    wrong_rst["lambda"] = lam

    for val_x, val_y in dataloader:
        if USE_CUDA:
            val_x, val_y = val_x.cuda(), val_y.cuda()

        output = model(val_x)
        
        all_real += val_y.cpu().numpy().reshape(-1).tolist()
        
        output_with_prob = torch.softmax(output, dim=1)

        if len(output.shape) == 1 or output.shape[1] == 1:
            # Regression
            pred = output.reshape(-1).detach().cpu().numpy()
            all_pred += pred.reshape(-1).tolist()
            all_prob += pred.reshape(-1).tolist() # the prob and pred should have same value for programming convenience
        elif output.shape[1] >= 2:
            # Classification problem
            confidence, pred = torch.max(output_with_prob, dim=1) # [0,1].cpu().item()
            
            pred = pred.cpu().numpy()
            all_pred += pred.reshape(-1).tolist()
            
            # pdb.set_trace()
            # convert to numpy to organize results
            val_x = val_x.cpu().numpy()
            val_y = val_y.cpu().numpy() 
            wrong_idx = np.where(pred != val_y)[0]
            wrong_rst["data"] += val_x[wrong_idx].tolist()
            wrong_rst["real"] += val_y[wrong_idx].tolist()
            wrong_rst["pred"] += pred[wrong_idx].tolist()

            if output.shape[1] == 2:            
                # If it is binary classification. We want Pr(positive)
                prob = output_with_prob[:, 1].detach().cpu().numpy()
                all_prob += prob.reshape(-1).tolist()
            else:
                # If it is multi-category classificaiton. We want confidence.
                prob = output_with_prob.detach().cpu().numpy()
                all_prob.append(prob)
        else:
            assert(False)
            
    wrong_rst["accuracy"] = np.sum(np.array(all_real) == np.array(all_pred)) / len(all_real)
    
    # pickle.dump(wrong_rst, open("rst/classification_wrong_%f.pickle" % lam, "wb"))

    if output.shape[1] > 2:
        all_prob = np.concatenate(all_prob)      
        
        
    return all_real, all_pred, all_prob

def sgin_experiment(train_features, train_labels, 
                    val_features=None, val_labels=None, 
                    test_features=None, test_labels=None,
                    feature_groups=[], input_dim=None, cv_id=-1, criterion=None, 
                    optmizer_method:str="sbcgd",
                    lam:float=0.01, layers=[],
                    num_epochs:int=10, train_batch_size:int=100, 
                    learning_rate:float=0.1, verbose=True, model_ofname=None):
    
    """
    Perform training, validation, and testing for SGIN models.
    
    Args:
        train_features (np.array): [n, d] numpy matrix
        train_labels (np.array): [n, ] numpy matrix
        val_features (np.array): optional [m1, d] numpy matrix
        val_labels (np.array): optional [m1, d] numpy matrix
        test_features (np.array): optional [m2, d] numpy matrix
        test_labels (np.array): optional [m2, d] numpy matrix
        feature_groups (np.array or dict): if it is numpy array, then it is
            a [d, ] array that specify the groups.
            If it is a dict, then it must use the "sbcgd" SGIN algorithm.
        cv_id (list or np.array): cross validation id. This is used to save the weight 
        criterion: PyTorch object for loss function. e.g. 
                torch.nn.CrossEntropyLoss(), torch.nn.MSELoss(), etc
        optmizer_method (str): either "sbcgd" or "sgd"
        lam (float): $\lambda$ used for regularization
        layers (list): an array to define the Neural Network structure for SGIN
        model_ofname (str): the output filename for the model (ends with .pt). If 
            this value is specified, the trained model will be saved with this
            filename.

    Returns: 
        real labels, prediction probablities, prediction labels, 
            number of sparse features, number of sparse groups
    """


    # Create dataset and dataloader
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_features), torch.Tensor(train_labels))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    

    if input_dim == None:
        input_dim = len(feature_groups)
    model = SGIN(input_dim, layers=layers, groups=feature_groups)
    print(model)
    
    if USE_CUDA: 
        model = model.cuda()

    
    if num_epochs < 10:
        decay = 0.5
    elif num_epochs < 50:
        decay = 0.9
    elif num_epochs < 100:
        decay = 0.95
    elif num_epochs < 200:
        decay = 0.98
    else:
        # Large number of epochs. 
        decay = 0.99
    
    print("sgin_experiment: Begin training", optmizer_method)
    # Train    
    for epoch in range(num_epochs):
        if optmizer_method == "sbcgd":
            sbcgd_train(model, criterion, train_dataloader, lr=learning_rate * (decay ** epoch), lam=lam, verbose=verbose) 
        elif optmizer_method == "sgd":
#            for _ in range(len(np.unique(feature_groups))):
            sgd_train(model, criterion, train_dataloader, lr=learning_rate * (decay ** epoch), lam=lam, verbose=verbose) 
                
        elif optmizer_method == "theory":            
            theory_sbcgd_train(model, criterion, train_dataloader, lr=learning_rate * (decay ** epoch), lam=lam, verbose=verbose) 
        else:
            raise("Unknow optimization method")
        if verbose:
            print("finished training...", cv_id, epoch)
            
    n_sparse_features, n_sparse_groups = model.count_sparsity()
    
    model = model.eval()


    val_rst = None
    test_rst = None

    if val_features is not None:
        val_dataset = torch.utils.data.TensorDataset(torch.Tensor(val_features), torch.Tensor(val_labels))
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_batch_size)
        val_rst = sgin_predict(model, val_dataloader, lam=lam)

    if test_features is not None:
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_features), torch.Tensor(test_labels))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=train_batch_size)
        test_rst = sgin_predict(model, test_dataloader, lam=lam)
        
    
    if model_ofname is not None and type(model_ofname) is str:
        torch.save(model, model_ofname)

    return val_rst, test_rst, n_sparse_features, n_sparse_groups, model.sparse_groups

# %%
def linear_predict(model, features, labels):
    predictions = model.predict(features)           
    all_real = labels.reshape(-1).tolist()
    all_prob = predictions.reshape(-1).tolist()
    all_pred = (predictions.reshape(-1) > 0.5).tolist()


    return all_real, all_pred, all_prob

def linear_experiment(train_features, train_labels, 
                      val_features=None, val_labels=None, 
                      test_features=None, test_labels=None, 
                      feature_groups=[], lambda_term=0.01, 
                      model_to_use="lasso"):
    
    """ Perform training and validation
    
    Args:
        train_features (np.array): [n, d] numpy matrix
        train_labels (np.array): [n, ] numpy matrix
        val_features (np.array): [m1, d] numpy matrix
        val_labels (np.array): [m1, d] numpy matrix
        test_features (np.array): [m2, d] numpy matrix
        test_labels (np.array): [m2, d] numpy matrix
        feature_groups (np.array): [d, ] array that specify the groups
        lambda_term (float): $\lambda$ used for regularization
        model_to_use (str): either "lasso", "linear regression", or 
                            "logistic regression"
        
    Returns: 
        real labels, prediction probablities, prediction labels, 
            number of sparse features, number of sparse groups
    """
    if model_to_use == "linear regression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif model_to_use == "logistic regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif model_to_use == "lasso":
        np.random.seed(int(time.time()))
        model = Lasso(alpha=lambda_term, 
                      random_state=np.random.randint(low=0, high=1e7), 
                      warm_start=False,
                      selection="random") 
        # coordiante descent random order
    elif model_to_use == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=lambda_term)
    elif model_to_use == "group lasso":
        model = GLM(distr="binomial", tol=1e-2, reg_lambda=lambda_term,
               group=feature_groups, score_metric="pseudo_R2",
               alpha=1.0)
    else:
        raise NameError("Unknown %s model to use. You shuold use lasso, linear regression, or logistic regression here" % model_to_use)

    model.fit(train_features, train_labels)

    val_rst = None
    test_rst = None
    
    if val_features is not None:
        val_rst = linear_predict(model, val_features, val_labels)
    
    if test_features is not None:
        test_rst = linear_predict(model, test_features, test_labels)

    if model_to_use == "group lasso":    
        coefs = model.beta_
    elif (model_to_use.find("svm") >= 0 or model_to_use.find("svr") >= 0) and model_to_use.find("linear") < 0:
        coefs = [1] * len(feature_groups)
    elif model_to_use.find("tree") >= 0 or model_to_use.find("forest") >= 0:
        coefs = [1] * len(feature_groups)
    else:
        coefs = model.coef_
    
    n_sparse_features, n_sparse_groups, sparse_groups = sparsity_count_for_linear_model(coefs, feature_groups)
    
    return val_rst, test_rst, n_sparse_features, n_sparse_groups, sparse_groups


