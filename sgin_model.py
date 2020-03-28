import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import tqdm
import random
import copy

from collections import defaultdict
import re


# %% Global Constants
TOL = 1e-5


USE_CUDA = torch.cuda.is_available()

RANDOM_GROUP_ORDER = True
INTERMEDIATE_ACTIVATION = "relu"
WEIGHT_DECAY = 0


# %%
class SGIN(nn.Module):
    def __init__(self, input_dim, layers=[], groups=[]):
        """The Model for SGIN (Sparsely Groupped Input Variables in Neural Network)
        
        
        Args:
            input_dim (int): input dimension
            layers (list): a list to specify the structure of this fully-connected
                neural network. For each element in the list: if the element is
                an integer, it represents a fully-connected layer with certain number
                of cells.; if the element is "relu" or "sigmoid", it means a non-linear
                activation layer; if the element is "dropout 0.3", it represents a
                dropout layer with 30% dropping probability. 
            groups (list, np.ndarray, or set): a list (or np array) of set that 
                defines the group. 
                    e.g. [0, 0, 1, 1, 1, 2] means 6 features in 3 groups.
                If it is a list, then the length of the list should match the 
                input dimension.
                    e.g. {0: [0, 1, 2],
                          1: [0, 1, 3, 5],
                          2: [2, 4]} represents 6 features in 3 overlapping groups.
                If it s a set, then they key is the group ID, and value is a list
                of variable IDs in that group.
        """
        super(SGIN, self).__init__()
        
        assert(len(layers) > 0)
        self.input_dim = input_dim
        
        
        self.groups = groups
        
        if type(self.groups) is list or type(self.groups) is np.ndarray:
            self.group_idx = defaultdict(list)
            assert input_dim == len(groups), "The input dimension is different from the size of the group definition."
            for idx, group_id in enumerate(groups):
                self.group_idx[group_id].append(idx)
        elif type(self.groups) is dict or type(self.groups) is defaultdict:
            self.group_idx = copy.deepcopy(groups)
        else:
            raise(TypeError("The groups should be one of list, np.array, or dict, but got " + 
                              str(type(self.groups))))
            
        self.sparse_groups = set() # a set to store the sparse groups
        self.sparse_features_arr = [] # to store the feature ids

        all_layers = []
        prev_l = input_dim
        for idx, l in enumerate(layers):
            # Activation and Dropout
            if l == "S":
                all_layers.append(nn.Sigmoid())      
                continue
            if l == "R":
                all_layers.append(nn.ReLU())      
                continue
            if l == "T":
                all_layers.append(nn.Tanh())
                continue
            
            if type(l) is str and l.find("dropout") >= 0:
                p = float(re.findall("\d+\.\d+", l)[0])
                all_layers.append(nn.Dropout(p=p))
                continue
            
            # Add Linear Layers. Note: no bias for the first layer
            all_layers.append(nn.Linear(prev_l, l, bias=idx != 0))
            
            # Default Activation for Linear Layers
            if idx != len(layers) - 1: 
                # add activation for all hidden layers
                if INTERMEDIATE_ACTIVATION == "sigmoid":
                    all_layers.append(nn.Sigmoid())
                elif INTERMEDIATE_ACTIVATION == "relu":
                    all_layers.append(nn.ReLU())
                else:
                    print("Unknown intermediate activation %s. Skip activation here." % INTERMEDIATE_ACTIVATION)
            prev_l = l
            
        self.layers = nn.Sequential(*all_layers)     
     
    def forward(self, x):
        """
        x has the shape [batch_size, input_dim]
        """
        # Set weight for features in sparse groups as 0
        self.layers[0].weight.data[:, self.sparse_features_arr] = nn.init.zeros_(self.layers[0].weight.data[:, self.sparse_features_arr])

        # Set input values to sparse groups as 0
        x[:, self.sparse_features_arr] = 0

        return self.layers(x)
    
    def set_group_to_sparse(self, group_id):
        """Set a group to sparse"""
        self.layers[0].weight.data[:, self.sparse_features_arr] = nn.init.zeros_(self.layers[0].weight.data[:, self.sparse_features_arr])

        # We already set this group to sparse in the past, ignore!
        if group_id in self.sparse_groups:
            return


        feature_ids = self.group_idx[group_id]
        sparse_features_set = set(self.sparse_features_arr) 
        [sparse_features_set.add(_) for _ in feature_ids]
        self.sparse_features_arr = list(sparse_features_set)
        self.sparse_groups.add(group_id)
        
        
    def tau_for_group(self, group_id):
        """ Calculate the $\tau_g$ defined in the paper, which is the number of 
        features in the group.
        """
        feature_ids = self.group_idx[group_id]
        p_g = len(feature_ids) # length (number of features) in the group

        return torch.sqrt(p_g * torch.sum(self.layers[0].weight[:, feature_ids] ** 2))
    
    def regularization_layer_1(self):
        w = self.layers[0].weight
        
        group_regularization = 0
        
        # Note: this feature regularization can be used for bi-level sparsity
        # However, we do not use it in this experiment
        feature_regularization = 0
        
        for group_id in self.group_idx.keys():
            feature_ids = self.group_idx[group_id]
            group_regularization += self.tau_for_group(group_id)
            
            for feature_id in feature_ids:
                feature_regularization += torch.sqrt(torch.sum(w[:, feature_id] ** 2))

        return group_regularization, feature_regularization

    def count_sparsity(self):
        """ Count how many features and groups are sparse in the model.
        We use TOL as the threshold (epsilon) for a parameter to be counted as 
        zero.
        
        Note that we should not use the self.sparse_groups and 
        self.sparse_features_arr directly, because SGD might find sparsity by 
        itself without using our algorithm.
        
        We also set the group to sparse (zero) if it is smaller than the 
        tolerance.
        """
        w = self.layers[0].weight.cpu().detach().numpy()

        num_sparse_features = 0
        num_sparse_groups = 0
    
        
        for group_id in self.group_idx.keys():
            curr_group_is_sparse = True
            
            for feature_id in self.group_idx[group_id]:
                curr_feature_is_sparse = True
                for parameter in w[:, feature_id]:
                    if abs(parameter) > TOL:
                        curr_feature_is_sparse = False
                        break
                
                if curr_feature_is_sparse:
                    num_sparse_features += 1
                else:
                    curr_group_is_sparse = False
                    break
            
            # Make sure the group is sparse here
            if curr_group_is_sparse:
                num_sparse_groups += 1
                self.set_group_to_sparse(group_id)
            
        return num_sparse_features, num_sparse_groups
    

# %%
def sbcgd_train(model, criterion, dataloader, lr=0.1, lam=0.1, verbose=True):
    """
    Train the model with the "Stochastic Blockwise Coordinated Gradient Descent"
    algorithm


    Args:
        model (torch.module): the PyTorch model
        criterion (torch.module): a PyTorch loss criterion
        dataloader (torch.utils.data.DataLoader): DESCRIPTION.
        lr (float, optional): Initial learning rate. Defaults to 0.1.
        lam (float, optional): Regularization term $\lambda$. Defaults to 0.1.
        verbose (bool, optional): If verbose, print more training details. Defaults to True.

    Returns:
        None.
    """
    
    if verbose:
        dataloader = tqdm.tqdm(dataloader)
        print("-" * 30, "Begin Stochastic Blockwise Coordinated Gradient Descent")
        
    if USE_CUDA:
        model = model.cuda()
    
    
    groups_ = list(model.group_idx.keys())
    
    if RANDOM_GROUP_ORDER:
        random.shuffle(groups_)
    
    losses = []
    for group_id in groups_:
        if len(model.sparse_groups) == len(model.group_idx) - 1:
            return # Done, sparsified all groups except the remaining one
            
        feature_ids = model.group_idx[group_id]
        
        if group_id in model.sparse_groups:
            # This group is sparse (we already throw it away)
            continue
        
        # The features that we don't care for this block coordinate
        non_feature_ids = list(set(range(model.input_dim)).difference(feature_ids))
        coordinate_loss = []
        
        # The following two arrays are used for logging
        losses_this_group = []
        regs_this_group = []

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, momentum=0)
            
        for x, y in dataloader:
            
            if USE_CUDA:
                x, y = x.cuda(), y.cuda()
            
            x = x.float()
            
            if type(criterion) is torch.nn.modules.loss.MSELoss or type(criterion) is torch.nn.modules.loss.L1Loss:
                y = y.float()
                y = y.reshape(-1, 1)
            else:
                y = y.long()
            
            optimizer.zero_grad()
            pred = model(x)            
            
            pred_loss = criterion(pred, y)

            # The Group Regularization Term we defined
            group_regularization = model.tau_for_group(group_id)

            loss_with_reg = pred_loss  + group_regularization * lam
            
            losses.append(pred_loss.cpu().detach().item())
            losses_this_group.append(pred_loss.cpu().detach().item())
            regs_this_group.append((group_regularization * lam).cpu().detach().item())

            # Calculate Loss for this coordinate                
            X2 = x.clone()
            X2[:, feature_ids] = 0
            pred2 = model(X2)
            loss2 = criterion(pred2, y)

            err = max((loss2 - pred_loss).item(), 0)            
            coordinate_loss.append(err)
            
            loss_with_reg.backward()            
            
            # manually set gradient for non-relevant features in the first layer to zero
            for p in model.parameters():
                p.grad[:, non_feature_ids] = 0 
                break # only clear gradient for the first layer

            optimizer.step()

        if np.mean(coordinate_loss) < lam * model.tau_for_group(group_id):       
            model.set_group_to_sparse(group_id)
        elif verbose:
            print("No sparse in this group. Avg. Coordinate Loss:", np.mean(coordinate_loss))

        assert not np.isnan(coordinate_loss[0]), "The coordinate loss is NaN. Programming error. Loss function, learning rate, or lambda is not well defined."
        assert not np.isnan(losses_this_group[0]), "Loss for group %d is NaN. Loss function, learning rate, or lambda is not well defined."

    
    print("   >>> Avg. loss in this epoch:", np.mean(losses), "with %d sparse groups" % len(model.sparse_groups))
    
    
# %%
def theory_sbcgd_train(model, criterion, dataloader, lr=0.1, lam=0.1, verbose=True):
    """ 
    Train the model with the "Stochastic Blockwise Coordinated Gradient Descent with Theoretical Guarantee"
    algorithm.
    
    This is the "Algorithm 2" in the paper appendix (i.e. Page 12 in https://arxiv.org/pdf/1911.13068.pdf)
    
    Args:
        model (torch.module): the PyTorch model
        criterion (torch.module): a PyTorch loss criterion
        dataloader (torch.utils.data.DataLoader): DESCRIPTION.
        lr (float, optional): Initial learning rate. Defaults to 0.1.
        lam (float, optional): Regularization term $\lambda$. Defaults to 0.1.
        verbose (bool, optional): If verbose, print more training details. Defaults to True.

    Returns:
        None.
    """
    if verbose:
        dataloader = tqdm.tqdm(dataloader)
        print("-" * 30, "Begin sbcgd Coordinate Descent")
        
    if USE_CUDA:
        model = model.cuda()
    
    
    groups_ = list(model.group_idx.keys())
    
    if RANDOM_GROUP_ORDER:
        random.shuffle(groups_)

    losses = []
    for group_id in groups_:
        if len(model.sparse_groups) == len(model.group_idx) - 1:
            return # Done, sparsified all groups except the remaining one

        feature_ids = model.group_idx[group_id]
        
        if group_id in model.sparse_groups:
            # This group is sparse (we already throw it away)
            continue
        
        # The features that we don't care for this block coordinate
        non_feature_ids = list(set(range(model.input_dim)).difference(feature_ids))

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, momentum=0)            
        for x, y in dataloader:
            
            if USE_CUDA:
                x, y = x.cuda(), y.cuda()
            
            x = x.float()
            
            if type(criterion) is torch.nn.modules.loss.MSELoss or type(criterion) is torch.nn.modules.loss.L1Loss:
                y = y.float()
            else:
                y = y.long()
            
            optimizer.zero_grad()
            pred = model(x)            

            pred_loss = criterion(pred, y)
            group_regularization = model.tau_for_group(group_id)
            loss_with_reg = pred_loss  + group_regularization * lam

            loss_with_reg.backward()            
            
            # manually set gradient for non-relevant features in the first layer to zero
            for p in model.parameters():
                p.grad[:, non_feature_ids] = 0 
                break # only clear gradient for the first layer

            optimizer.step()
            
        # The following two arrays are used for logging
        losses_this_group = []
        regs_this_group = []
        coordinate_loss = []
        for x, y in dataloader:
            if USE_CUDA:
                x, y = x.cuda(), y.cuda()
            
            x = x.float()
            
            if type(criterion) is torch.nn.modules.loss.MSELoss or type(criterion) is torch.nn.modules.loss.L1Loss:
                y = y.float()
            else:
                y = y.long()
            
            optimizer.zero_grad()
            pred = model(x)            
            pred_loss = criterion(pred, y)
            
            losses.append(pred_loss.cpu().detach().item())
            losses_this_group.append(pred_loss.cpu().detach().item())
            regs_this_group.append((group_regularization * lam).cpu().detach().item())

            # Calculate Loss for this coordinate                
            X2 = x.clone()
            X2[:, feature_ids] = 0
            pred2 = model(X2)
            loss2 = criterion(pred2, y)

            err = max((loss2 - pred_loss).item(), 0)            
            coordinate_loss.append(err)
                    
        if np.mean(coordinate_loss) < lam * model.tau_for_group(group_id):       
            model.set_group_to_sparse(group_id)
        elif verbose:
            print("No sparse in this group. Avg. Coordinate Loss:", np.mean(coordinate_loss))
            
    print("   >>> Avg. loss in this epoch:", np.mean(losses), "with %d sparse groups" % len(model.sparse_groups))

# %%
def sgd_train(model, criterion, dataloader, lr=0.1, lam=0.1, verbose=True):
    """
    Train the model with standard "Stochastic Gradient Descent" algorithm.
    
    This function and algorithm is NOT recommended for real project. It is here
    just for comparison with other algorithms. Use sbcgd_train(...) function,
    which provides faster and better performance. 
    
    
    Args:
        model (torch.module): the PyTorch model
        criterion (torch.module): a PyTorch loss criterion
        dataloader (torch.utils.data.DataLoader): DESCRIPTION.
        lr (float, optional): Initial learning rate. Defaults to 0.1.
        lam (float, optional): Regularization term $\lambda$. Defaults to 0.1.
        verbose (bool, optional): If verbose, print more training details. Defaults to True.

    Returns:
        None.
    """


    if verbose:
        dataloader = tqdm.tqdm(dataloader)
        print("-" * 30, "Begin Standard SGD")
        
    if USE_CUDA:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, momentum=0)

    losses = []    
    
    losses_this_group = []
    regs_this_group = []
    for x, y in dataloader:
        
        if USE_CUDA:
            x, y = x.cuda(), y.cuda()
        
        x = x.float()

        optimizer.zero_grad()
        pred = model(x)

        if type(criterion) is torch.nn.modules.loss.MSELoss or type(criterion) is torch.nn.modules.loss.L1Loss:
            y = y.float()
        else:
            y = y.long()

        pred_loss = criterion(pred, y)

        group_regularization, _ = model.regularization_layer_1()
        loss_with_reg = pred_loss  + group_regularization * lam        
        loss_with_reg.backward()            

        losses.append(pred_loss.cpu().detach().item())
        losses_this_group.append(pred_loss.cpu().detach().item())
        regs_this_group.append((group_regularization * lam).cpu().detach().item())
        
        optimizer.step()
        
    print("   >>> Avg. loss in this epoch:", np.mean(losses), "with %d sparse groups" % len(model.sparse_groups))

