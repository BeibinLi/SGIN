"""
Note:
    before running this code, you shoud run get_mnist.sh and put the data
    in "MNIST" folder.
"""

from mnist import MNIST # pip install python-mnist

import pickle
import pandas as pd
import numpy as np


import tqdm # pip install tqdm
import pywt # pip install PyWavelets 

import random
    
# %%
# Note: the gz can be True or False, depends on your OS and do you have the gz
#       file or not
mndata = MNIST('mnist', gz=False)

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()


train_images = np.array(train_images) / 255.0
test_images = np.array(test_images) / 255.0

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# %%
feature_group = None
def transform_img(img, get_feature_group=False, level=4):
    global feature_group
    img = img.reshape(28, 28)
    coeffs = pywt.wavedec2(img, "haar", level=level)
    coeffs_flat = [coeffs[0]] + [_ for arr in coeffs[1:] for _ in arr]
    
    rst = []
    if get_feature_group:
        feature_group = []
    for idx, np_arr in enumerate(coeffs_flat):
        rst += np_arr.reshape(-1).tolist()
        if get_feature_group:
            feature_group += [idx] * np_arr.reshape(-1).shape[0]    
    return rst


def transform_dataset(X):
    rst = []
    for idx in tqdm.tqdm(range(X.shape[0])):
        rst.append(transform_img(X[idx, :], get_feature_group=False))
        
    rst = np.array(rst)
    return rst
    
    
x = transform_img(train_images[0, :], get_feature_group=True, level=4)
# %%
test_images = transform_dataset(test_images)

assert(test_images.shape[0] == test_labels.shape[0]) 
assert(test_images.shape[1] >= 28 * 28)

train_images = transform_dataset(train_images)

# %%
random.seed(12345)

val_idx = np.random.choice(range(train_images.shape[0]), size=1000, replace=False)

# Save the validation index for future duplication (in case seed becomes different)
np.savetxt("val_idx_for_mnist.txt", val_idx)

val_images = train_images[val_idx, :]
val_labels = train_labels[val_idx]

train_idx = [True] * train_images.shape[0]

for vi in val_idx: 
    train_idx[vi] = False


my_train_images = train_images[np.array(train_idx)]
my_train_labels = train_labels[np.array(train_idx)]

print(pd.Series(val_labels).value_counts())

# %%    
pickle.dump([my_train_images, my_train_labels, 
             val_images, val_labels,
             test_images, test_labels, 
             feature_group], 
        open("mnist_wavelet.pickle", "wb"))
