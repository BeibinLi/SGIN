"""
Note:
    before running this code, you shoud run get_mnist.sh and put the data
    in "MNIST" folder.
"""

from mnist import MNIST # pip install python-mnist

import pickle
import pandas as pd
import numpy as np
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
# %% List definition for the sparse groups
feature_group_by_row = []
for i in range(28):
    feature_group_by_row += [i] * 28
    
# %%    
pickle.dump([my_train_images, my_train_labels, 
             val_images, val_labels,
             test_images, test_labels, 
             feature_group_by_row], 
        open("mnist_raw.pickle", "wb"))
