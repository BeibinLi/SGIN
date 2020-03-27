"""
Note: this code cleans the data in MEM dataset.

Note: we will compare our result with GRPLasso package in R (Group Lasso for Logistic Regression)


Beibin
"""
import numpy as np
import pandas as pd

import pdb
import os

# %%

def one_hot_df(pos_filename, neg_filename):
    order = ["A", "C", "G", "T"]
    pos_file = open(pos_filename, "r")
    neg_file = open(neg_filename, "r")
    positive_sequences = [str(line.strip().upper()) for idx, line in
                          enumerate(pos_file.readlines())
                          if ">" not in line]
    
    negative_sequences = [str(line.strip().upper()) for idx, line in
                          enumerate(neg_file.readlines())
                          if ">" not in line]
    
    def letter_to_one_hot(gene):
        idx = order.index(gene)
        rst = [0] * 4
        rst[idx] = 1
        return rst
    
    
    positive_vector_matrix = np.array([[letter_to_one_hot(_) for _ in s]
                                       for s in positive_sequences])
        
    positive_vector_matrix = positive_vector_matrix.reshape(positive_vector_matrix.shape[0], -1)
    
    negative_vector_matrix = np.array([[letter_to_one_hot(_) for _ in s]
                                       for s in negative_sequences])
        
    negative_vector_matrix = negative_vector_matrix.reshape(negative_vector_matrix.shape[0], -1)
    
    df = pd.DataFrame(data=np.vstack((positive_vector_matrix,
                                      negative_vector_matrix)))
    df.loc[0:positive_vector_matrix.shape[0], "Label"] = 1.0
    df.loc[positive_vector_matrix.shape[0]:, "Label"] = 0.0
    
    groups = np.array([[_] * 4 for _ in range(7)]).reshape(-1).tolist()
    
    return df, groups


def traditional_csv(pos_filename, neg_filename):
    pos_file = open(pos_filename, "r")
    neg_file = open(neg_filename, "r")
    positive_sequences = [str(line.strip().upper()) for idx, line in
                          enumerate(pos_file.readlines())
                          if ">" not in line]
    
    negative_sequences = [str(line.strip().upper()) for idx, line in
                          enumerate(neg_file.readlines())
                          if ">" not in line]

    positive_vector_matrix = np.array([[_.split() for _ in s]
                                       for s in positive_sequences])
        
    positive_vector_matrix = positive_vector_matrix.reshape(positive_vector_matrix.shape[0], -1)
    
    negative_vector_matrix = np.array([[_.split() for _ in s]
                                       for s in negative_sequences])
        
    negative_vector_matrix = negative_vector_matrix.reshape(negative_vector_matrix.shape[0], -1)
    
    df = pd.DataFrame(data=np.vstack((positive_vector_matrix,
                                      negative_vector_matrix)))
    df.loc[0:positive_vector_matrix.shape[0], "Label"] = 1.0
    df.loc[positive_vector_matrix.shape[0]:, "Label"] = 0.0

    return df

def process(pfname, nfname, prefix):
    seed_pos = 2019
    seed_neg = 5
    df = traditional_csv(pfname, nfname)
    
    # Create String (categorical) encoding for R code
    df.to_csv(os.path.join(DATA_DIR, "%s_5_str.csv" % prefix), index=False)    
    
    if prefix == "train":
        df_pos = df[df.Label == 1]
        df_neg = df[df.Label == 0]
            
        pos_idx = df_pos.sample(n=5610, random_state=seed_pos).index
        neg_idx = df_neg.sample(n=5610, random_state=seed_neg).index
        
        df_pos_train = df_pos.loc[pos_idx]
        df_neg_train = df_neg.loc[neg_idx]
        
        df_pos_val = df_pos.drop(pos_idx, axis=0)
        df_neg_val = df_neg.drop(neg_idx, axis=0)
        
        
        pd.concat([df_pos_train, df_neg_train], axis=0).to_csv(os.path.join(DATA_DIR, "train_short_5_str.csv"), index=False)
        pd.concat([df_pos_val, df_neg_val], axis=0).to_csv(os.path.join(DATA_DIR, "validate_short_5_str.csv"), index=False)
        
        del df_pos_train, df_neg_train, df_pos_val, df_neg_val

    # Create one-hot vector encoding for Python Code
    df_one_hot, groups = one_hot_df(pfname, nfname)  # cast to one hot vector encoding
    
    df_one_hot.to_csv(os.path.join(DATA_DIR, "%s_5_one_hot.csv" % prefix), index=False)

    np.savetxt(os.path.join(DATA_DIR, "groups_definition_one_hot.txt"), groups)

    if prefix == "train":
        df_pos = df_one_hot[df_one_hot.Label == 1]
        df_neg = df_one_hot[df_one_hot.Label == 0]
        
        df_pos_train = df_pos.loc[pos_idx]
        df_neg_train = df_neg.loc[neg_idx]
        
        df_pos_val = df_pos.drop(pos_idx, axis=0)
        df_neg_val = df_neg.drop(neg_idx, axis=0)
                
        pd.concat([df_pos_train, df_neg_train], axis=0).to_csv(os.path.join(DATA_DIR, "train_short_5_one_hot.csv"), index=False)
        pd.concat([df_pos_val, df_neg_val], axis=0).to_csv(os.path.join(DATA_DIR, "validate_short_5_one_hot.csv"), index=False)
    
# %%
if __name__ == "__main__":
    DATA_DIR = "mit_gene/"
    process(os.path.join(DATA_DIR, "train5_hs.txt"), os.path.join(DATA_DIR, "train0_5_hs.txt"), "train")
    process(os.path.join(DATA_DIR, "test5_hs.txt"), os.path.join(DATA_DIR, "test0_5_hs.txt"), "test")
