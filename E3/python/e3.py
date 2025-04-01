import numpy as np
from utils_ps3 import load_npz
from e1 import preprocess_traces, sbox
from e2 import pearson_corr, hw
import matplotlib.pyplot as plt

import os
root = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)

def training_phase(index, time_idx, pts, ks, trs):
    """
    index: index of the byte on which to perform the attack.
    time_idx: time sample to consider when building the model
    pts: the plaintexts used in the training set
    ks: the keys used in the training set
    trs: the traces of the training set
    return: the list [us,ss] where
        us: is a numpy array containing the mean of each class
        ss: is a numpy array containing the standard deviation of each class
    """
    
    return [np.arange(256), np.arange(256)]

def online_phase(index, time_idx, models, atck_pts, atck_trs):
    """
    index: index of the byte on which to perform the attack.
    time_idx: time sample to consider when building the model
    models: the list [us, ss] corresponding to the models of each class
    atck_pts: plaintexts used in the attack set
    atck_trs: traces of the attack set
    return: a numpy array containing the probability of each byte value.
    """
    
    return np.zeros(256)


def ta_byte(index, train_pts, train_ks, train_trs, atck_pts, atck_trs):
    """
    index: index of the byte on which to perform the attack.
    train_pts: the plaintexts used in the training set
    train_ks: the keys used in the training set
    train_trs: the traces of the training set
    atck_pts: plaintexts used in the attack set
    atck_trs: traces of the attack set
    return: a np.array with the key bytes, from highest probable to less probable
    """
    
    return np.arange(256)


def run_full_ta_known_key(
    tr_plain, tr_keys, tr_trs, atck_plain, atck_key, atck_trs, idx_bytes
):
    print("Run TA the with the known key for bytes idx: \n{}".format(idx_bytes))
    key_bytes_found = 16 * [0]
    key_ranks = 16 * [0]
    for i in idx_bytes:
        print("Run TA for byte {}...".format(i), flush=True)
        ta_res = ta_byte(i, tr_plain, tr_keys, tr_trs, atck_plain, atck_trs)
        key_bytes_found[i] = ta_res[0]
        key_ranks[i] = np.where(ta_res == atck_key[i])[0]
        print("Rank of correct key: {}".format(key_ranks[i]), end="")
        if key_bytes_found[i] == atck_key[i]:
            print("")
        else:
            print("--> FAILURE")

    hits = np.sum(atck_key[:] == key_bytes_found)
    print("{} out of {} correct key bytes found.".format(hits, len(idx_bytes)))
    print("Avg key rank: {}".format(np.mean(key_ranks)))
    return key_ranks


if __name__ == "__main__":
    # Load the data
    train_dataset = load_npz("training_set.npz")
    train_plaintexts = train_dataset["xbyte"]
    train_keys = train_dataset["kv"]
    train_traces = train_dataset["traces"].astype(np.float)

    am_train_tr = min(10000, train_plaintexts.shape[0])
    train_plaintexts = train_plaintexts[:am_train_tr, :]
    train_keys = train_keys[:am_train_tr, :]
    train_traces = train_traces[:am_train_tr, :]

    #
    atck_dataset = load_npz("attack_set_known_key.npz")
    atck_plaintexts = atck_dataset["xbyte"]
    atck_key = atck_dataset["kv"][0, :]
    atck_traces = atck_dataset["traces"].astype(np.float)

    am_atck_tr = min(40, atck_traces.shape[0])
    atck_plaintexts = atck_plaintexts[:am_atck_tr, :]
    atck_traces = atck_traces[:am_atck_tr, :]

    # Preprocess traces
    train_traces = preprocess_traces(train_traces)
    atck_traces = preprocess_traces(atck_traces)

    # Run the attack
    # Indexes of byte to attack
    idxes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    run_full_ta_known_key(
        train_plaintexts,
        train_keys,
        train_traces,
        atck_plaintexts,
        atck_key,
        atck_traces,
        idxes,
    )
