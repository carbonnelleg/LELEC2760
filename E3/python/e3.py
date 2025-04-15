import numpy as np
from utils_ps3 import load_npz
from e1 import preprocess_traces, sbox
from e2 import pearson_corr, vector_pearson_corr, hw
import matplotlib.pyplot as plt

import os
root = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)

def display_bells_and_curves(us, ss):
    """
    us: numpy array containing the mean of each class
    ss: numpy array containing the standard deviation of each class
    
    This function plots two parallel graphs:
      - Left: Gaussian bells for each class.
      - Right: A curve of the class means with a shaded region representing ± one standard deviation.
    """
    # Generate x axis for the bells plot
    x = np.linspace(-0.2, 1.2, 1000)
    
    # Create the figure and two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Gaussian bells
    bell = lambda x, u, s: (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - u) / s) ** 2)
    for i in range(len(us)):
        axes[0].plot(x, bell(x, us[i], ss[i]), label=f"Class {i}")
    axes[0].set_title("Gaussian Bells")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    
    # Right plot: Curves of means with standard deviation as error ribbon
    classes = np.arange(len(us))
    axes[1].plot(classes, us, 'o-', label="Mean")
    axes[1].fill_between(classes, us - ss, us + ss, color='gray', alpha=0.5, label="±1 Std Dev")
    axes[1].set_title("Class Means and Variability")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

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

    # pts[:, index] has shape (nb_traces,)
    # Broadcast with np.arange(256) to get a (256, nb_traces) array:
    m_vec = np.bitwise_xor(pts[:, index], np.arange(256).reshape(-1, 1))
    m_vec = sbox[m_vec]       # Still shape (256, nb_traces)
    hw_vec = hw[m_vec]        # Also (256, nb_traces)

    # Compute the correlation block:
    corr_block = vector_pearson_corr(trs, hw_vec)  # shape: (nb_samples, 256)

    # Select keys
    corr_block = corr_block[time_idx, :]  # shape: (256,)
    corr_block = corr_block[ks] # shape: (nb_keys,)

    # Compute the mean and std of each class
    us = np.mean(corr_block, axis=0)  # shape: (256,)
    ss = np.std(corr_block, axis=0)   # shape: (256,)

    display_bells_and_curves(us, ss)

    return us, ss

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

    # Step 1: Training phase
    time_idx = 0
    models = training_phase(index, time_idx, train_pts, train_ks, train_trs)

    # Step 2: Online phase
    atck_pts = atck_pts[:, index]
    atck_trs = atck_trs[:, time_idx]
    prob = online_phase(index, time_idx, models, atck_pts, atck_trs)
    # Step 3: Sort the probabilities
    sorted_indices = np.argsort(prob)[::-1]
    return sorted_indices

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
    train_traces = train_dataset["traces"].astype(np.float32)

    am_train_tr = min(10000, train_plaintexts.shape[0])
    train_plaintexts = train_plaintexts[:am_train_tr, :]
    train_keys = train_keys[:am_train_tr, :]
    train_traces = train_traces[:am_train_tr, :]

    #
    atck_dataset = load_npz("attack_set_known_key.npz")
    atck_plaintexts = atck_dataset["xbyte"]
    atck_key = atck_dataset["kv"][0, :]
    atck_traces = atck_dataset["traces"].astype(np.float32)

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
