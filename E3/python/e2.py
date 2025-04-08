import numpy as np
from utils_ps3 import load_npz
from e1 import preprocess_traces, sbox
import matplotlib.pyplot as plt
import tqdm

import os
root = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)

hw = load_npz("HW.npz")["HW"][0]

def pearson_corr(x, y):
    """
    x: raw traces, as an np.array of shape (nb_traces, nb_samples)
    y: model, as an np.array of shape (nb_traces, nb_samples)
    return: the pearson coefficient of each samples as a np.array of shape (1,nb_samples)
    """
    numerator = np.cov(x, y, rowvar=False)[0, 1:]
    denominator = np.std(x, axis=0) * np.std(y, axis=0)
    return numerator / denominator

def cpa_byte_out_sbox(index, pts, traces):
    """
    index: index of the byte on which to perform the attack.
    pts: the plaintext of each encryption performed.
    traces: the power measurements performed for each encryption.
    return: an np.array with the key bytes, from highest probable to less probable
    when performing the attack targeting the input of the sbox
    """

    # Define a function that computes the correlation block between
    # traces and a model (here, hw_vec) along the same observation axis.
    def vector_pearson_corr(x, y):
        # x: traces, shape (nb_traces, nb_samples)
        # y: hw_vec, shape (256, nb_traces)
        # We want to correlate each trace sample (variable) with each hypothesis.
        # To do that, we first convert hw_vec so that each column becomes a variable.
        # Observations are the different traces (rows in x and in y.T).
        nb_samples = x.shape[1]
        y_t = y.T  # shape becomes (nb_traces, 256)
        # Now, with rowvar=False, rows are observations, columns are variables.
        c = np.corrcoef(x, y_t, rowvar=False)  # shape: (nb_samples+256, nb_samples+256)
        # The block of correlations between the trace samples (first nb_samples columns)
        # and the key hypotheses (last 256 columns) is:
        corr_block = c[:nb_samples, nb_samples:]
        return corr_block  # shape: (nb_samples, 256)
    
    # pts[:, index] has shape (nb_traces,)
    # Broadcast with np.arange(256) to get a (256, nb_traces) array:
    m_vec = np.bitwise_xor(pts[:, index], np.arange(256).reshape(-1, 1))
    m_vec = sbox[m_vec]       # Still shape (256, nb_traces)
    hw_vec = hw[m_vec]        # Also (256, nb_traces)
    
    # Compute the correlation block:
    corr_block = vector_pearson_corr(traces, hw_vec)  # shape: (nb_samples, 256)
    # For each hypothesis (each column), take the maximum absolute correlation across samples:
    key_scores = np.max(np.abs(corr_block), axis=0)  # shape: (256,)
    
    # Return key candidates sorted most-likely first:
    return np.argsort(key_scores)[::-1]
    
def cpa_byte_in_sbox(index, pts, traces):
    """
    index: index of the byte on which to perform the attack.
    pts: the plaintext of each encryption performed.
    traces: the power measurements performed for each encryption.
    return: an np.array with the key bytes, from highest probable to less probable
    when performing the attack targeting the output of the sbox
    """
    # TODO
    return np.arange(16)


def run_full_cpa_known_key(pts, ks, trs, idx_bytes, out=True):
    print("Run CPA the with the known key for bytes idx: \n{}".format(idx_bytes))
    if out:
        print("Target: sbox output")
    else:
        print("Target: sbox input")
    key_bytes_found = 16 * [0]
    key_ranks = 16 * [0]
    for i in idx_bytes:
        print("Run CPA for byte {}...".format(i), flush=True)
        if out:
            cpa_res = cpa_byte_out_sbox(i, pts, trs)
        else:
            cpa_res = cpa_byte_in_sbox(i, pts, trs)
        key_bytes_found[i] = cpa_res[0]
        key_ranks[i] = np.where(cpa_res == ks[0, i])[0]
        print("Rank of correct key: {}".format(key_ranks[i]), end="")
        if key_bytes_found[i] == ks[0, i]:
            print("")
        else:
            print("--> FAILURE")

    hits = np.sum(ks[0, :] == key_bytes_found)
    print("{} out of {} correct key bytes found.".format(hits, len(idx_bytes)))
    print("Avg key rank: {}".format(np.mean(key_ranks)))
    return key_ranks


if __name__ == "__main__":
    # Load the data
    dataset = load_npz("attack_set_known_key.npz")
    plaintexts = dataset["xbyte"]
    keys = dataset["kv"]
    traces = dataset["traces"].astype(np.float64)

    # Amount trace taken
    am_tr = min(100, plaintexts.shape[0])

    plaintexts = plaintexts[:am_tr, :]
    keys = keys[:am_tr, :]
    traces = traces[:am_tr, :]

    # Prepocess traces
    traces = preprocess_traces(traces)

    # Run the attack
    # Indexes of byte to attack
    idxes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    run_full_cpa_known_key(plaintexts, keys, traces, idxes, out=True)
