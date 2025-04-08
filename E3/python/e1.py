import numpy as np
from utils_ps3 import load_npz, plot_traces, dec2bits
import os

root = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)
sbox = load_npz("AES_Sbox.npz")["AES_Sbox"][0]


def preprocess_traces(traces):
    """
    traces: raw traces.
    return: return the preprocessed set of traces
    """
    
    return traces[:, 2400:3800]/np.max(traces[:, 2400:3800], axis=1)[:, None]


def vectorized_dpa_byte(index, pts, traces):
    """
    Vectorized version of the DPA attack for a given byte.
    
    index: the index of the byte on which to perform the attack.
    pts: plaintexts used in encryptions (shape: [num_enc, 16]).
    traces: power measurements (shape: [num_enc, num_samples]).
    
    return: an np.array of key candidates ordered from highest to lowest score.
    """
    # Create an array of candidate keys [0, 1, 2, ..., 255]
    candidate_keys = np.arange(256, dtype=np.uint8)

    # Compute the XOR between the plaintext byte and every candidate key in parallel.
    # pts[:, index][:, None] has shape (num_enc, 1) and candidate_keys[None, :] has shape (1, 256).
    xored = np.bitwise_xor(pts[:, index][:, None], candidate_keys[None, :])  # Shape: (num_enc, 256)

    # Compute the hypothetical intermediate values for all encryptions and all candidate keys.
    # Using broadcasting, m will have shape (num_enc, 256).
    m = sbox[xored]

    # Apply the leakage mask.
    # Here we use: group1_mask = (m & 0b10000000) != 0 and group0_mask = inverse.
    # Both masks have shape (num_enc, 256).
    mask1 = ((m & 0b10000000) != 0).astype(np.float32)
    mask0 = 1 - mask1

    # For each candidate key, compute the sum (and subsequently the mean) of traces
    # for each group. We do this with matrix multiplication.
    # Sum over the encryption (trace) axis.
    # For group1: shape becomes (256, num_samples)
    sum1 = mask1.T @ traces  
    # Similarly for group0.
    sum0 = mask0.T @ traces  

    # Count how many traces fall into each group for each candidate key.
    count1 = mask1.sum(axis=0)  # Shape: (256,)
    count0 = mask0.sum(axis=0)  # Shape: (256,)

    # Avoid division by zero by identifying candidates with empty groups.
    # We'll set these counts to np.nan later so that their final scores become zero.
    count1[count1 == 0] = np.nan
    count0[count0 == 0] = np.nan

    # Compute the mean traces for each candidate key and group.
    # The resulting shapes are (256, num_samples)
    mean1 = sum1 / count1[:, None]
    mean0 = sum0 / count0[:, None]

    # Compute the difference-of-means for each candidate key.
    diff = mean1 - mean0

    # Compute a score per candidate: the maximum absolute difference
    # across all time samples.
    key_scores = np.max(np.abs(diff), axis=1)

    # Replace any nan scores (from division by zero) with 0.
    key_scores = np.nan_to_num(key_scores, nan=0.0)

    # Sort the candidate keys by their score in descending order.
    key_bytes = np.argsort(key_scores)[::-1]
    return key_bytes


def dpa_byte(index, pts, traces):
    """
    index: index of the byte on which to perform the attack.
    pts: the plaintext of each encryption performed. (100, 16)
    traces: the power measurements performed for each encryption. (100, 1400)
    return: an np.array with the key bytes, from highest probable to less probable
    """
    key_scores = np.zeros(256)

    for k in range(256):
        # Compute the hypothetical intermediate values for all encryptions.
        m_i = sbox[pts[:, index] ^ k]

        # Split traces into groups based on m_i.
        group0 = traces[m_i & 0b10000000 == 0, :]  # use rows corresponding to MSB==0
        group1 = traces[m_i & 0b10000000 >= 1, :]  # use rows corresponding to MSB==1

        if group0.size == 0 or group1.size == 0:
            # Avoid computing mean over empty groups.
            key_scores[k] = 0
        else:
            # Compute difference of means for each time sample.
            diff = np.mean(group1, axis=0) - np.mean(group0, axis=0)
            # For example, use the maximum absolute difference as key score:
            key_scores[k] = np.max(np.abs(diff))

    # Sort the key candidates by their score in descending order.
    key_bytes = np.argsort(key_scores)[::-1]
    return key_bytes

def run_full_dpa_known_key(pts, ks, trs, idx_bytes):
    print("Run DPA the with the known key for bytes idx: \n{}".format(idx_bytes))

    key_bytes_found = 16 * [0]
    key_ranks = 16 * [0]
    for i in idx_bytes:
        print("Run DPA for byte {}...".format(i), end="", flush=True)
        dpa_res = vectorized_dpa_byte(i, pts, trs)
        key_bytes_found[i] = dpa_res[0]
        key_ranks[i] = np.where(dpa_res == ks[0, i])[0]
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
    am_tr = min(1000, plaintexts.shape[0])

    plaintexts = plaintexts[:am_tr, :]
    keys = keys[:am_tr, :]
    traces = traces[:am_tr, :]

    # Uncomment the next line to plot the first trace
    #plot_traces(traces[0,:])

    # Preprocess traces
    traces = preprocess_traces(traces)

    # Run the attack
    # Indexes of byte to attack
    idxes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    run_full_dpa_known_key(plaintexts, keys, traces, idxes)
