import numpy as np
import matplotlib.pyplot as plt

def load_npz(filename):
    dic_load = {}
    with open(filename,'rb') as f:
        coll = np.load(f,allow_pickle=True)
        for k in coll.keys():
            dic_load[k] = coll[k]
    return dic_load

def dec2bits(dec,size=8):
    dec = np.array(dec, dtype=np.uint8)
    s = list(dec.shape)
    s[0] = s[0] * size
    bits = np.zeros(tuple(s), dtype=np.uint8)
    for i, x in enumerate(dec):
        for j in range(size):
            bits[i * size + j] = (x >> (size - 1 - j)) & 0x1
    return bits

def plot_traces(traces):
    f = plt.figure("Traces plot")
    plt.title("Traces")
    plt.xlabel("Time index")
    plt.ylabel("Power value")
    plt.plot(traces.T)
    plt.show()


if __name__ == "__main__":
    dic_load = load_npz("attack_set_known_key.npz")
    print(dic_load)
    for k in dic_load.keys():
        print(k)
