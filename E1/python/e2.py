import numpy as np
from utils import dec2bits, bits2dec


def compute_bias_table(sbox, l):
    """
    Computes the biases table of the input S-box.

    The first dimension (row) is the input mask.
    The second dimension (column) is the output mask.
    Each entry is a probability bias comprised between -0.5 and +0.5.

    Parameters
    ----------
    sbox : np.ndarray[np.uint8]
        substitution table

    l : int
        input/output mask length ( = 2**n where S-box has n inputs and outputs)

    Returns
    -------
    bias_table : np.ndarray[float]
        bias table

        first dimension in table is input mask

        second dimension in table is output mask
    """

    table = np.full((l, l), -8.0)
    for x in range(l):
        y = sbox[x]
        for i in range(l):
            for j in range(l):
                bool_eq = dec2bits([i, j]) & dec2bits([x, y])
                table[i][j] += np.bitwise_xor.reduce(bool_eq)
    return table/l


def print_bias_table(table):
    l = len(table)
    print("    |" + "|".join(["%8x" % (x) for x in range(l)]))
    for i in range(l):
        print("%4x|" % (i) + "|".join([" %6.3f " % (x) for x in table[i, :]]))


if __name__ == "__main__":

    sbox = np.load("sbox.npz")["sbox"]
    bias_table = compute_bias_table(sbox, 16)

    print_bias_table(bias_table)

    np.save("my_bias_table.npy", bias_table)
