from main import random_bit_pattern,print_round_keys
from aes import AES,sbox_inv
import numpy as np

def hw(x):
    """ 
        Computes Hamming Weight of an integer
    """
    y = 0
    while x > 0:
        y += x & 0x1
        x = x >> 1
    return y

def gen_faults_bit_flip_last_round(aes,n,locx,locy):
    """
        Generates n bit flip at location locx locy

        aes: AES object
        n: number of correct / faulty ciphertexts
        locx : position in AES state matrix
        locy: position in AES state matrix
    """
    f_pattern = random_bit_pattern(locx,(locy + locx)%4)
    cts = [None for _ in range(n)]
    cts_f = [None for _ in range(n)]

    for i in range(n):
        pt = np.random.randint(0,256,16)
        ct = aes.encrypt_block(pt)
        f_ct = aes.encrypt_block_fault(pt,f_pattern=f_pattern,
                f_loc="sbox",f_round=10)
        cts[i] = ct
        cts_f[i] = f_ct
    return cts,cts_f

def attack_bit_flip_last_round(cts,cts_f,locx,locy):
    """
        performs attack with bit flip faults before the last
        Sbox of the AES. Returns the key byte at the position locx locy
        
        cts : correct ciphertexts 
        cts_f: corresponding faulty ciphertexts
        locx : position of the byte to recover 
        locy : position of the byte to recover 
    """
    #TODO
    return 0


if __name__ == "__main__":
    np.random.seed(0)
    # generate a random key and init the AES 
    key = np.random.randint(0,256,16)
    aes = AES(key)
    
    n_faults = 16 # TODO: what number should we put here ?

    # Where the key will be stored
    last_round_key = np.zeros((4,4), dtype=np.uint8)

    for locx in range(4): # key position to recover
        for locy in range(4): # key position to recover
            # generate the faults
            cts,cts_f = gen_faults_bit_flip_last_round(aes,n_faults,locx,locy)
            # recover key byte
            last_round_key[locx,locy] = attack_bit_flip_last_round(cts,cts_f,locx,locy)

    print("Recovered last round key:")
    print(last_round_key)
    print("Expected last round key:")
    print(aes._key_matrices[-1].T)
    assert(np.all(last_round_key == aes._key_matrices[-1].T))
