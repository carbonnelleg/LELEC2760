from aes import AES
import numpy as np

def print_round_keys(aes):
    """
        Print the round keys
    """
    for i in range(aes.n_rounds+1):
        print(f"Round key {i+1}")
        print(aes._key_matrices[i].T)

def random_bit_pattern(locx,locy):
    """ 
        Generate pattern for random bit faults
        locx: position of the fault in the AES state
        locy: position of the fault in the AES state
    """
    f_pattern = np.zeros((8,4,4),dtype=np.uint8)
    for i in range(8):
        f_pattern[i][locx][locy] = 1<<i
    return f_pattern 

def random_byte_pattern(locx,locy):
    """ 
        Generate pattern for random byte faults
        locx: position of the fault in the AES state
        locy: position of the fault in the AES state
    """
    f_pattern = np.zeros((255,4,4),dtype=np.uint8)
    for i in range(1,256):
        f_pattern[i-1][locx][locy] = i
    return f_pattern


if __name__ == "__main__":
    np.random.seed(0)

    # generate random key and plaintext
    key = np.random.randint(0,256,16)
    pt = np.random.randint(0,256,16)
    
    # generate fault pattern. 
    f_pattern = random_bit_pattern(locx = 0, locy = 3)
    
    # create the AES
    aes = AES(key)

    # encrypt without fault
    ct = aes.encrypt_block(pt)
    
    # Encrypt with fault in the 9-th round before the last Sbox
    ct_fault = aes.encrypt_block_fault(pt,f_round = 9, f_loc = "sbox",
            f_pattern=f_pattern)

    print_round_keys(aes)
    print("--------------------")
    print("Correct ciphertext")
    print(ct)
    print("Fault ciphertext")
    print(ct_fault)
    print("Difference")
    diff = ct_fault ^ ct
    print(diff)
            
