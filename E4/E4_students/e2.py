from main import random_byte_pattern
from aes import AES, sbox_inv, xtime, mix_single_column, matrix2bytes 
import numpy as np

def encrypt_fault_SB9(aes_obj,pt,n,locx,locy):
    """
        Encrypt a plaintext with a random byte fault injection in the 
        Sbox layer of the 9th round.
        aes_obj: AES, object, similar to encrypt.
        pt, plaintext, as a numpy array of np.uint8.
        n: amount of faulty ciphertext generated 
        locx: [0-3], line chosen to inject the fault
        locy: [0-3], column chosen to inject the fault.

        Return a list, where the first element is the correct
        ciphertext, and where the second element is a list 
        of n faulty ciphertexts.
    """

    fp = random_byte_pattern(locx,locy)
    ctfs = n * [None]
    for i in range(n):
        ctfs[i] = aes_obj.encrypt_block_fault(
                pt,
                f_pattern=fp,
                f_loc="sbox",
                f_round=9
                )
    return ctfs

def compute_ciphertext_fpositions(locxSB9,locySB9):
    """
        Compute the positions of ciphertext faulty bytes considering 
        a fault injection in the Sbox layer of round 9.

        Return a list of 4 tuples, each storing the line and the column indexes.
    """

    # Compute which columns will be faulty after MC
    col_atck_post_MC = (locySB9+ 4 - locxSB9)%4    
    # Compute the faults positions in the ciphertext
    pos = [(i,(col_atck_post_MC+4-i)%4) for i in range(4)]
    return pos

def generate_faults_patterns_postMC(locxSB9):
    """
        Generate all the possible faults patterns after the MC operation, considering
        a fault occuring at a specific line.
        locxSB9: The line index of the fault. 

        Return a list of tuples, each of tuples being a possible error patterns. 
    """
    #TODO
    return [(0,0,0,0)]


def compute_4tuples_subkeys(ct,ctf,fpos,fpatterns):
    """
        Generate all the 4-bytes subkeys possibilities, taking into account
        the fault positions in the ciphertext and the fault patterns.
        ct: correct ciphertext matrix.
        ctf: a faulty ciphertext matrix, resulting of the same encryption process as the correct one.
        fpos: list of position of faulty bytes in the ciphertext matrix.
        fpatterns: possible faults patterns.

        Return a list of tuples, each tuples being composed of 4 bytes values. Each tuple
        is a possible subkey of 32-bits.
    """

    # TODO 
    return [(0,0,0,0)]

def attack_rndbyte_SB9(ct,ctfs,locxSB9,locySB9):
    """
        Attack a subkey of 32-bits.
        ct: correct ciphertext matrix.
        ctfs: a list of faulty ciphertext matrix, for the same inpus as the correct one.
        locxSB9: [0-3], line where fault(s) have been injected.
        locySB9: [0-3], column where the fault(s) have been injected.
    """

    #TODO
    return [(0,0,0,0)]

if __name__ == "__main__":
    # Initialise aes object
    key = np.random.randint(0,256,16)
    aes = AES(key)

    # Generate attack cases
    locx_fault = 2
    locy_fault = 3
    n = 2

    pt = np.random.randint(0,256,16)
    ct = aes.encrypt_block(pt)
    ctfs = encrypt_fault_SB9(aes,pt,n,locx_fault,locy_fault)

    #########
    # Get the real value of the key
    fpos = compute_ciphertext_fpositions(locx_fault,locy_fault)
    last_key = aes._key_matrices[-1].T
    targeted_byte = [
            last_key[fpos[i][0]][fpos[i][1]]
            for i in range(4)
                ]
    ttarget = tuple(targeted_byte)

    # Compute the set of possibles keys remaining
    setk = attack_rndbyte_SB9(ct,ctfs,locx_fault,locy_fault)

    # Print result
    print("Using key:\n{}".format(key))
    print("Using plaintext:\n{}".format(pt))
    print("")
    print("Correct ciphertext:\n{}".format(ct))
    print("")
    print("Last round keys:\n{}".format(aes._key_matrices[-1].T))
    print("")
    print("Random key byte injected on byte [{},{}] at SB9".format(locx_fault,locy_fault))
    for i in range(n):
        print("Faulty ciphertext n°{}:\n{}\n".format(i,ctfs[i]))
    print("")
    print("Target subkey: {}".format(ttarget))
    print("Subkey found after the attack:")
    for ei,e in enumerate(setk):
        print("Skey n°{} -> {}".format(ei,e))
    if len(setk)>1:
        print("CAUTION: more than 1 subbyte key isolated...")
    print("")
    if(ttarget in setk):
        print("Successfully isolated the correct subkey.")
    else:
        print("Fail to isolate the correct subkey...")
