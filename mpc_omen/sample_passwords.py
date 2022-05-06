import string
import sys
from tqdm import tqdm
import random
import numpy as np

def help():
    print("-"*5, "HELP".center(20), "-"*5)
    print("alphabet".rjust(15), ": 0-3")
    print("length".rjust(15), ": int")
    print("sample number".rjust(15), ": int")
    print("output file name".rjust(15), ": string")

def get_alphabet(index):
    if index == 0:
        return string.digits[1:]
    elif index == 1:
        return string.ascii_letters
    elif index == 2:
        return string.punctuation
    else:
        raise NotImplementedError

def sample_single_word(alphabet, length):
    pwd = ""
    rand_indexs = np.random.choice(len(alpbabet), length, replace=False)
    for i in range(length):
        pwd += alpbabet[rand_indexs[i]]
    return pwd

if __name__ == '__main__':
    if len(sys.argv)!=5:
        help()
        sys.exit()

    base_path = "./pwd_data/"

    alphabet_index = int(sys.argv[1])
    length = int(sys.argv[2])
    file_name = sys.argv[3]
    sample_number = int(sys.argv[4])

    print("Alphabet index: {}, Max length is {}, Output file name is {}".format(alphabet_index, length, base_path+file_name))

    out_file = open(base_path+file_name, 'w')
    alpbabet = get_alphabet(alphabet_index)

    for i in tqdm(range(sample_number)):
        pwd = sample_single_word(alpbabet, length)
        out_file.write(pwd)
        out_file.write("\n")

    print("END".center(20))


