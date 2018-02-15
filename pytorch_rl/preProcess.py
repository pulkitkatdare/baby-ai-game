import torch
import numpy as np


def preProcessImage(img):
    '''
    The image has to be swapped in order to match the pytorch setting
    '''
    return(img.transpose(2, 0, 1))



def stringEncoder(string,maxSizeOfMissions=200):
    ''' 
    encode a string using the ASCII encoding
    '''
    code=np.zeros(maxSizeOfMissions)
    for i in range(len(string)):
        code[i]=ord(string[i])
    return(torch.from_numpy(code))


def stringDecoder(code):
    '''
    decode a pytorch Tensor containing the ASCII codes of the original string
    '''
    string=''
    for x in code:
        if x==0:
            return(string)
        else:
            string+=chr(int(x))
    return(string)