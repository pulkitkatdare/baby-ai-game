import torch
import numpy as np
import os,sys,inspect


class PreProcessor(object):
    def __init__(self,
                 modelForSentenceEmbedding=False):
        
        self.modelForSentenceEmbedding=modelForSentenceEmbedding
        
        if self.modelForSentenceEmbedding:
            print(sys.path[0])

            currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            parentdir = os.path.dirname(currentdir)+'\model'
            sys.path.insert(0,parentdir) 
            print(sys.path[0])
            import sentenceEmbedder
            self.languageModel=sentenceEmbedder.Sentence2Vec()
            sys.path=sys.path[1:]            
            print('language model loaded')
        
    def preProcessImage(self,img):
        '''
        The image has to be swapped in order to match the pytorch setting
        
        For now we use a env wrapper to swap the dimensions but once we will get rid of it,
        this should be implemented
        '''
        return(img)
        
        #return(img.transpose(2, 0, 1))
    
    
    
    def stringEncoder_LanguageModel(self,string):
        ''' 
        encode a string or a sequence of strings using the ASCII encoding
        '''    
        return(self.modelForSentenceEmbedding.encodeSent(string))
    
    
    
    def stringEncoder(self,string,maxSizeOfMissions=200):
        ''' 
        encode a string using the ASCII encoding
        '''
        code=np.zeros(maxSizeOfMissions)
        for i in range(len(string)):
            code[i]=ord(string[i])
            return(torch.from_numpy(code))
            
    def stringDecoder(self,code):
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