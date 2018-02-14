


def preProcessImage(img):
    '''
    The image has to be swapped in order to match the pytorch setting
    '''
    return(img.transpose(2, 0, 1))

