import numpy as np
from PIL import Image

class EigenFace(object):

    def __init__(self):
        pass

    def TrainWithImages(self, imageNames, res=None):
        assert(res is not None)
        trainingMatrix = np.zeros(shape=(res[0] * res[1], len(imageNames)))
        
        for i, imageName in enumerate(imageNames):
            img = Image.open(imageName).convert('L')        
            trainingMatrix[:,i] = np.array(img)
    
ef = EigenFace()
ef.TrainWithImage(["../Images/zuck.jpg"])
