import numpy as np
from PIL import Image

class EigenFace(object):

    def __init__(self):
        pass

    def TrainWithImage(self, imageName):
        img = Image.open(imageName).convert('L')
        imgAry = np.array(img)
        print imgAry[:,1]
    
ef = EigenFace()
ef.TrainWithImage("../Images/zuck.jpg")
