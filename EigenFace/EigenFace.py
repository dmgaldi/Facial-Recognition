import numpy as np
import Image
from Image import 

class EigenFace(object):

    def TrainWithImage(self, imageName):
        img = Image.open(imageName).convert('RGBA')
        imgAry = np.array(img)
    