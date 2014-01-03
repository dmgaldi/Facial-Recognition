import numpy as np
import Utility
from PIL import Image

class EigenFace(object):

    def __init__(self):
        self.eigenfaces = []
        self.imageNames = []
        self.res = None

    def TrainWithImages(self, imageNames, res=None):
        """
        Trains the EigenFace model with a set of images with the same resolution

        Parameters: imageNames: a python list of image file locations

                    res: a tuple consisting of the length and width of the images in pixels
                    
        Returns: A numpy array of EigenFaces
        """
        
        assert(res is not None)

        N = len(imageNames) #number of Training Images
        X = res[0] #rows of pixels of image
        Y = res[1] #cols of pixels of image

        trainingMatrix = np.zeros(shape=(X * Y, N))
        
        for i, imageName in enumerate(imageNames):
            img = Image.open(imageName).convert('L')

            if not (img.size[0] == res[0] and img.size[1] == res[1]):
                raise ValueError("Invalid dimensions for image " + str(i))
   
            trainingMatrix[:,i] = np.ndarray.flatten(np.array(img))

        ## Perform PCA (limited by colums of trainingMatrix)
        eigvals, eigvectors = Utility.PCA(trainingMatrix, N)

        self.res = res
        self.imageNames = imageNames
        self.eigenfaces = eigvectors.copy()

    def TrainSingleImage(self, imageName):
        img = Image.open(imageName.convert('L'))
        if self.res is None:
            self.res = (img.size[0], img.size[1])
            self.imageNames.append(imageName)
        else:
            if not (img.size[0] == res[0] and img.size[1] == res[1]):
                raise ValueError("Invalid dimensions for new image")
            else:
                self.imageNames.append(imageName)
                self.TrainWithImages(imageNames, res)

ef = EigenFace()
ef.TrainWithImages(["Images/zuck.jpg", "Images/gates.jpg", "Images/brin.jpg"], (402, 402))
print ef.eigenfaces
