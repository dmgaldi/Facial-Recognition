import numpy as np
from PIL import Image

class EigenFace(object):

    def __init__(self):
        self.eigenfaces = []

    def TrainWithImages(self, imageNames, res=None):
        """
        Trains the EigenFace model with a set of images with the same resolution

        Parameters: imageNames: a python list of image file locations

                    res: a tuple consisting of the length and width of the images in pixels
                    
        Returns: A numpy array of EigenFaces
        """
        
        assert(res is not None)

        N = len(imageNames) #number of Training Images
        X = res[0]
        Y = res[1]

        trainingMatrix = np.zeros(shape=(X * Y, N))
        
        for i, imageName in enumerate(imageNames):
            img = Image.open(imageName).convert('L')

            if not (img.size[0] == res[0] and img.size[1] == res[1]):
                raise ValueError("Invalid dimensions for image " + str(i))
   
            trainingMatrix[:,i] = np.ndarray.flatten(np.array(img))

            mean = np.mean(trainingMatrix, axis=0)

            print trainingMatrix
            print mean
        
        
    
ef = EigenFace()
ef.TrainWithImages(["../Images/zuck.jpg"], (634, 396))
