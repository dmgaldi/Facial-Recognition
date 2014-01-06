import numpy as np
from glob import glob
import Utility
import Plotting
from PIL import Image
from os import path
import os

class EigenFace(object):

    def __init__(self, res=None):
        self.eigenfaces = []
        self.imageNames = []
        self.res = res
        self.mean = 0
        self.currentDirectory = os.getcwd()
        

    def TrainWithImages(self, imageNames, res=None):
        """
        Trains the EigenFace model with a set of images with the same resolution.
        
        Raises a value error if any image has a resolution inconsistent with the res parameter.

        Parameters: imageNames: a python list of image file locations

                    res: a tuple consisting of the length and width of the images in pixels
                    
        Returns: A numpy array of EigenFaces
        """
        
        images = []
        for i in imageNames:
            if path.isdir(i):
                images = images + glob(str(i) + "/*.*")
                os.chdir(self.currentDirectory)
            else:
                images.append(i)
        imageNames = images
                

        N = len(imageNames) #number of Training Images
        X = res[0] #rows of pixels of image
        Y = res[1] #cols of pixels of image

        trainingMatrix = np.zeros(shape=(X * Y, N))
        
        for i, imageName in enumerate(imageNames):
            img = Image.open(imageName).convert('L')
            
            ## If no default resolution is set default to the first image
            if self.res is None:
                self.res = img.size

            if not (img.size[0] == res[0] and img.size[1] == res[1]):
                raise ValueError("Invalid dimensions for image " + str(i))
   
            trainingMatrix[:,i] = np.ndarray.flatten(np.array(img))

        ## Perform PCA (limited by colums of trainingMatrix)
        eigvals, eigvectors, self.mean = Utility.PCA(trainingMatrix, N)

        print np.argsort(eigvals)

        self.res = res
        self.imageNames = imageNames
        self.eigenfaces = eigvectors.copy()

    def TrainSingleImage(self, imageName, imageLabel):
        img = Image.open(imageName).convert('L')
        if self.res is None:
            self.res = (img.size[0], img.size[1])
            self.imageNames.append(imageName)
        else:
            if not (img.size[0] == res[0] and img.size[1] == res[1]):
                raise ValueError("Invalid dimensions for new image")
            else:
                self.imageNames.append(imageName)
                self.TrainWithImages(self.imageNames, res)
                
    def AssessImage(self, imageName):
        """
        Projects the query image into the subspace obtained from PCA. Determines
        
        which of the training images is closest in terms of Euclidean distance
        """
        queryImg = Image.open(imageName).convert('L')
        if self.res is None:
            raise ValueError("The model has not yet been trained")
        elif self.res != queryImg.size:
            raise ValueError("Invalid dimensions for new image")
        else:
            queryAry = np.ndarray.flatten(np.array(queryImg))
            shortestDist = 1e300
            nearest = None
            #print Utility.Project(queryAry, self.eigenfaces, self.mean)
            
            for i, name in enumerate(self.imageNames):
                img = Image.open(name).convert('L')
                x = Utility.Project(np.ndarray.flatten(np.array(img))/np.linalg.norm(np.ndarray.flatten(np.array(img))), self.eigenfaces, self.mean) - Utility.Project(queryAry/np.linalg.norm(queryAry), self.eigenfaces, self.mean)
                dist = np.dot(x.T, x)
                print dist
                if dist < shortestDist:
                    shortestDist = dist
                    nearest = self.imageNames[i]
            
            Plotting.ShowImage(np.array(Image.open(imageName).convert('L')), "Query Image")
            x = Utility.Project(np.ndarray.flatten(np.array(img)), self.eigenfaces, self.mean)
            Plotting.ShowImage(np.array(Image.open(nearest).convert('L')), "Matching Image")
            #print "This is most likely a picture of " + person

                
    def Reset(self):
        self.eigenfaces = []
        self.imageNames = []
        self.res = None
        self.mean = 0

ef = EigenFace()
ef.TrainWithImages(["Images/Yalefaces"], (320, 243))
#Plotting.ShowImage(np.resize(Utility.Normalize(ef.eigenfaces[:,0]), (402, 402)), "Test")
Plotting.ShowImage(Utility.Normalize(np.resize(ef.eigenfaces[:,6], ef.res)), "title")
ef.AssessImage("Images/subject02.happy.gif")

