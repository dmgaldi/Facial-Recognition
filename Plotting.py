import matplotlib.cm as cm
import matplotlib.pyplot as plt

def ShowImage(m, title):
    plt.imshow(m, cmap = cm.Greys_r)
    plt.show()

