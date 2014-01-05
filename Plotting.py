import matplotlib.cm

def ShowImage(m, title):
    cm.subplot(title=title, images=[m], rows=1, cols=1, 
            sptitle="Eigenface", colormap=cm.jet)

