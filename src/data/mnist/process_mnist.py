__author__ = 'Charlie'


from mnist import MNIST
import numpy as np
from scipy.misc import imresize


mndata = MNIST('./raw')
imgs, clss = mndata.load_training()
imgs2, clss2 = mndata.load_testing()

#we dont care about test/training here, so we'll use all of the images
imgs.append(imgs2)
clss.append(clss2)


#convert to numpy arrays
imgs = np.array(imgs, dtype=np.float64)
clss = np.array(clss, dtype=np.int32)


#reshape to
imgs = np.reshape(imgs, (-1, 28, 28))

#now save scaled images with
scld = np.zeros((70000, 48, 48), dtype=np.float64)
for index, img in enumerate(imgs):
    scld[index,:,:] = imresize(img, (48, 48))


np.save('mnist-images.npz')
np.save('mnist-classes.npy')
