# tooYoungTooSimple
# this script is to combine the small images to a large movie
# the imsave function is different in tifffile and image packages
# be careful

import tifffile
import numpy as np
from glob import glob
from scipy.misc import imread

# read multiple tiff files from a certain folder
files = sorted(glob("/Users/xiaodong/Downloads/neurofinder.00.00/images/*.tiff"))
imgs = np.array([imread(f) for f in files])

tifffile.imsave('haha2.tif', imgs)
