from cellpose import io, models, utils
import matplotlib.pyplot as plt
import numpy as np

seg_path = r"/home/edwheeler/Documents/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentationt1_seg.npy"

img = io.imread(r"/home/edwheeler/Documents/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentationt1.tif")
#img = img.astype('uint8')
dat = np.load(seg_path , allow_pickle = True).item()
masks = dat['masks']
masks = masks.astype('uint8')

outlines = utils.outlines_list(masks[:,:,40])
plt.imshow(img[:,:,40], cmap = 'gray')
for o in outlines:
    plt.plot(o[:,0] , o[:,1])
plt.show()
