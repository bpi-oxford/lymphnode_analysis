'''
Code to try and identify the cell type from a simple staining. Will take a segmentation 
and use it to compare signal in the two channels.

The idea to to take a population of cells - find there corresponding signal in the red channel
(e.g. antibody staining). Then plot the signal histogram - expect to see two populations.

Inputs: 

2 channel raw image
segmentation

Outputs:

A prediction for each cell of which class it belongs too.

'''
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_dilation

def register_image():
    return


def classify_track(labelled_video , track_df , raw_2channel_image):
    '''
    Takes in a mask video and classifies each frame of the video
    '''
    track_id = track_df['track_id'].unique()
    for i in range(len(track_df['t'])):
        labelled_image = labelled_video[i,...]
        x = track_df.iloc[i]['x']
        y = track_df.iloc[i]['y']
        z = track_df.iloc[i]['y']
        single_cell_image = (labelled_image == track_id).astype(int)
    return

def get_2channel_intensity(single_labelled_image, red_channel):
    # Create a mask for the region just outside the cell
    dilated_image = binary_dilation(single_labelled_image, iterations=5)
    outer_region = dilated_image & ~single_labelled_image

    # Calculate the mean intensity in the outer region
    int_intensity = (outer_region * red_channel).sum() / outer_region.sum()
    return int_intensity

def classify_video(labelled_video , track_df , raw2channel_image):
    '''
    classifies the cells based on the second channel - appends to the tracks
    '''
    #initialise the track classification as 0
    track_df['classification'] = [0] * len(track_df)

    num_frames = labelled_video.shape[0]
    for i in range(num_frames):
        i
    return

def image_histogram(labelled_image , red_channel):
    labels = np.unique(labelled_image)
    intensities = np.zeros_like(labels)
    count = 0
    for label in labels:
        single_labelled_image =  (labelled_image == label).astype(int)
        second_channel_int_intensity = get_2channel_intensity(single_labelled_image , red_channel)
        intensities[count] = second_channel_int_intensity
        print(label)
        print(intensities[count])
        count+=1



    # Plot the histogram
    plt.hist(intensities)
    plt.xlabel('intensity')
    plt.ylabel('count')
    plt.title('Histogram of Red Channel Intensities')

    # Save the plot
    plt.savefig(r'/mnt/Work/Group Fritzsche/Ed/intensity_histogram.png')
    plt.close()

if __name__ == "__main__":
    label_path = r'/home/edwheeler/Documents/cropped_region_5_Bcells/for_segmentation/b2-2a_2c_pos6-01_deskew_cgt_crop_5_2channel1.t0_seg_masks.tif'
    labels = tiff.imread(label_path)
    raw_2channel_path = r'/home/edwheeler/Documents/cropped_region_5_Bcells/b2-2a_2c_pos6-01_deskew_cgt_crop_5_2channel1.tiff'
    raw_2channel = tiff.imread(raw_2channel_path)
    print(raw_2channel.shape)
    #labels = labels[0:70, :, :]
    #raw_2channel = raw_2channel[:, : , 0:70,:,:]
    image_histogram(labels , raw_2channel[0,0,...])
