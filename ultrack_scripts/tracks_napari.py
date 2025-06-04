import napari
import pandas as pd
import tifffile as tiff

#labelled_mask = tiff.imread(r"opening_frame_labelled_mask.tif")
#print(labelled_mask.shape)
#raw_data = tiff.imread(r"/home/edwheeler/Documents/cellpose_final_output/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentation.tif")

tracks = pd.read_csv(r'/home/edwheeler/Documents/cropped_region_1/seg_frames/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentationt36_seg_ma_tracks.csv')
properties = {
    'track_id': tracks['track_id'],

}
napari_viewer = napari.Viewer()
#napari_viewer.add_image(labelled_mask, name='labelled_mask', colormap='gray')
#napari_viewer.add_image(raw_data, name='raw_data', colormap='gray')
napari_viewer.add_tracks(tracks[["track_id", "t","z", "y", "x"]], properties=properties, graph=None)
napari.run()