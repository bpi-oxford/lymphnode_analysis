import napari
import ultrack
from ultrack import MainConfig, Tracker,add_flow, segment, link, solve, to_tracks_layer, tracks_to_zarr, to_ctc
import tifffile as tiff
import pandas as pd
from ultrack.imgproc.flow import timelapse_flow, advenct_from_quasi_random, trajectories_to_tracks
from ultrack.utils.cuda import on_gpu
from ultrack.utils.edge import labels_to_contours
from ultrack.tracks.stats import tracks_df_movement
import numpy as np
# import to avoid multi-processing issues
if __name__ == "__main__":

   # Load your data
   label_path = r"/home/edwheeler/Documents/cropped_region_1/seg_frames/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentation_filtered_masks_stack.tif"
   labels = tiff.imread(label_path)
   raw_data_path = r"/home/edwheeler/Documents/cropped_region_1/raw_video/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentation.tif"
   raw_data = tiff.imread(raw_data_path) 
   
   #take a test region
   #labels   = labels[0:10 ,10:50, ...]
   raw_data = raw_data[0:10, ...]

   print(labels.shape)
   n_workers = 6
   # Create a config
   config = MainConfig()
   print(config)
   print(ultrack.config.MainConfig)
   config.segmentation_config.n_workers = n_workers
   config.linking_config.max_distance = 25
   config.linking_config.n_workers = n_workers
   # this removes irrelevant segments from the image
   # see the configuration section for more details
   config.segmentation_config.min_frontier = 0.05
   config.tracking_config.division_weight = -10
   config.tracking_config.appear_weight = -1
   config.tracking_config.disappear_weight = -1
   config.tracking_config.image_border_size = (1, 10, 10)
   print(config)
   #flow?

   flow = timelapse_flow(raw_data, store_or_path="flow.zarr", n_scales=3, lr=1e-2, num_iterations=2_000)
   napari_viewer = napari.Viewer()
   napari_viewer.add_image(flow)
   napari_viewer.add_image(
    flow,
    contrast_limits=(-0.001, 0.001),
    colormap="turbo",
    visible=False,
    scale=(4,) * 3,
    channel_axis=1,
    name="flow field",
    )
   
   foreground, edges = labels_to_contours(labels.astype(np.int16))


   trajectory = advenct_from_quasi_random(flow, foreground.shape[-3:], n_samples=1000)
   flow_tracklets = pd.DataFrame(
      trajectories_to_tracks(trajectory),
      columns=["track_id", "t", "z", "y", "x"],
   )
   flow_tracklets[["z", "y", "x"]] += 0.5  # napari was crashing otherwise, might be an openGL issue
   flow_tracklets[["dz", "dy", "dx"]] = tracks_df_movement(flow_tracklets)
   flow_tracklets["angles"] = np.arctan2(flow_tracklets["dy"], flow_tracklets["dx"])

   flow_tracklets.to_csv("flow_tracklets.csv", index=False)

   napari_viewer.add_tracks(
      flow_tracklets[["track_id", "t", "z", "y", "x"]],
      name="flow vectors",
      visible=True,
      tail_length=25,
      features=flow_tracklets[["angles", "dy", "dx"]],
      colormap="hsv",
   ).color_by="angles"
   napari.run()
      

   segment(foreground, edges , config, overwrite=True)
   add_flow(config, flow)
   link(config, overwrite=True)
   tracks= solve(config)
   print(tracks)
   print('segmented')

   '''
   # Run the tracking
   tracker = Tracker(config=config)
   tracker.track(foreground=foreground, contours=edges, overwrite = True)

   # Visualize the results
   tracks, graph = tracker.to_tracks_layer()
   print(tracks)
   tracks.to_csv( 'tracks.csv')

   napari.view_tracks(tracks[["track_id", "t","z", "y", "x"]], graph=graph)
   napari.run()
   '''