import segment3D.parameters as uSegment3D_params
import segment3D.usegment3d as uSegment3D
import os
from cellpose import models, io
import numpy as np
from skimage.io import imsave
import tifffile as tiff

def segment_by_slices(image3D ,custom_model_path,voutput_dir, view):
   model = models.CellposeModel(gpu=True)# , pretrained_model  = custom_model_path)
   all_probs = []
   all_flows = []
   if view == 'xy':
      z_length = image3D.shape[0]
      for i in range(z_length):
         print(i)
         frame = image3D[i,...]
         print(frame.shape)
         masks, flows, styles = model.eval(frame, batch_size=256, compute_masks=True)
         print([f.shape for f in flows])
         all_probs.append(flows[2])
         all_flows.append(flows[0])

         # Save the probability map (flows[1]) for each slice
         prob_output_path = os.path.join(output_dir, f'prob_map_slice_{i}.tif')
         imsave(prob_output_path, flows[2])  # Save as 32-bit TIFF
         #print(f"Probability map for slice {i} saved to {prob_output_path}")

         # Save the frame for each slice
         frame_output_path = os.path.join(output_dir, f'frame_slice_{i}.tif')
         imsave(frame_output_path, frame.astype(np.uint16))  # Save as 16-bit TIFF
         #print(f"Frame for slice {i} saved to {frame_output_path}")


   elif view == 'xz':
      image3D = image3D.transpose(1, 0, 2)
      z_length = image3D.shape[0]
      for i in range(z_length):
         print(i)
         frame = image3D[i,...]
         print(frame.shape)
         masks, flows, styles = model.eval(frame, batch_size=8, compute_masks=True)
         print([f.shape for f in flows])
         all_probs.append(flows[2])
         all_flows.append(flows[0])

         # Save the probability map (flows[1]) for each slice
         prob_output_path = os.path.join(output_dir, f'prob_map_slice_{i}.tif')
         imsave(prob_output_path, flows[2])  # Save as 32-bit TIFF
         #print(f"Probability map for slice {i} saved to {prob_output_path}")

         # Save the frame for each slice
         frame_output_path = os.path.join(output_dir, f'frame_slice_{i}.tif')
         imsave(frame_output_path, frame.astype(np.uint16))  # Save as 16-bit TIFF
        # print(f"Frame for slice {i} saved to {frame_output_path}")

   elif view == 'yz':
      image3D = image3D.transpose(2, 0, 1)
      z_length = image3D.shape[0]
      for i in range(z_length):
         print(i)
         frame = image3D[i,...]
         print(frame.shape)
         masks, flows, styles = model.eval(frame, batch_size=8, compute_masks=True)
         print([f.shape for f in flows])
         all_probs.append(flows[2])
         all_flows.append(flows[0])

         # Save the probability map (flows[1]) for each slice
         prob_output_path = os.path.join(output_dir, f'prob_map_slice_{i}.tif')
         imsave(prob_output_path, flows[2])  # Save as 32-bit TIFF
         #print(f"Probability map for slice {i} saved to {prob_output_path}")

         # Save the frame for each slice
         frame_output_path = os.path.join(output_dir, f'frame_slice_{i}.tif')
         imsave(frame_output_path, frame.astype(np.uint16))  # Save as 16-bit TIFF
         #print(f"Frame for slice {i} saved to {frame_output_path}")
      
   all_probs = np.array(all_probs, dtype=np.float32)
   all_flows = np.array(all_flows, dtype=np.float32)
   all_flows = all_flows.transpose(3, 0, 1, 2)

   if view == 'xy':
        all_flows = all_flows.transpose(0,1,2,3)
        
   if view == 'xz':
        all_probs = all_probs.transpose(1,0,2)
        all_flows = all_flows.transpose(0,2,1,3) # the first channel is the flow!.
        
   if view == 'yz':
        all_probs = all_probs.transpose(1,2,0)
        all_flows = all_flows.transpose(0,2,3,1) # the first channel is the flow!.
      

   # Save all_probs and all_flows to the output directory as TIFF files
   probs_output_path = os.path.join(output_dir, f'all_probs_{view}.tif')
   flows_output_path = os.path.join(output_dir, f'all_flows_{view}.tif')

   imsave(probs_output_path, all_probs.astype(np.float32))  # Save as 32-bit TIFF
   imsave(flows_output_path, all_flows.astype(np.float32))  # Save as 32-bit TIFF

   print(f"All probabilities saved to {probs_output_path}")
   print(f"All flows saved to {flows_output_path}")

   print(all_probs.shape)
   print(all_flows.shape)
   return all_probs ,all_flows




if __name__ == "__main__":
   # Define paths
   input_image_path = r'/home/edwheeler/Documents/cropped_region_1/raw_frames/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentationt0.tif'  # Replace with the path to your input image
   output_dir = r'/mnt/Work/Group Fritzsche/Ed/u_segment_tests/'  # Replace with the path to your output directory



   # Load the image
   image = io.imread(input_image_path)
   print(image.shape)
   #image = image[30:35,30:40,30:40]

   print(image.shape)
   # Initialize Cellpose model
   custom_model_path = r"/home/edwheeler/Documents/training_data/train/models/CP_20250430_181517"

   segment = True
   if segment:
      img_segment_2D_xy_probs, img_segment_2D_xy_flows = segment_by_slices(image, custom_model_path, output_dir, view='xy')
      img_segment_2D_xz_probs, img_segment_2D_xz_flows = segment_by_slices(image, custom_model_path, output_dir, view='xz')
      img_segment_2D_yz_probs, img_segment_2D_yz_flows = segment_by_slices(image, custom_model_path, output_dir, view='yz')
   else:
      # Load all_probs and all_flows from saved files
      img_segment_2D_xy_probs = tiff.imread(os.path.join(output_dir, 'all_probs_xy.tif'))
      img_segment_2D_xy_flows = tiff.imread(os.path.join(output_dir, 'all_flows_xy.tif'))
      img_segment_2D_xz_probs = tiff.imread(os.path.join(output_dir, 'all_probs_xz.tif'))
      img_segment_2D_xz_flows = tiff.imread(os.path.join(output_dir, 'all_flows_xz.tif'))
      img_segment_2D_yz_probs = tiff.imread(os.path.join(output_dir, 'all_probs_yz.tif'))
      img_segment_2D_yz_flows = tiff.imread(os.path.join(output_dir, 'all_flows_yz.tif'))
      # Print shapes of loaded probability and flow arrays
      print("Shape of img_segment_2D_xy_probs:", img_segment_2D_xy_probs.shape)
      print("Shape of img_segment_2D_xy_flows:", img_segment_2D_xy_flows.shape)
      print("Shape of img_segment_2D_xz_probs:", img_segment_2D_xz_probs.shape)
      print("Shape of img_segment_2D_xz_flows:", img_segment_2D_xz_flows.shape)
      print("Shape of img_segment_2D_yz_probs:", img_segment_2D_yz_probs.shape)
      print("Shape of img_segment_2D_yz_flows:", img_segment_2D_yz_flows.shape)


   # instantiate default parameters
   aggregation_params = uSegment3D_params.get_2D_to_3D_aggregation_params()
   aggregation_params['combine_cell_probs']['cellpose_prob_mask'] = True 
   aggregation_params['combine_cell_probs']['threshold_n_levels'] = 3

   print('========== Default 2D-to-3D aggregation parameters ========')
   print(aggregation_params)    
   print('============================================')

   # integrate labels_xy, labels_xz, labels_yz into one single 3D segmentation. Give a single-channel volume image, img we define its xy view as img, its xz view as img.transpose(1,2,0) and its yz view as img.transpose(2,0,1)
   segmentation3D, (probability3D, gradients3D) = uSegment3D.aggregate_2D_to_3D_segmentation_direct_method(probs=[img_segment_2D_xy_probs,
                                                                                                                           img_segment_2D_xz_probs,
                                                                                                                              img_segment_2D_yz_probs], 
                                                                                                                     gradients =   [img_segment_2D_xy_flows, 
                                                                                                                     img_segment_2D_xz_flows,
                                                                                                                     img_segment_2D_yz_flows], 
                                                                                                                     params=aggregation_params,
                                                                                                                  savefolder=None,
                                                                                                                  basename=None)

   # Save outputs as TIFF files
   segmentation_output_path = os.path.join(output_dir, 'segmentation3D.tif')
   probability_output_path = os.path.join(output_dir, 'probability3D.tif')
   gradients_output_path = os.path.join(output_dir, 'gradients3D.tif')

   tiff.imwrite(segmentation_output_path, segmentation3D.astype(np.uint16))  # Save segmentation as 16-bit TIFF
   tiff.imwrite(probability_output_path, probability3D.astype(np.float32))  # Save probability as 32-bit TIFF
   tiff.imwrite(gradients_output_path, gradients3D.astype(np.float32))  # Save gradients as 32-bit TIFF

   print(f"Segmentation saved to {segmentation_output_path}")
   print(f"Probability saved to {probability_output_path}")
   print(f"Gradients saved to {gradients_output_path}")