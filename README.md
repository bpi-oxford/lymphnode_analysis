# BPI_tissue

A respository where I have tried to concisely summarise the work on the lymph node tracking

Repository Structure:

BPI_tissue
    /src - the location of all the code
         - my current method for running the scripts is to run them from 'main.py' files found directly in this folder. They could probably be significanlty improved as notebooks or form the command line

         /cellpose_scripts - the location for the segmentation of the central region of the cells using cellpose. This contains two files - cellpose_batch.py for segmentation of whole videos and gamma_transformation.py for a super simple gamm filter. Note this folder does not contain the code for the secondary surface segmentation (/interface_detection) or a script for model training (I used the GUI thus far)

         /interface_detection - very prelimary code for the detection of interfaces through a secondary segmentation and following the bounding box of a cell.

         /scripts_to_remove - stuff I thought might be useful but is mainyl redundant and could have been confusing if left in

         /tracking_analysis - mainly very simple scripts to analyse the output tracks from ultrack and make some plots. Mostly take arguments of file paths or directroy that contain track_csv files. 

         /tracking_benchmarking - very simple scripts to check the track length of the ultrack tracks.csv files. Also a script for comparing against ground truth tracks.

         /ultrack_scripts - mostly scripts for visualisation of ultrack outputs. Mostly not used in the main workflow other than the flow_computation.py script which calcualte the flow field to aid with ultrack tracking.


    
        
          
