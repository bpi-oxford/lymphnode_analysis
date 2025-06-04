'''
function to search over ultrack parameters
'''

import pandas as pd
import numpy as np
from track_max_neighbour_testing import linking_param_search

if __name__ == "__main__":

    n_workers         = 4       
    max_distance      = None    
    max_neighbors     = None   
    appear_weight     = -0.001    
    disappear_weight  = -0.001
    division_weight   = -10
    image_border_size =(1,10,10)
    min_frontier      = 0.05

    max_distance_array  = np.linspace(10 , 50   , num = 10).astype(int)
    max_neighbors_array = np.linspace(5  , 35   , num = 10).astype(int)
    output_dir = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs'

    for i in range(len(max_distance_array)):
        for j in range(len(max_neighbors_array)):
            parameter_dict = {}
            parameter_dict['n_workers']        = n_workers
            parameter_dict['max_distance']     = max_distance_array[i]
            parameter_dict['max_neighbours']   = max_neighbors_array[j]
            parameter_dict['appear_weight']    = appear_weight
            parameter_dict['disappear_weight'] = disappear_weight
            parameter_dict['division_weight']  = disappear_weight
            parameter_dict['image_border_size']= image_border_size 
            parameter_dict['min_frontier']     = min_frontier
            linking_param_search(output_dir=output_dir , parameter_dict=parameter_dict)



    