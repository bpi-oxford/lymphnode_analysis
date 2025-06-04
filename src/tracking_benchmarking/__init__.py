import os
import sys

'''
A folder to carry out 'benchmarking' of the tracking


1. Detect lengths/positions of tracks to attempt to judge accuracy
2. Compare the effects of various ultrack post-processing parameters/functions
3. Compare the effects of various ultrack tracking/segmentation parameters


'''
# Add the parent directory to the system path to access adjacent modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
print(parent_dir)
sys.path.append(parent_dir)
print(sys.path)
