'''
A mini module to identify interfaces between cells in a tissue sample
Relies on work in adjacent libraries

Idea: take labels from initial cellpose segmentation and resegment 
using a further cellpose model + new methods?

Will need capabilities:
- to identify interesting interfaces from the previous ultrack results
- to then run a further model/analysis to find the wider segmentation of the cell boundary
- to then extract quantitative information about the interface

'''



