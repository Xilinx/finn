"""
How to write an analysis pass for FINN
--------------------------------------

An analysis pass traverses the graph structure and produces information about
certain properties. The convention is to take in a ModelWrapper, and return
a dictionary of named properties that the analysis extracts.
"""
