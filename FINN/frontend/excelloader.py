#!/usr/bin/env python

__author__ = "Ken O'Brien"
__email__ = "kennetho@xilinx.com"

from loader import Loader
import os
import pandas
import logging

class ExcelLoader(Loader):
    """ Abstract base class defining what a loader must support """
    
    def __init__(self, filename, excelSheet):
        self.filename = filename
        self.excelSheet = excelSheet

    def load(self):
	# The excel files are non-standard
        logging.warning("Excel file format is non standard")
        layers = []
	if self.filename is not None and 'xlsx' in self.filename:
		logging.info("Loading Excel file: %s" % self.filename)
		df = pandas.read_excel(open(self.filename, 'rb'), sheetname= self.excelSheet, skiprows=1)
		for index, row in df.iterrows():
			if "Pruning" in self.filename:	
				layers.append({ 'type': row['type'], 'in_dim': row['in dim'], 'in_channels': row['in channels'], 'out_channels': row['out channels'], 'out_dim': row['out dim'], 'ops': row['ops, total: 5164.118473M'], 'SIMD': 1, 'PE': 1, 'MMV': 1 , 'parallel': row['Parallel'], 'filter_dim': row['filter dim']})	
			elif "Dorefa" in self.filename:
				layers.append({ 'type': row['type'], 'in_dim': row['in dim'], 'in_channels': row['in channels'], 'out_channels': row['out channels'], 'out_dim': row['out dim'], 'ops': row['ops, total: 3874.173385M'], 'SIMD': 1, 'PE': 1, 'MMV': 1 , 'parallel': row['Parallel'], 'filter_dim': row['filter dim']})	
			else:
				layers.append({ 'type': row['type'], 'in_dim': row['in dim'], 'in_channels': row['in channels'], 'out_channels': row['out channels'], 'out_dim': row['out dim'], 'ops': row['ops, total: 2270.520598M'], 'SIMD': 1, 'PE': 1, 'MMV': 1 , 'filter_dim': row['filter dim'], 'parallel': 1})
	return layers
          
