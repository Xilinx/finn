# Copyright (c) 2018, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#    3. Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Imports NN description from file"""

__author__="Ken O'Brien"
__email__="kennetho@xilinx.com"

import pandas
import logging

def load_from_file(filename, excelSheet=""):
	# TODO: Excel support for prototyping only. Implement support from prototxt.
	layers = []
	if filename is not None and 'xlsx' in filename:
		logging.info("Loading Excel file: %s" % filename)
		df = pandas.read_excel(open(filename, 'rb'), sheetname= excelSheet, skiprows=1)
		for index, row in df.iterrows():
			if "Pruning" in filename:	
				layers.append({ 'type': row['type'], 'in_dim': row['in dim'], 'in_channels': row['in channels'], 'out_channels': row['out channels'], 'out_dim': row['out dim'], 'ops': row['ops, total: 5164.118473M'], 'SIMD': 1, 'PE': 1, 'MMV': 1 , 'parallel': row['Parallel'], 'filter_dim': row['filter dim']})	
			elif "Dorefa" in filename:
				layers.append({ 'type': row['type'], 'in_dim': row['in dim'], 'in_channels': row['in channels'], 'out_channels': row['out channels'], 'out_dim': row['out dim'], 'ops': row['ops, total: 3874.173385M'], 'SIMD': 1, 'PE': 1, 'MMV': 1 , 'parallel': row['Parallel'], 'filter_dim': row['filter dim']})	
			else:
				layers.append({ 'type': row['type'], 'in_dim': row['in dim'], 'in_channels': row['in channels'], 'out_channels': row['out channels'], 'out_dim': row['out dim'], 'ops': row['ops, total: 2270.520598M'], 'SIMD': 1, 'PE': 1, 'MMV': 1 , 'filter_dim': row['filter dim'], 'parallel': 1})
	return layers
