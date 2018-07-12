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
"""
	Device class storing resources counts for a given device.
	
	Device structure in JSON:

	{
  		"name": "XLNX:VU9P",
  		"type": "fpga"
		"part": "xcvu9p-flgb2104-2-i"
		"frequency": 250,
		"resources": {
    		"LUT": 2586000,
    		"DSP": 6840,
    		"BRAM": 75.9,
    		"URAM": 270
  		}
	}

"""

__author__="Ken O'Brien"
__email__="kennetho@xilinx.com"

import json
import sys
import logging
import config
import math

class Device:
	def __init__(self, filename, frequency=None):
		try:
			with open(config.DEVICE_DIR+filename) as dev_file:
				data = json.load(dev_file)
		except (OSError, IOError):
			print "Unable to open file %s or it is not correctly formatted" % filename
			sys.exit(-1)
	
		self.lut_proportion = config.LUT_PROPORTION
		self.bram_proportion = config.BRAM_PROPORTION
		self.name = data['name']
		self.type = data['type']
		self.part = data['part']
		resources = data['resources']
		self.luts = resources['LUT']
		self.dsps = resources['DSP']
		# In Xilinx datasheets, total BRAM is stated in units of Mebibits, 
		self.brams = math.ceil(self.toMebiByte(float(resources['BRAM'])) / (18.0*1024)) # Number of available BRAMS. Note: all BRAM counts in bits, not bytes.
		self.urams = resources['URAM']
		if frequency is None:
                    self.frequency = data['frequency']
                else:
                    self.frequency = frequency

	def toMebiByte(self, val):
		return val * (2**20)

	def __repr__(self):
		return 'Device Name: %s, Type: %s, luts: %d, bram: %d, uram: %d, dsp: %d\n' % (self.name, self.type, self.luts, self.brams, self.urams, self.dsps)

if __name__ == "__main__":
	d = Device(sys.argv[1])
	print d
