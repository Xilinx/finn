#!/usr/bin/env python

"""Backend to find and execute external tools"""
__author__="Ken O'Brien"
__email__="kennetho@xilinx.com"

import FINN.core.config as config
import subprocess
import os
import math
import sys
import FINN.backend.fpga.layers_fpga as layers_fpga
import FINN.core.layers as layers_base
import FINN.core.nn as nn
from FINN.core.perf_model import PerfModel

class FPGABackendProduct:
	def  __init__(self, path, fpga_ir, dev):
		self.ir = fpga_ir
		self.path = path
		self.sim_layer = None
		self.hls_synth_results = None
		self.bitfile_synth_results = None
		self.dev = dev
		# TODO ExternalTools shouldn't try to find all tools at once
		# e.g. the user may not have xocc or vivado_hls installed.
		#self.runner = ExternalTools()

	def getSimCompileScriptPath(self):
		return self.path + "/simcompile.sh"

	def getSimExecutablePath(self):
		return self.path + "/sim"

	def getHLSSynthScriptPath(self):
		return self.path + "/hls_syn.tcl"

	def getBitfileSynthScriptPath(self):
		# TODO this should depend on the particular donut used...
		return self.path + "/make_pynq_bitfile.sh"

	def getSimLayer(self, force_recompile=False):
		if not os.path.exists(self.getSimExecutablePath()) or force_recompile:
			# run the rawhls sim compilation
			# first, make the script executable
			subprocess.check_call(["chmod", "+x", self.getSimCompileScriptPath()])
			# do the compilation
			subprocess.check_call(["sh", self.getSimCompileScriptPath()])
		ret = [layers_base.ExternalExecutionLayer(self.getSimExecutablePath())]
		# add interleaver and deinterleaver if i/o is multichannel
		if layers_fpga.isMultichannel(self.ir[0]):
			# TODO this assumes igroup = ifm -- may not be true in the future
			ichans = self.ir[0].getIGroup()
			idim = self.ir[0].getNumInputElems() / ichans
			idim = int(math.sqrt(idim))
			ret = [layers_base.ChanInterleaveLayer(idim, ichans)] + ret
		if layers_fpga.isMultichannel(self.ir[-1]):
			# TODO this assumes igroup = ifm -- may not be true in the future
			ochans = self.ir[-1].getOGroup()
			odim = self.ir[0].getNumOutputElems() / ochans
			odim = int(math.sqrt(odim))
			ret = ret + [layers_base.ChanDeinterleaveLayer(odim, ochans)]
		return ret

	def synthesis_HLS(self):
		# run the HLS synthesis script
		subprocess.check_call(["vivado_hls", "-f", self.getHLSSynthScriptPath()], cwd=self.path)

	def synthesis_bitfile(self):
		# run the bitfile synthesis script
		subprocess.check_call(["sh", self.getBitfileSynthScriptPath()], cwd=self.path)

	def getFPGAPerformanceModel(self):
		# return FPGA performance/cost model
		return PerfModel(nn.NN(layers=self.ir), self.dev)


class ExternalTools:

	def __init__(self):
		self.gxx = self.get_path("g++")
		self.hls = self.get_path("vivado_hls")
		self.xocc = self.get_path("xocc")

	def get_path(self, tool):
		"""Given the name of a commandline tool, return the fullpath to that executable
		or tell the user to add it to their path and exit."""
		paths = os.getenv('PATH').split(':')
		tool_path = None
		for path in paths:
			if os.path.isfile(path+"/"+tool):
				tool_path = path+"/"+tool
		if tool_path is None:
			print 'Error: Unable to locate '+tool+' in PATH.'
			sys.exit(1)
		return tool_path

	def execute(self, tool, args):
		if tool == "g++":
			print "Running g++"
			ret = subprocess.call([self.gxx, config.GXX_ARGS + args])
			print ret

if __name__=="__main__":
	e = ExternalTools()
	print e.gxx
	print e.hls
	print e.xocc
	e.execute("g++", "test.cpp")
