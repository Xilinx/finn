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


__author__ = "Ken O'Brien"
__email__ = "kennetho@xilinx.com"

import logging

def factors(target):
	"""Returns the factors of target"""
	_factors = []
	i = 1
	while i <= target :
		if target % i == 0:
			_factors.append(i)
		i+=1
	return _factors

def is_factor(candidate, target):
        if target % candidate == 0:
            return True
        return False


def next_factor(target, current):
	"""Given a target to factorise, find the next highest factor above current"""
        assert(target != 0 and current != 0)
	assert(current<=target) # current factor must be less than target
	if target==current or target==current+1:
		return target
	if current < 2: # 0 and 1 are not viable factors
		current = 2
	i = current+1 # Skip the current factor
	while i <= target :
		if target % i == 0:
			return i
		i+=1

def prev_factor(target, current):
	"""Given a target to factorise, find the next highest factor above current"""
	assert(current<=target)
	candidates = factors(target)
	if len(candidates) == 1:
		return 1
	logging.info("Selecting previous factor %d of %d given %d" % (candidates[candidates.index(current)-1], target, current))
	return candidates[candidates.index(current)-1]

if __name__ == "__main__":
	print "Factors of 3 ", factors(3)
	print "Factors of 12 ", factors(12)
	print "Factors of 17 ", factors(17)
	print "Next Factor of 12", next_factor(12,12)
	print "Prev of 12", prev_factor(12,6)
	print "Prev of 13", prev_factor(13,13)
	assert(next_factor(100, 40)==50)
