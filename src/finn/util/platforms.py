# Copyright (c) 2021, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from abc import abstractmethod

# contains the amount of available FPGA resources for several
# Xilinx platforms, as well as certain resource limit guidelines
# for creating designs that can achieve timing closure

# explicit value for res types/costs we don't care about
DONT_CARE = -1
# recommended resource limits from Xilinx for timing closure
# respectively for LUT, FF, BRAM_18K, URAM, DSP res types
DEFAULT_RES_LIMITS = np.array([0.7, 0.5, 0.80, 0.80, 0.80])
DEFAULT_AVG_CONSTRAINTS = [((2, 3, 4), 0.7)]  #

# resources required to instantiate certain infrastructure components
# such as memory controllers and network interfaces
DDR_RESOURCE_REQUIREMENTS = {
    "LUT": 33256,
    "FF": 44889,
    "BRAM_18K": 199,
    "URAM": 0,
    "DSP": 3,
}
HBM_RESOURCE_REQUIREMENTS = {
    "LUT": 10718,
    "FF": 21793,
    "BRAM_18K": 8,
    "URAM": 0,
    "DSP": 0,
}

# we assume use of VNx Alveo UDP stack
# see: https://gitenterprise.xilinx.com/mruiznog/vitis_network_layer
ETH_RESOURCE_REQUIREMENTS = {
    "LUT": 35219,
    "FF": 86269,
    "BRAM_18K": 183,
    "URAM": 0,
    "DSP": 0,
}


class Platform:
    def __init__(
        self,
        nslr=1,
        ndevices=1,
        sll_count=[],
        hbm_slr=-1,
        ddr_slr=[0],
        eth_slr=0,
        eth_gbps=0,
        limits=DEFAULT_RES_LIMITS,
        avg_constraints=DEFAULT_AVG_CONSTRAINTS,
    ):
        self.nslr = nslr
        self.sll_count = sll_count
        self.eth_slr = eth_slr
        self.eth_gbps = eth_gbps
        self.ndevices = ndevices
        self.hbm_slr = hbm_slr
        self.ddr_slr = ddr_slr
        # limits must be a np.array either of
        # the same shape as compute_resources
        # or broadcastable to it
        self.res_limits = limits
        # list of tuples of the form ( tuple of resource positions to avg, limit )
        self.avg_constraints = avg_constraints

    @property
    @abstractmethod
    def compute_resources(self):
        pass

    @property
    def guide_resources(self):
        guide = []
        # TODO: assert limits is of correct size
        guide_res = (
            np.tile(np.array(self.compute_resources), (self.ndevices, 1))
        ).astype(int)
        for i in range(self.nslr * self.ndevices):
            # when in multi-FPGA mode, subtract cost of UDP connection from eth_slr
            local_slr = i % self.nslr
            if self.ndevices > 1 and local_slr == self.eth_slr:
                guide_res[i][0] -= ETH_RESOURCE_REQUIREMENTS["LUT"]
                guide_res[i][1] -= ETH_RESOURCE_REQUIREMENTS["FF"]
                guide_res[i][2] -= ETH_RESOURCE_REQUIREMENTS["BRAM_18K"]
                guide_res[i][3] -= ETH_RESOURCE_REQUIREMENTS["URAM"]
                guide_res[i][4] -= ETH_RESOURCE_REQUIREMENTS["DSP"]
            # subtract the cost of memory controllers
            # if we have a choice between DDR and HBM, use HBM
            if local_slr == self.hbm_slr:
                guide_res[i][0] -= HBM_RESOURCE_REQUIREMENTS["LUT"]
                guide_res[i][1] -= HBM_RESOURCE_REQUIREMENTS["FF"]
                guide_res[i][2] -= HBM_RESOURCE_REQUIREMENTS["BRAM_18K"]
                guide_res[i][3] -= HBM_RESOURCE_REQUIREMENTS["URAM"]
                guide_res[i][4] -= HBM_RESOURCE_REQUIREMENTS["DSP"]
            elif local_slr in self.ddr_slr:
                guide_res[i][0] -= DDR_RESOURCE_REQUIREMENTS["LUT"]
                guide_res[i][1] -= DDR_RESOURCE_REQUIREMENTS["FF"]
                guide_res[i][2] -= DDR_RESOURCE_REQUIREMENTS["BRAM_18K"]
                guide_res[i][3] -= DDR_RESOURCE_REQUIREMENTS["URAM"]
                guide_res[i][4] -= DDR_RESOURCE_REQUIREMENTS["DSP"]
            guide.append(list(guide_res[i]))
        return guide

    @property
    def resource_count_dict(self):
        res = dict()
        for i in range(self.nslr * self.ndevices):
            slr_res = dict()
            slr_res["LUT"] = self.compute_resources[i % self.nslr][0]
            slr_res["FF"] = self.compute_resources[i % self.nslr][1]
            slr_res["BRAM_18K"] = self.compute_resources[i % self.nslr][2]
            slr_res["URAM"] = self.compute_resources[i % self.nslr][3]
            slr_res["DSP"] = self.compute_resources[i % self.nslr][4]
            res["slr" + str(i)] = slr_res
        return res

    @property
    def compute_connection_cost(self):
        x = np.full((self.nslr * self.ndevices, self.nslr * self.ndevices), DONT_CARE)
        # build connection cost matrix for one device's SLRs
        xlocal = np.full((self.nslr, self.nslr), DONT_CARE)
        for i in range(self.nslr):
            for j in range(self.nslr):
                if i == j:
                    xlocal[i][j] = 0
                elif abs(i - j) == 1:
                    xlocal[i][j] = 1
        # tile connection cost matrices for entire system
        for i in range(self.ndevices):
            x[
                i * self.nslr : (i + 1) * self.nslr, i * self.nslr : (i + 1) * self.nslr
            ] = xlocal
        # set cost for ethernet connections, assuming daisy-chaining
        for i in range(self.ndevices - 1):
            x[i * self.nslr + self.eth_slr][(i + 1) * self.nslr + self.eth_slr] = 10
            x[(i + 1) * self.nslr + self.eth_slr][i * self.nslr + self.eth_slr] = 10
        return x

    @property
    def compute_connection_resource(self):
        sll = np.full((self.nslr * self.ndevices, self.nslr * self.ndevices), 0)
        # build connection resource matrix for one device's SLRs
        slllocal = np.full((self.nslr, self.nslr), -1)
        for i in range(self.nslr):
            for j in range(self.nslr):
                if i == j:
                    # no SLL constraint when going from one SLR to itself
                    slllocal[i][j] = -1
                else:
                    slllocal[i][j] = self.sll_count[i][j]
        # tile connection cost matrices for entire system
        for i in range(self.ndevices):
            sll[
                i * self.nslr : (i + 1) * self.nslr, i * self.nslr : (i + 1) * self.nslr
            ] = slllocal
        # set cost for ethernet connections, assuming daisy-chaining
        eth = np.full((self.nslr * self.ndevices, self.nslr * self.ndevices), 0)
        # no Eth throughput constraints from one SLR to itself
        for i in range(self.ndevices * self.nslr):
            eth[i][i] = -1
        # apply symmetric ETH throughput constraints between the SLRs that have GTXes
        for i in range(self.ndevices - 1):
            eth[i * self.nslr + self.eth_slr][
                (i + 1) * self.nslr + self.eth_slr
            ] = self.eth_gbps * (10**9)
            eth[(i + 1) * self.nslr + self.eth_slr][
                i * self.nslr + self.eth_slr
            ] = self.eth_gbps * (10**9)
        # pack sll and eth info in one list-of-list-of-tuple structure
        constraints = []
        for i in range(self.ndevices * self.nslr):
            constraints_line = []
            for j in range(self.ndevices * self.nslr):
                # make sure not to constrain both resources at the same time
                # constrain for Eth throughput between SLRs on different devices
                # constrain for SLLs between SLRs on same device
                is_offchip = i // self.nslr != j // self.nslr
                constraints_line.append(
                    (-1 if is_offchip else sll[i][j], eth[i][j] if is_offchip else -1)
                )
            constraints.append(constraints_line)
        return constraints

    def map_device_to_slr(self, idx):
        """Given a global SLR index, return device id and local slr index"""
        assert idx <= self.nslr * self.ndevices
        return (idx % self.nslr, idx // self.nslr)


class Zynq7020_Platform(Platform):
    def __init__(
        self,
        ndevices=1,
        limits=DEFAULT_RES_LIMITS,
        avg_constraints=DEFAULT_AVG_CONSTRAINTS,
    ):
        super(Zynq7020_Platform, self).__init__(
            nslr=1,
            ndevices=ndevices,
            sll_count=[[0]],
            ddr_slr=[],
            eth_slr=0,
            eth_gbps=1,
            limits=limits,
            avg_constraints=avg_constraints,
        )

    @property
    def compute_resources(self):
        return [[53200, 2 * 53200, 280, 0, 220] for i in range(1)]


class ZU3EG_Platform(Platform):
    def __init__(
        self,
        ndevices=1,
        limits=DEFAULT_RES_LIMITS,
        avg_constraints=DEFAULT_AVG_CONSTRAINTS,
    ):
        super(ZU3EG_Platform, self).__init__(
            nslr=1,
            ndevices=ndevices,
            sll_count=[[0]],
            ddr_slr=[],
            eth_slr=0,
            eth_gbps=1,
            limits=limits,
            avg_constraints=avg_constraints,
        )

    @property
    def compute_resources(self):
        return [[71000, 2 * 71000, 412, 0, 360] for i in range(1)]


class ZU7EV_Platform(Platform):
    def __init__(
        self,
        ndevices=1,
        limits=DEFAULT_RES_LIMITS,
        avg_constraints=DEFAULT_AVG_CONSTRAINTS,
    ):
        super(ZU7EV_Platform, self).__init__(
            nslr=1,
            ndevices=ndevices,
            sll_count=[[0]],
            ddr_slr=[],
            eth_slr=0,
            eth_gbps=1,
            limits=limits,
            avg_constraints=avg_constraints,
        )

    @property
    def compute_resources(self):
        return [[230000, 2 * 230000, 610, 92, 1728] for i in range(1)]


class ZU9EG_Platform(Platform):
    def __init__(
        self,
        ndevices=1,
        limits=DEFAULT_RES_LIMITS,
        avg_constraints=DEFAULT_AVG_CONSTRAINTS,
    ):
        super(ZU9EG_Platform, self).__init__(
            nslr=1,
            ndevices=ndevices,
            sll_count=[[0]],
            ddr_slr=[],
            eth_slr=0,
            eth_gbps=1,
            limits=limits,
            avg_constraints=avg_constraints,
        )

    @property
    def compute_resources(self):
        return [[274000, 2 * 274000, 1824, 0, 2520] for i in range(1)]


class ZU28DR_Platform(Platform):
    def __init__(
        self,
        ndevices=1,
        limits=DEFAULT_RES_LIMITS,
        avg_constraints=DEFAULT_AVG_CONSTRAINTS,
    ):
        super(ZU28DR_Platform, self).__init__(
            nslr=1,
            ndevices=ndevices,
            sll_count=[[0]],
            ddr_slr=[],
            eth_slr=0,
            eth_gbps=1,
            limits=limits,
            avg_constraints=avg_constraints,
        )

    @property
    def compute_resources(self):
        return [[425000, 2 * 425000, 2160, 80, 4272] for i in range(1)]


class Alveo_NxU50_Platform(Platform):
    def __init__(
        self,
        ndevices=1,
        limits=DEFAULT_RES_LIMITS,
        avg_constraints=DEFAULT_AVG_CONSTRAINTS,
    ):
        # according to Vivado: 23040 SLR0 <-> SLR1
        sll_counts = [[0, 5000], [5000, 0]]
        super(Alveo_NxU50_Platform, self).__init__(
            nslr=2,
            ndevices=ndevices,
            sll_count=sll_counts,
            ddr_slr=[],
            hbm_slr=0,
            eth_slr=1,
            eth_gbps=100,
            limits=limits,
            avg_constraints=avg_constraints,
        )

    @property
    def compute_resources(self):
        # According to UG1120:
        # U50 has identical resource counts on both SLRs
        # return [[365000,2*365000,2*564, 304, 2580] for i in range(2)]
        # we observe from Vivado that the resource counts are actually:
        return [
            [374400, 2 * 374400, 2 * 564, 304, 2592],
            [368160, 2 * 368160, 2 * 564, 304, 2760],
        ]


class Alveo_NxU200_Platform(Platform):
    def __init__(
        self,
        ndevices=1,
        limits=DEFAULT_RES_LIMITS,
        avg_constraints=DEFAULT_AVG_CONSTRAINTS,
    ):
        sll_counts = [[0, 5000, 0], [5000, 0, 5000], [0, 5000, 0]]
        super(Alveo_NxU200_Platform, self).__init__(
            nslr=3,
            ndevices=ndevices,
            sll_count=sll_counts,
            ddr_slr=[0, 2],
            eth_slr=2,
            eth_gbps=100,
            limits=limits,
            avg_constraints=avg_constraints,
        )

    @property
    def compute_resources(self):
        # According to UG1120:
        # return [[355000, 723000, 2*638, 320, 2265],
        #        [160000, 331000, 2*326, 160, 1317],
        #        [355000, 723000, 2*638, 320, 2265]]
        # we observe from Vivado that the resource counts are actually:
        return [
            [385920, 2 * 385920, 2 * 714, 320, 2268],
            [199680, 2 * 199680, 2 * 420, 160, 1320],
            [385920, 2 * 385920, 2 * 714, 320, 2268],
        ]


class Alveo_NxU250_Platform(Platform):
    def __init__(
        self,
        ndevices=1,
        limits=DEFAULT_RES_LIMITS,
        avg_constraints=DEFAULT_AVG_CONSTRAINTS,
    ):
        sll_counts = [
            [0, 5000, 0, 0],
            [5000, 0, 5000, 0],
            [0, 5000, 0, 5000],
            [0, 0, 5000, 0],
        ]
        super(Alveo_NxU250_Platform, self).__init__(
            nslr=4,
            ndevices=ndevices,
            sll_count=sll_counts,
            ddr_slr=[0, 1, 2, 3],
            eth_slr=3,
            eth_gbps=100,
            limits=limits,
            avg_constraints=avg_constraints,
        )

    @property
    def compute_resources(self):
        # According to UG1120:
        # U250 has identical resource counts on all 4 SLRs:
        # return [[345000,2*345000,2*500, 320, 2877] for i in range(4)]
        # we observe from Vivado that the resource counts are actually:
        return [[375000, 2 * 375000, 2 * 576, 320, 2880] for i in range(4)]


class Alveo_NxU280_Platform(Platform):
    def __init__(
        self,
        ndevices=1,
        limits=DEFAULT_RES_LIMITS,
        avg_constraints=DEFAULT_AVG_CONSTRAINTS,
    ):
        sll_counts = [[0, 5000, 0], [5000, 0, 5000], [0, 5000, 0]]
        super(Alveo_NxU280_Platform, self).__init__(
            nslr=3,
            ndevices=ndevices,
            sll_count=sll_counts,
            ddr_slr=[0, 1],
            hbm_slr=0,
            eth_slr=2,
            eth_gbps=100,
            limits=limits,
            avg_constraints=avg_constraints,
        )

    @property
    def compute_resources(self):
        # according to UG1120
        # return [[369000, 746000, 2*507, 320, 2733],
        #        [333000, 675000, 2*468, 320, 2877],
        #        [367000, 729000, 2*512, 320, 2880]]
        # observed from Vivado:
        return [
            [400800, 2 * 400800, 2 * 600, 320, 2736],
            [382080, 2 * 382080, 2 * 576, 320, 2880],
            [380640, 2 * 380640, 2 * 576, 320, 2880],
        ]


# TODO: ADD KV260 to platform list
platforms = dict()
platforms["U50"] = Alveo_NxU50_Platform
platforms["U200"] = Alveo_NxU200_Platform
platforms["U250"] = Alveo_NxU250_Platform
platforms["U280"] = Alveo_NxU280_Platform
platforms["Pynq-Z1"] = Zynq7020_Platform
platforms["Pynq-Z2"] = Zynq7020_Platform
platforms["Ultra96"] = ZU3EG_Platform
platforms["ZCU104"] = ZU7EV_Platform
platforms["ZCU102"] = ZU9EG_Platform
platforms["ZCU111"] = ZU28DR_Platform
# platforms["kv260_som"] = # TODO kv260 platform... xck26_