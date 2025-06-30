# Copyright (C) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

import copy
import numpy as np
import scipy
import warnings
from onnx import TensorProto, helper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor
from wrapdisc import Objective
from wrapdisc.var import GridVar
from qonnx.core.modelwrapper import ModelWrapper
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import part_map
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.platforms import DEFAULT_RES_LIMITS, platforms

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
)


def parameter_whitelist(padding_input):
    d = {}
    d["SIMD"] = {}
    d["PE"] = {}
    #d["ram_style"] = {}
    #d["resType"] = {}

    #d [ <parameter name> ] [ < op_type > ] [ padding amount, allow folding or not ]
    d["SIMD"]["DownSampler_hls"]=[padding_input,True,"NumChannels"]
    d["SIMD"]["FMPadding_hls"]=[padding_input,True,"NumChannels"]
    d["SIMD"]["FMPadding_rtl"]=[padding_input,True,"NumChannels"]
    d["SIMD"]["FMPadding_Pixel_hls"]=[padding_input,True,"NumChannels"]

    d["SIMD"]["ConvolutionInputGenerator_hls"]=[padding_input,False,"IFMChannels"]  
    d["SIMD"]["ConvolutionInputGenerator_rtl"]=[padding_input,False,"IFMChannels"]    

  #  d["ram_style"]["ConvolutionInputGenerator_hls"]=[0,True]    

    d["PE"]["AddStreams_hls"]=[padding_input,True,"NumChannels"]
    d["PE"]["ChannelwiseOp_hls"]=[padding_input,True,"NumChannels"]
  #  d["ram_style"]["ChannelwiseOp_hls"]=[0,True,None]
    d["PE"]["DuplicateStreams_hls"]=[padding_input,True,"NumChannels"]
    d["PE"]["GlobalAccPool_hls"]=[0,True,"NumChannels"]
    d["PE"]["Thresholding_hls"]=[padding_input,True,"NumChannels"]
    d["PE"]["Thresholding_rtl"]=[padding_input,True,"NumChannels"]
    d["PE"]["StreamingMaxPool_hls"]=[padding_input,True,"NumChannels"]
    d["PE"]["StreamingMaxPool_rtl"]=[padding_input,True,"NumChannels"]

    # Pool nodes are always optimized in tandem with a producer SWG
    d["PE"]["Pool_hls"]=[0,True,"Channels"]                           

    # only supported for rtl variant, need to add exceptions
    # so that only if every condition to create a dsp variant is met, 
    # to then allow folding this parameter
    d["SIMD"]["VVAU_hls"]=[0,False,"Kernel"] 

    d["PE"]["VVAU_hls"]=[padding_input,True,"Channels"]

    d["SIMD"]["VVAU_rtl"]=[0,True,"Kernel"]
    d["PE"]["VVAU_rtl"]=[padding_input,True,"Channels"]


   # d["resType"]["VVAU_hls"]=[0,True,None]
   # d["resType"]["VVAU_rtl"]=[0,True,None]

   # d["ram_style"]["VVAU_hls"]=[0,True,None]
  #  d["ram_style"]["VVAU_rtl"]=[0,True,None]

    d["SIMD"]["MVAU_hls"]=[padding_input,True,"MW"]
    d["PE"]["MVAU_hls"]=[padding_input,True,"MH"]

    d["SIMD"]["MVAU_rtl"]=[padding_input,True,"MW"]
    d["PE"]["MVAU_rtl"]=[padding_input,True,"MH"]
  #  d["ram_style"]["MVAU_rtl"]=[0,True,None,[3,2,1,0]]
   # d["ram_style"]["MVAU_hls"]=[0,True,None,[3,2,1,0]]
   # d["ram_style_thresholds"]["MVAU_rtl"]=[0,True,None,[2,1,0]]
   # d["ram_style_thresholds"]["MVAU_hls"]=[0,True,None,[2,1,0]]    
  #  d["resType"]["MVAU_rtl"]=[0,True,None,[1,0]]
   # d["resType"]["MVAU_hls"]=[0,True,None,[1,0]]

    # we do not fold LabelSelect due to it
    # potentially ruining fmax (TODO: heuristic for when
    # its safe to? Like certain topk to label ratio which
    # routes without issues? Or bring back once LabelSelect
    # has been improved / RTL variant added
    
    d["PE"]["LabelSelect_hls"]=[0,False,"Labels"]


    return d

def allowed_divisors(cap, bounding_value_exponent=1, max_padding_count=0, skip_folding = False):
    """
    compute all possible folding factors for a given
    upper bound variable

    max_padding_count allows generating values with the assumption
    that the bounding variable could be padded by up to that many
    elements, which dramatically increases the possible folding
    parameters with even a small amount of extra values

    bounding_value_exponent, if set to two, forces the folding factors into
    square roots of the bounding variable (applicable in some cases)
    """


    all_divs = []
    all_bounding_values = []
    factors = []

    if skip_folding:
        all_divs = [1]
        all_bounding_values = [cap]
    else:
        for i in range(cap, cap + max_padding_count + 1):
            for x in range(1, i + 1):
                if (i**bounding_value_exponent % x) == 0:
                    if (x not in all_divs) and (x <= cap) and (i//x not in factors):
                        all_divs.append(x)
                        all_bounding_values.append(i)
                        factors.append(i//x)


    return zip(*sorted(zip(all_divs, all_bounding_values)))


class Parameter:
    def __init__(
            self,
            name = None, # SWU_SIMD, MVAU_SIMD, MVAU_PE etc
            target_value_name = None,
            target_value = None,
            bound_name = None,
            bound_value = None,
            bound_value_last = None,
            update_threshold_input = False,
            update_weights_input = False,
            update_input_tensor_shape = False,
            update_output_tensor_shape = False,
            node = None,                    # node instance!
            node_index = None,
            op_type = None,
            model = None,
    ):
        self.name = name
        self.target_value_name = target_value_name
        self.target_value = target_value
        self.bound_name = bound_name
        self.bound_value = bound_value
        self.bound_value_last = bound_value_last
        self.update_threshold_input = update_threshold_input
        self.update_weights_input = update_weights_input
        self.update_input_tensor_shape = update_input_tensor_shape
        self.update_output_tensor_shape = update_output_tensor_shape
        self.node = node
        self.node_index = node_index
        self.op_type = op_type
        self.model = model


    def update_threshold_tensor(self):
        if self.op_type in ["Thresholding_hls","Thresholding_rtl"]:
            input_index = 1
            dim0 = self.node.get_nodeattr("NumChannels")

        elif self.op_type in ["VVAU_hls","VVAU_rtl"]:
            input_index = 2
            dim0 = self.node.get_nodeattr("Channels")
            if len(self.model.graph.node[self.node_index].input) < 3:
                # if the MVAU doesnt have a threshold input, just skip
                return
            
        elif self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            input_index = 2
            dim0 = self.node.get_nodeattr("MH")
            if len(self.model.graph.node[self.node_index].input) < 3:
                # if the MVAU doesnt have a threshold input, just skip
                return

        # thresholding nodes have a weight matrix which needs to be
        # adjusted if padding or cropping were introduced
        # MVAU and VVAU nodes can also have it so we stay flexible
            
        T = self.model.get_initializer(
            self.model.graph.node[self.node_index].input[input_index]
            )
        
        adt = self.model.get_tensor_datatype(
            self.model.graph.node[self.node_index].input[input_index]
        )
        T_new = gen_finn_dt_tensor(adt, (dim0, T.shape[1]))
        T_new[...] = 0

        T_new[: min(dim0, T.shape[0]), :] = T[: min(dim0, T.shape[0]), :]

        self.model.set_initializer(
            self.model.graph.node[self.node_index].input[input_index], T_new
        )

        self.model.set_tensor_shape(
            self.model.graph.node[self.node_index].input[input_index], T_new.shape
        )

        print(f"updated thresholding tensor for {self.node}")


    def update_weight_tensor(self):

        if self.op_type in ["VVAU_hls","VVAU_rtl"]:
            input_index = 1
            dim0 = self.node.get_nodeattr("Channels")
            dim1 = self.node.get_nodeattr("Kernel")

        elif self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            input_index = 1
            dim0 = self.node.get_nodeattr("MW")
            dim1 = self.node.get_nodeattr("MH")


        W = self.model.get_initializer(self.model.graph.node[self.node_index].input[input_index])
        
        if self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            if (dim0, dim1) == W.shape:
                return False
        
        if self.op_type in ["VVAU_hls","VVAU_rtl"]:
            if W.shape[0] == dim0 and W.shape[-2:] == tuple(dim1):
                return False

        wdt = self.model.get_tensor_datatype(
            self.model.graph.node[self.node_index].input[input_index]
        )

        if self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            W_new = gen_finn_dt_tensor(wdt, (dim0, dim1))
            W_new[...] = 0

            W_new[: min(dim0, W.shape[0]), : min(dim1, W.shape[1])] = W[
                : min(dim0, W.shape[0]), : min(dim1, W.shape[1])
            ]
            self.model.set_initializer(
                self.model.graph.node[self.node_index].input[1], W_new
            )

        if self.op_type in ["VVAU_hls","VVAU_rtl"]:
            W_new = gen_finn_dt_tensor(wdt, (dim0 ,W.shape[1],dim1[0],dim1[1]))
            W_new[...] = 0

            W_new[ : min(dim0, W.shape[0]) , : , \
                : min(dim1[0], W.shape[2]) , \
                : min(dim1[1], W.shape[3])] = W[:min(dim0, W.shape[0]) , : \
                ,: min(dim1[0], W.shape[2]) , : min(dim1[1], W.shape[3])]
            
            self.model.set_initializer(
                self.model.graph.node[self.node_index].input[input_index], W_new
            )

        self.model.set_tensor_shape(
            self.model.graph.node[self.node_index].input[1], W_new.shape
        )

        print(f"updated weight tensor for {self.node}")

        return True


    def apply_value(self,final=True):

        # update the target value being optimized
        #if self.target_value != self.last_value_last:
       # print(f"updating: {self.name}, {self.target_value_name}, {self.target_value} {final}")
        print(f"final: {final}")
        old_value = self.node.get_nodeattr(self.target_value_name)
        self.node.set_nodeattr(self.target_value_name,self.target_value)

        print(f"Updated folding value {self.target_value_name} of node {self.node} from {old_value} to {self.target_value}")
        # if the bounding value has changed (ie,. MW of an MVAU) as
        # a result of padding the node, update it as well
        #if self.bound_value != self.bound_value_last:
        if self.bound_name is not None:
            old_value = self.node.get_nodeattr(self.bound_name)
            print(f"Updated bounding value {self.bound_name} of node {self.node} from {old_value} to {self.bound_value}")
            self.node.set_nodeattr(self.bound_name,self.bound_value)
        #    self.bound_value_last = self.bound_value
        

        # make certain parallel window is set right
        if self.bound_name == "IFMChannels":
            if self.target_value < self.bound_value:
                self.node.set_nodeattr("parallel_window",0)

        # if this is the end of the minimizer routine, we update the tensor
        # shapes as well to retain functional correctness
        if final:
            op_type = self.op_type

            # first the io tensors only
            if self.update_input_tensor_shape:
                new_shape = self.node.get_normal_input_shape()
                self.model.set_tensor_shape(
                    self.model.graph.node[self.node_index].input[0], new_shape
                )                

            print(f"updated input tensor for {self.node} to {new_shape}")        

            if self.update_output_tensor_shape:
                new_shape = self.node.get_normal_output_shape()
                self.model.set_tensor_shape(
                    self.model.graph.node[self.node_index].output[0], new_shape
                )
            print(f"updated output tensor for {self.node} to {new_shape}")


            if self.update_threshold_input:
                self.update_threshold_tensor()

            if self.update_weights_input:
                self.update_weight_tensor()

class MetaParameter:
    """
    A meta parameter defines a single optimizable integer value (meta_value)
    which translates into a set of finn-onnx graph node attributes
    which are tighly linked together (called values)

    Examples: 
    -SIMD and PE values of a VVAU + SIMD of the SWU if necessary
    -SIMD value of an SWU and the PE and SIMD values of an MVAU (convolution)
    -SWU and Pool layer SIMD values (max pooling using SWU)

    - NOTE that MVAU PE and SIMD values are optimized independently, since
    - both 1-2 and 2-1 SIMD-PE combinations would have the same meta value
    - while having different resource characteristics

    All possible (legal) combinations of real values are stored in a list and an
    address translation is performed to map each meta_value to a set
    of real values when applying them
    """

    def __init__(
            self,
            name = None,
            meta_value = None,              # current value
            possible_values = [],      # all possible values
            real_values = [],               # list of real values for each possible value
            model = None,
            node_index = None,
    ):
        self.name = name
        self.meta_value = None
        assert len(real_values) == len(possible_values)
        self.possible_values = possible_values
        self.real_values = real_values
        self.model = model
        self.updated = False
        self.index = 0
        self.node_index = node_index
        
        """
        we build up a list of unique nodes related to this meta parameter
        for future cycle calculations
        """

        # sort the values first
        pairs = [(x,y) for (x,y) in sorted(zip(self.possible_values, self.real_values), key=lambda pair: pair[0])]
        self.possible_values = [x[0] for x in pairs]
        self.real_values = [x[1] for x in pairs]
        
        self.unique_nodes = []
        for val in real_values[0]:
            if val.node not in self.unique_nodes:
                self.unique_nodes.append(val.node)
           # self.name += f"{val.op_type}_{val.name}+"


    def update_value(self,value):
        #print(f"updating value from {self.meta_value} to {value}")
        if self.meta_value == value:
            self.updated = False
        else:
            self.meta_value = value
            self.updated = True

    def apply_value(self,final=False,filter=["PE","SIMD","parallel_window"]):
        # make sure to run this once before minimizing
        self.index = self.possible_values.index(self.meta_value)
        for val in self.real_values[self.index]:
            if val.target_value_name in filter:
                val.apply_value(final)

    def get_cycles(self):
        """
        This function assumes all parameters in the unique nodes are
        updated.
        """
        return max([n.get_exp_cycles() for n in self.unique_nodes])

    def print_value_table(self):
        s = f"++++++++++\n{self.name}\n"
        for i,value in enumerate(self.possible_values):
            s += f"\n{value}: "
            for real_value in self.real_values[i]:
                s += f"{real_value.name}: {real_value.target_value}, bound: {real_value.bound_value}|"
        s += "\n"
        print(s)

class ParameterSet:
    def __init__(self):
        self.parameters = []
        self.index_list = []
        self.nodes = []

    def filter(self, params_to_filter):
        # filter parameters we want to use in the set
        # useful for multi-pass optimization
        self.parameters = [x for x in self.parameters if x.name in params_to_filter]

    def get_max_cycles(self):
        cycles = [(n.get_exp_cycles(),n) for n in self.nodes]
        return max([n.get_exp_cycles() for n in self.nodes])

    def get_vals(self):
        return [p.value for p in self.parameters]

    def get_min_vals(self):
        # get minimum possible folding values in the set
        return [p.possible_values[0] for p in self.parameters]

    def get_max_vals(self):
        # get maximum possible folding values in the set
        return [p.possible_values[-1] for p in self.parameters]

    def add_all_params_to_index_list(self):
        self.index_list = [x for x in range(len(self.parameters))]

    def set_values(self, values):
        for i in range(len(self.index_list)):
            self.parameters[self.index_list[i]].update_value(values[i])

    def apply_updates(self, final=False, filter=[]):
        # a
        for i in self.index_list:
            self.parameters[i].apply_value(final, filter)

    def assign_involved_nodes(self):
        nodes = []
        for i in range(len(self.index_list)):
            p = self.parameters[self.index_list[i]]
            for node in p.unique_nodes:
                nodes.append(node)
        self.nodes = list(set(nodes))  # make this unique

class Optimizer:
    """
    Class responsible for the 'inner loop' of the folding optimization. We set all minimizer-specific
    Hyper-parameters here, model node & parameter partitioning, minimizer instantation, cost model function
    and the overarching loop of minimizing the partitions are performed in this class.
    """
    def __init__(
        self,
        model,
        name,
        targets,
        hard_constraint_target=             "max_cycles",
        target_cycles_per_frame=            1,
        padding=                            0,
        maxfun_per_parameter=               100,
        fpgapart=                           "xc7z020clg400-1",
        parameters_to_apply=                ["SIMD", "PE", "ram_style", "resType","ram_style_thresholds"],
        enable_folding_dwc_heuristic=       True,
        verbose=                            False,
        mvau_wwidth_max=                    1024,
        value_to_minimize_relaxation=       0.98,
        max_parameters_per_partition=       4,
        init_run=                           False,
        maxiter=                            200,
        accept = -                          0.5,
        pad_io_nodes =                      False,
        optimization_parameters =            ["SIMD", "PE", "ram_style", "resType","ram_style_thresholds"],
    ):
        self.params = None
        self.targets = targets
        self.updated_nodes = []
        self.param_indexes = []  # this might require insertion!!!
        self.param_ranges = []
        self.all_nodes = []
        self.target_cycles_per_frame = target_cycles_per_frame
        self.padding = padding
        self.mvau_wwidth_max = mvau_wwidth_max
        self.model = model
        self.pad_io_nodes = pad_io_nodes
        self.name = name
        self.fpgapart = fpgapart
        self.metrics = None
        self.init_run = init_run
        self.maxiter = maxiter
        self.accept = accept

        # 0-100, relax whether we MUST hit the required bounding value,
        # for example max_cycles
        self.value_to_minimize_relaxation = value_to_minimize_relaxation
        self.max_parameters_per_partition = max_parameters_per_partition
        self.maxfun_per_parameter = maxfun_per_parameter
        
        self.hard_constraint_target = hard_constraint_target
        self.parameters_to_apply = parameters_to_apply
        self.enable_folding_dwc_heuristic = enable_folding_dwc_heuristic
        self.verbose=verbose
        self.optimization_parameters = optimization_parameters

        # total number of nodes which got padded
        self.total_paddings = 0

    def cleanup_pass(self):
        # some corrections that may be necessary
        pass


    def compute_hls_dwc_cost(self, model, nodes, lut_capacity, hls_dwc_cost_penalty=8):

        # Given a set of nodes and a model,
        # consider the stream widths between all adjacent nodes
        # and apply a cost penalty if the shapes mismatch relative
        # to the cost of introducing a DataWidthConverter

        # this heuristic is critical for preventing overuse of 
        # DWCs with enormous resource costs

        # hls_dwc_cost_penalty is a rough heuristic for how much
        # an HLS variant consumes in LUTs

        cost = 0
        for node in nodes:
            prod = model.find_producer(node.onnx_node.input[0])

            # check if this is not the first node of a model
            if prod is not None:
                output_name = prod.output[0]
                prod_inst = getCustomOp(prod)
                inWidth = prod_inst.get_outstream_width()
                outWidth = prod_inst.get_instream_width()

                n0_out_shape = prod_inst.get_folded_output_shape()

                # mvau has a special case with external memory
                # where we have to consider a different input
                if (
                    node.onnx_node.op_type.startswith("MVAU")
                    and node.get_nodeattr("mem_mode") == "external"
                ) or (node.onnx_node.op_type.startswith("StreamingConcat")):
                    # get input idx
                    in_idx = None
                    for idx, n_input in enumerate(node.onnx_node.input):
                        if output_name == n_input:
                            in_idx = idx
                    assert in_idx is not None, "Malformed model"
                    n1_in_shape = node.get_folded_input_shape(in_idx)
                else:
                    # use default folded input shape
                    n1_in_shape = node.get_folded_input_shape()


                # dwcs cannot be inserted between mvau/vvau and pool/swg 
                # so we only run it for other combinations
                if (not ((prod.name.startswith("ConvolutionInputGenerator") or prod.name.startswith("Pool")) and 
                    (node.onnx_node.name.startswith("Pool") or node.onnx_node.name.startswith("MVAU") or node.onnx_node.name.startswith("VVAU")))): 
                    n1_in_shape = node.get_folded_input_shape()

                    # check if we need a DWC
                    if  np.prod(n0_out_shape) != np.prod(n1_in_shape) or n0_out_shape[-1] != n1_in_shape[-1]:
                        # HLS DWC needed, expensive
                        if (max(inWidth, outWidth) % min(inWidth, outWidth) != 0) or (np.prod(n0_out_shape) != np.prod(n1_in_shape)):
                            cost += ((inWidth + outWidth) * hls_dwc_cost_penalty) / lut_capacity

                        # RTL DWC can be used cheaply
                        else:
                            cost += (inWidth + outWidth) / lut_capacity

            # extra cost penalizing large widths
          #  cost += ((opt.params.nodes[0].get_instream_width() * 4) / opt.targets["LUT"])
           # cost += ((opt.params.nodes[-1].get_outstream_width() * 4) / opt.targets["LUT"])
        return cost

    def cost_model(self, param_guess, opt):
        """
        the function used for determining how
        'good' a given folding configuration is
        in respect to optimization targets
        any heuristics to consider as effects
        of folding on the effectiveness of the final
        model should go here
        """
        cost = 0

        # 1. apply the folding parameters
        opt.params.set_values(param_guess)
        opt.params.apply_updates(final=False, filter=self.parameters_to_apply)

        # 2. compute results
        cycles = opt.params.get_max_cycles()
        resources = self.get_resources(opt.params.nodes)
        metrics = {**{"max_cycles": cycles}, **resources}

        # 3. update cost based on all minimizable targets
        # the hard constraint (usually max_cycles) enforces
        # which target MUST be met.
        for value_to_minimize in opt.targets:
            if value_to_minimize != opt.hard_constraint_target:
                cost += metrics[value_to_minimize] / opt.targets[value_to_minimize]
            else:
                if metrics[value_to_minimize]*self.value_to_minimize_relaxation > (opt.targets[value_to_minimize]):
                    cost = np.inf     

        # 4. Add additional heuristic costs

        # 4.1 DWC heuristic to decrease the use of HLS DWCs
        # which can have massive LUT resource consumption
        # increases. All pairs are considered because
        # we optimize partitions left to right and consider
        # the DWC between a node and its left neighbor
        if self.enable_folding_dwc_heuristic:
            cost += self.compute_hls_dwc_cost(opt.model, opt.params.nodes, opt.targets["LUT"])

        return cost

    def execute_minimizer(self, discrete_args, init_guess):
        """
        the specific minimizer for performing the parameter optimization
        for a single parameter set is called with this function
        # argument bounds are applied using the wrap library
        """
        wrapped_objective = Objective(
            self.cost_model,
            variables=discrete_args,
        )
        bounds = wrapped_objective.bounds

        if len(bounds) == 0:
            return np.array(init_guess)
        
        encoded_init_guess = wrapped_objective.encode((init_guess))
        fixed_args = tuple([self])

        optimal_args = scipy.optimize.dual_annealing(
            func=wrapped_objective,
            x0=encoded_init_guess,
            maxiter=self.maxiter,
            accept=self.accept,
            visit = 2.0,
            maxfun=self.maxfun_per_parameter * len(init_guess),
            # niter=self.optimizer_ites,
            # stepsize=self.stepsize,
            # T=self.temp,
            args=(fixed_args),
            bounds=bounds,
        )

        optimized_params = optimal_args.x
        optimized_params = np.array(wrapped_objective.decode(optimized_params))
       # print("optimal params:", optimized_params)
        

        return optimized_params

    def optimize(
        self,
        partitions=2,
        initial_guess="max",
        max_nodes_in_partition=2,
        target_parameters=["SIMD", "PE"],
    ):
        """
        A single optimization pass across an entire model
        initial guess can be "min" or "max" for what folding values to use
        at the start of optimization
        min = least folding (makes sense when the hard constraint is resource use)
        max = maximum folding (makes sense when the hard constraint is max_cycles)
        It is critical to select these values in a way that lets the optimizer know
        a legal solution exists for the problem, otherwise it will give up after a set
        number of iterations

        we peform partition splitting in this function
        """

        print("STARTED OPTIMIZER WITH PARAMS:")
        print("enable_folding_dwc_heuristic: ",self.enable_folding_dwc_heuristic)
        print("padding: ",self.padding)
        print("effort: ",self.maxfun_per_parameter)


        # 1. Split parameters into partitions to optimize locally

        # calculate number of partitions if not set to 1
        param_count = len(self.params.parameters)
        if param_count > self.max_parameters_per_partition and partitions != 1:
            partitions = param_count // self.max_parameters_per_partition

        if partitions == 1:
            self.params.add_all_params_to_index_list()

        indexes = self.params.index_list = [x for x in range(len(self.params.parameters))]

        if initial_guess == "min":
            init_guess = self.params.get_min_vals()
        elif initial_guess == "max":
            init_guess = self.params.get_max_vals()
        self.params.set_values(init_guess)

        self.params.apply_updates(filter=target_parameters)
        self.params.assign_involved_nodes()
        params = self.params.parameters
        #assert True==False
        # node-based partitioning

        partitions = 0
        old_node_index = 0
        index_partitions = []
        init_guess_partitions = []
        params_partitions = []

        tmp_index_partitions = []
        tmp_init_guess_partitions = []
        tmp_params_partitions = []

        i = 0
        nodes_in_partition = 1
        for param in params:
            if param.name in target_parameters:
                new_node_index = param.node_index

                if new_node_index != old_node_index:
                    nodes_in_partition += 1

                if nodes_in_partition > max_nodes_in_partition:
                    # store set and start a new one
                    if len(tmp_index_partitions) > 0:
                        index_partitions.append(tmp_index_partitions)
                        init_guess_partitions.append(tmp_init_guess_partitions)
                        params_partitions.append(tmp_params_partitions)
                        tmp_index_partitions = []
                        tmp_init_guess_partitions = []
                        tmp_params_partitions = []
                        partitions += 1
                        nodes_in_partition = 1
                if nodes_in_partition <= max_nodes_in_partition:
                    tmp_index_partitions.append(indexes[i])
                    tmp_init_guess_partitions.append(init_guess[i])
                    tmp_params_partitions.append(params[i])

                old_node_index = new_node_index
            i += 1

        # add remaining lefover tail partition
        if len(tmp_index_partitions) > 0:
            if len(tmp_index_partitions) > 0:
                index_partitions.append(tmp_index_partitions)
                init_guess_partitions.append(tmp_init_guess_partitions)
                params_partitions.append(tmp_params_partitions)
                partitions += 1


        # 2. Perform local optimization of partitions
        for p in range(partitions):

            
            # generate discrete argument list based on possible values
            # this is the input for the scipy minimizer
            discrete_args = []
            for arg in params_partitions[p]:
                discrete_args.append(GridVar(tuple(arg.possible_values)))

            # filter out parameters to the ones of the requested partition
            self.params.index_list = index_partitions[p]
            self.params.assign_involved_nodes()

            # fetch the respective initial list of parameters
            # it is very important that the initial guess is feasible
            # for the minimizer so that the cost_model call returns a non-infinity cost
            # otherwise the optimizer might give up believing there is no solution
            init_guess = init_guess_partitions[p]
            
            # an initial run to get resource consumption bounds
            if self.init_run:
                optimized_params = init_guess
            else:
                print("PARAMS IN PARTITION:")
                for param in params_partitions[p]:
                    print(param.name, param.unique_nodes)
                optimized_params = self.execute_minimizer(discrete_args, init_guess)

            # apply final values, adjusting the model accordingly
            self.params.set_values(optimized_params)
            self.params.apply_updates(final=True, filter=target_parameters)

        # final surgery of the model
        self.cleanup_pass()

        print(f"final parameters, init_run={self.init_run}:")

        s = ""
        total_params = 0
        total_padding = 0

        self.padding_result = f"{total_padding} / {total_params}"
        print("optimizer padding ratio: ",self.padding_result)
        #print("optimized param values: ",self.params.get_vals())
        for p in self.params.parameters:
            self.total_paddings += total_padding

    def get_resources(self, nodes):
        resources = {}
        for n in nodes:
            resources[n] = n.node_res_estimation(self.fpgapart)
        return aggregate_dict_keys(resources)



    def generate_parameter_set(self):

        # given a model, extract all optimizable parameters from it
        # as well as the possible values on these parameters
        # and the respective bounding parameter which might need to
        # be adjusted in case of padding

        model = self.model

        whitelist = parameter_whitelist(self.padding)

        graph = model.graph
        pset = []
        node_indx = 0
        arg_indx = 0
        node_count = len(graph.node)
        skips = 0
        

        for node_indx in range(0,node_count):
            print(f"NODE INDX: {node_indx}")
            if skips > 0:
                skips-=1
                continue
            node =  graph.node[node_indx]

            maximum_padding = self.padding

            if node is None:
                continue
            
            if node.op_type == "StreamingDataWidthConverter":
                continue

            if not (is_hls_node(node) or is_rtl_node(node)):
                continue

            # restrict padding if applicable
            if self.pad_io_nodes is not True:
                if node_indx == 0 or node_indx == len(graph.node)-1:
                    # do not allow padding IO nodes
                    maximum_padding = 0
                

            if node.op_type in ["ConvolutionInputGenerator_hls","ConvolutionInputGenerator_rtl"]:
                # a convolution input generator is always followed by a node which is tied to it
                # we have to handle these cases with a larger meta-parameter(s)

                node_inst = getCustomOp(node)
                second_node = graph.node[node_indx+1]
                second_node_inst = getCustomOp(second_node)
                # SWU should only be consumed by pool, vvau or mvau nodes
                assert second_node.op_type in ["Pool_rtl","Pool_hls","MVAU_hls","MVAU_rtl","VVAU_hls","VVAU_rtl"]


                # start extracting the SWU parameters we will need
                bound_swu = node_inst.get_nodeattr("IFMChannels")
                kernel_size = np.prod(node_inst.get_nodeattr("ConvKernelDim"))  
                depthwise = node_inst.get_nodeattr("depthwise")
                M = node_inst.get_nodeattr("M")
                parallel_window = node_inst.get_nodeattr("parallel_window")

                padding_internal_swu = np.min([whitelist["SIMD"][node.op_type][0],maximum_padding])

                possible_values_swu, bounding_values_swu = allowed_divisors(bound_swu, 1, padding_internal_swu)  

                
                # reset simd to
                #node_inst.set_nodeattr("SIMD", 1)

                if second_node.op_type in ["Pool_rtl","Pool_hls"]:
                    # SWU->Pool pair, we are optimizing SWU_SIMD, POOL_PE
                    # SIMD values identical, SWU_IFMChannels == Pool_ChannelNum when padding
                    assert node_inst.get_nodeattr("depthwise") == 1
                    # a single meta value (1 -> max SIMD)
                    values_simd = []
                    for i,ifmchannels_new in enumerate(list(bounding_values_swu)): 
                        simd = possible_values_swu[i]
                        value_swu_simd = Parameter(
                            name = "SWU_SIMD",
                            target_value_name = "SIMD",
                            target_value = simd,
                            bound_name = "IFMChannels",
                            bound_value = ifmchannels_new,
                            update_threshold_input=False,
                            update_weights_input=False,
                            update_input_tensor_shape=True,
                            update_output_tensor_shape=True,
                            node=node_inst,
                            node_index=node_indx,
                            op_type=node.op_type,
                            model=self.model,
                            )

                        value_pool_pe = Parameter(
                            name = "Pool_PE",
                            target_value_name = "PE",
                            target_value = simd,
                            bound_name = "Channels",
                            bound_value = ifmchannels_new,
                            update_threshold_input=False,
                            update_weights_input=False,
                            update_input_tensor_shape=True,
                            update_output_tensor_shape=True,
                            node=second_node_inst,
                            node_index=node_indx+1,
                            op_type=second_node.op_type,
                            model=self.model,
                            )
                        
                        values_simd.append([value_swu_simd,value_pool_pe])


                    # construct meta parameter
                        
                    meta_parameter_simd = MetaParameter(
                        name="SIMD",
                        meta_value = possible_values_swu[0],
                        possible_values = possible_values_swu,
                        real_values = values_simd,
                        model=self.model,
                        node_index = node_indx,
                    ) 
                    pset.append(meta_parameter_simd)     
                    meta_parameter_simd.print_value_table()
                    skips = 1
                    
                    pass
                elif second_node.op_type in ["MVAU_hls","MVAU_rtl"]:
                    # SWU->MVAU pair, we are optimizing SWU_SIMD, MVAU_SIMD, MVAU_PE
                    # MVAU_SIMD is linked to SWU_SIMD, parallel window to push MVAU_SIMD > SWU_SIMD
                    # two meta values (1 -> max SIMD and 1 -> max PE)
                    skips = 1
                    second_node_inst = getCustomOp(second_node)
                    # simd
                    values_simd = []
                    all_possible_values = []
                    factors = []

                    ww = second_node_inst.get_weight_datatype().bitwidth()
                   # assert True==False
                    # limit to unique bounds
                    bounding_values_swu = list(set(bounding_values_swu))
                    for i,ifmchannels_new in enumerate(bounding_values_swu): 
                        mw_new = kernel_size * ifmchannels_new
                        possible_values_mvau_simd, bounding_values_mw = allowed_divisors(mw_new, 1, 0)
                         
                        #all_possible_values.append(*possible_values_mvau_simd)
                        for i,simd in enumerate(possible_values_mvau_simd):
                            bound_value_mw = bounding_values_mw[i]

                            print(f"tried: ifmch, mw and simd: {ifmchannels_new}, {mw_new}, {simd}")
                            if ifmchannels_new % simd == 0 and  simd not in all_possible_values and bound_value_mw//simd not in factors and simd <= ifmchannels_new and (ww * simd) < self.mvau_wwidth_max and simd > (bound_value_mw/1024):
                                print("pass")
                                all_possible_values.append(simd)
                                factors.append(bound_value_mw//simd)
                                if simd < ifmchannels_new:
                                    simd_swu = simd
                                    parallel_window = 0
                                else:
                                    simd_swu = ifmchannels_new
                                    parallel_window = 1


                                value_swu_parallel_window = Parameter(
                                    name = "SWU_parallel_window",
                                    target_value_name = "parallel_window",
                                    target_value = parallel_window,
                                    bound_name = None,
                                    bound_value = None,
                                    update_threshold_input=False,
                                    update_weights_input=False,
                                    update_input_tensor_shape=True,
                                    update_output_tensor_shape=True,
                                    node=node_inst,
                                    node_index=node_indx,
                                    op_type=node.op_type,
                                    model=self.model,
                                    )     

                                value_swu_simd = Parameter(
                                    name = "SWU_SIMD",
                                    target_value_name = "SIMD",
                                    target_value = simd_swu,
                                    bound_name = "IFMChannels",
                                    bound_value = ifmchannels_new,
                                    update_threshold_input=False,
                                    update_weights_input=False,
                                    update_input_tensor_shape=True,
                                    update_output_tensor_shape=True,
                                    node=node_inst,
                                    node_index=node_indx,
                                    op_type=node.op_type,
                                    model=self.model,
                                    )
                                
                    
                                value_mvau_simd = Parameter(
                                    name = "MVAU_SIMD",
                                    target_value_name = "SIMD",
                                    target_value = simd,
                                    bound_name = "MW",
                                    bound_value = bound_value_mw,
                                    update_threshold_input=True,
                                    update_weights_input=True,
                                    update_input_tensor_shape=True,
                                    update_output_tensor_shape=True,
                                    node=second_node_inst,
                                    node_index=node_indx+1,
                                    op_type=second_node.op_type,
                                    model=self.model,
                                    )   

                                

                                values_simd.append([value_swu_simd,
                                                    value_swu_parallel_window,
                                                    value_mvau_simd])
                    #assert True == False
                    meta_parameter_simd = MetaParameter(
                        name="SIMD",
                        meta_value = possible_values_mvau_simd[0],
                        possible_values = all_possible_values,
                        real_values = values_simd,
                        model=self.model,
                        node_index = node_indx,
                    )

                    
                    meta_parameter_simd.print_value_table()
                    pset.append(meta_parameter_simd)
                    # values_simd
                    values_pe = []
                    factors = []
                    # pe
                    mh = second_node_inst.get_nodeattr("MH")
                    #padding_internal_mvau_mh = np.min([whitelist["PE"][second_node.op_type][0],maximum_padding])
                    padding_internal_mvau_mh = 0 # do not allow independent padding of an mvau used as a conv layer

                    possible_values_mvau_pe, bounding_values_mh = allowed_divisors(mh, 1, padding_internal_mvau_mh)
                    for i,value_mvau_pe in enumerate(possible_values_mvau_pe):
                        bound_mvau_mh = bounding_values_mh[i]

                        factor = bound_mvau_mh // value_mvau_pe
                        if factor not in factors:
                            factors.append(factor)
                            value_mvau_pe = Parameter(
                                name = "MVAU_PE",
                                target_value_name = "PE",
                                target_value = value_mvau_pe,
                                bound_name = "MH",
                                bound_value = bound_mvau_mh,
                                update_threshold_input=True,
                                update_weights_input=True,
                                update_input_tensor_shape=True,
                                update_output_tensor_shape=True,
                                node=second_node_inst,
                                node_index=node_indx+1,
                                op_type=second_node.op_type,
                                model=self.model,
                                )   
                            values_pe.append([value_mvau_pe])

                    meta_parameter_pe = MetaParameter(
                        name = "PE",
                        meta_value = possible_values_mvau_pe[0],
                        possible_values = possible_values_mvau_pe,
                        real_values = values_pe,
                        model=self.model,
                        node_index = node_indx,
                    )      
                    pset.append(meta_parameter_pe)               


                    pass
                elif second_node.op_type in ["VVAU_hls","VVAU_rtl"]:
                    # SWU->VVAU pair, we are optimizing SWU_SIMD, VVAU_SIMD, VVAU_PE
                    # VVAU_PE is linked to SWU_SIMD, VVAU_SIMD relies on SWU_parallel_window

                    # one meta value (1 -> (max SIMD *  max PE))
                    skips = 1
                    # make sure the SWU is depth-wise
                    assert node_inst.get_nodeattr("depthwise") == 1

                    # SIMD of the VVAU and SWU depend on the IFM channel count
                  
                    # for possible simd values[-1]:
                    # add with pe=1
                    #vvau_simd = possible_simd_values[-1]
                    #for vvau_pe in possible_pe_values:
                    #    append([swu_simd,vvau_simd,vvau_pe])
                    # swu_simd == vvau_pe!!!

                    second_node_inst = getCustomOp(second_node)
                    # simd
                    values_swu_vvau = []
                    all_possible_values = []
                    factors_pe = []

                    ww = second_node_inst.get_weight_datatype().bitwidth()
                   # assert True==False
                    # limit to unique bounds
                    bounding_values_swu = list(set(bounding_values_swu))
                    for i,ifmchannels_new in enumerate(bounding_values_swu): 

                        (kernel_dim0,kernel_dim1) = node_inst.get_nodeattr("ConvKernelDim")

                        # we cant pad the simd of the VVAU, since this is a kernel size
                        possible_values_vvau_pe, bounding_values_pe = allowed_divisors(ifmchannels_new, 1, 0)
                         
                        #all_possible_values.append(*possible_values_mvau_simd)
                        for i,pe in enumerate(possible_values_vvau_pe):
                            pe_bound = bounding_values_pe[i]
                            factor = bounding_values_pe[i]// pe
                            if factor not in factors_pe:
                                factors_pe.append(factor)
                                bound_value_kernel = (kernel_dim0,kernel_dim1)

                                if pe < ifmchannels_new:
                                    simd_limit = 1
                                    parallel_window = 0
                                else:
                                    simd_limit = np.prod(bound_value_kernel)
                                    parallel_window = 1

                                possible_values_mvau_simd, bounding_values_mw = allowed_divisors(simd_limit, 1, 0)
                                factors_simd = []
                                for i,simd in enumerate(possible_values_mvau_simd):
                                    if simd*pe not in all_possible_values and np.prod(bound_value_kernel)//simd not in factors_simd:
                                        print("pass")

                                        all_possible_values.append(simd*pe)

                                        factors_simd.append(np.prod(bound_value_kernel)//simd)



                                        value_swu_parallel_window = Parameter(
                                            name = "SWU_parallel_window",
                                            target_value_name = "parallel_window",
                                            target_value = parallel_window,
                                            bound_name = None,
                                            bound_value = None,
                                            update_threshold_input=False,
                                            update_weights_input=False,
                                            update_input_tensor_shape=True,
                                            update_output_tensor_shape=True,
                                            node=node_inst,
                                            node_index=node_indx,
                                            op_type=node.op_type,
                                            model=self.model,
                                            )     

                                        value_swu_simd = Parameter(
                                            name = "SWU_SIMD",
                                            target_value_name = "SIMD",
                                            target_value = pe,
                                            bound_name = "IFMChannels",
                                            bound_value = ifmchannels_new,
                                            update_threshold_input=False,
                                            update_weights_input=False,
                                            update_input_tensor_shape=True,
                                            update_output_tensor_shape=True,
                                            node=node_inst,
                                            node_index=node_indx,
                                            op_type=node.op_type,
                                            model=self.model,
                                            )
                                        
                            
                                        value_vvau_simd = Parameter(
                                            name = "VVAU_SIMD",
                                            target_value_name = "SIMD",
                                            target_value = simd,
                                            bound_name = "Kernel",
                                            bound_value = (kernel_dim0,kernel_dim1),
                                            update_threshold_input=True,
                                            update_weights_input=True,
                                            update_input_tensor_shape=True,
                                            update_output_tensor_shape=True,
                                            node=second_node_inst,
                                            node_index=node_indx+1,
                                            op_type=second_node.op_type,
                                            model=self.model,
                                            )   

                                        value_vvau_pe = Parameter(
                                            name = "VVAU_PE",
                                            target_value_name = "PE",
                                            target_value = pe,
                                            bound_name = "Channels",
                                            bound_value = pe_bound,
                                            update_threshold_input=True,
                                            update_weights_input=True,
                                            update_input_tensor_shape=True,
                                            update_output_tensor_shape=True,
                                            node=second_node_inst,
                                            node_index=node_indx+1,
                                            op_type=second_node.op_type,
                                            model=self.model,
                                            )   


                                        values_swu_vvau.append([value_swu_simd,
                                                            value_swu_parallel_window,
                                                            value_vvau_simd,
                                                            value_vvau_pe])


                    #assert True == False
                    meta_parameter_swu_vvau = MetaParameter(
                        name="SIMD",
                        meta_value = all_possible_values[0],
                        possible_values = all_possible_values,
                        real_values = values_swu_vvau,
                        model=self.model,
                        node_index = node_indx,
                    )

                    
                    meta_parameter_simd.print_value_table()
                    pset.append(meta_parameter_swu_vvau)
                    # values_simd
                    #assert True==False
            else:
                # simple singular node with one parameter
                skips = 0
                # one meta value (either 1 -> max SIMD or 1 -> max PE)
                # or none at all, then we skip
                op_type = node.op_type

                node_inst = getCustomOp(node)
                for p in self.optimization_parameters:
                    if p in whitelist:
                        if op_type in whitelist[p]:
                            # p is our parameter
                            (padding_internal,fold,bounding_parameter_name) = whitelist[p][op_type]
                            padding_internal = min(padding_internal,maximum_padding)
                           # if fold:
                            # generate list


                            factors = []
                            possible_values_final = []

                            bound = node_inst.get_nodeattr(bounding_parameter_name)
                            possible_values, bounding_values = allowed_divisors(bound, 1, padding_internal, skip_folding=True)
                            possible_values = list(possible_values)

                            values = []

                            if op_type in ["Thresholding_rtl","Thresholding_hls","MVAU_hls","MVAU_rtl","VVAU_hls","VVAU_rtl"]:
                                update_threshold_input = True
                            else:
                                update_threshold_input = False

                            if op_type in ["MVAU_hls","MVAU_rtl","VVAU_hls","VVAU_rtl"]:
                                update_weights_input = True
                            else:
                                update_weights_input = False
                                
                            
                            for i,value in enumerate(possible_values):
                                bounding_value = bounding_values[i]
                                #if op_type in ["MVAU_hls","MVAU_rtl"] and p == "SIMD":
                                    #ww = node_inst.get_weight_datatype().bitwidth()
                                    #if not ((ww * value) < self.mvau_wwidth_max and value > (bounding_value/1024)):
                                    #    continue


                                if op_type in ["VVAU_rtl","VVAU_hls" and p == "SIMD"]:
                                    bounding_value = node_inst.get_nodeattr("Kernel") # dont mess with kernel sizes

                                factor = np.prod(bounding_value)//value
    
                                if factor not in factors:
                                    factors.append(factor)
                                    possible_values_final.append(value)
                                    value_obj = Parameter(
                                        name = f"{op_type}_{p}",
                                        target_value_name = p,
                                        target_value = value,
                                        bound_name = bounding_parameter_name,
                                        bound_value = bounding_value,
                                        update_threshold_input=update_threshold_input,
                                        update_weights_input=update_weights_input,
                                        update_input_tensor_shape=True,
                                        update_output_tensor_shape=True,
                                        node=node_inst,
                                        node_index=node_indx,
                                        op_type=node.op_type,
                                        model=self.model,
                                        )   
                                    values.append([value_obj])

                            # construct meta parameter    
                                                                
                            meta_parameter = MetaParameter(
                                name = p,
                                meta_value = possible_values_final[0],
                                possible_values = possible_values_final,
                                real_values = values,
                                model=self.model,
                                node_index = node_indx,
                            )      
                            pset.append(meta_parameter)   
                            meta_parameter.print_value_table()
                            # if op_type == "MVAU_rtl":
                            #     assert True==False

            # we skip nodes in case of tighly coupled nodes like swu->mvau

            op_type = node.op_type
            node_inst = getCustomOp(node)

            # start checking

            self.all_nodes.append(node_inst)

        pset_obj = ParameterSet()
        pset_obj.parameters = pset
        self.params = pset_obj

def insert_and_size_fifos(model_dir,model,board,fpga_part,consider_dwc_costs, auto_fifo_strategy):
    if not consider_dwc_costs:
        model = model.transform(InsertDWC())

    import finn.builder.build_dataflow as build
    import finn.builder.build_dataflow_config as build_cfg
    from finn.builder.build_dataflow_steps import step_set_fifo_depths

    cfg = DataflowBuildConfig(
        output_dir = "",
        auto_fifo_depths = True,
        split_large_fifos = True,
        auto_fifo_strategy = auto_fifo_strategy,
        folding_config_file = None,
        synth_clk_period_ns=5.0,
        fpga_part = fpga_part,
        steps=["step_set_fifo_depths"],
        generate_outputs = [],
        board=board,
        extract_hw_config=False,
    )
    #build.build_dataflow_cfg(model_dir, cfg)
    #model_dir_new = "inter_folding_output/step_set_fifo_depths.onnx"
    #ModelWrapper(model_dir_new)

    model = step_set_fifo_depths(model,cfg)

    return model

class SetFolding(Transformation):

    """
    Attempt to set parallelism attributes in all nodes to meet a specific
    target expressed as cycles per frame target_cycles_per_frame. For each
    HLSCustomOp node type, the attribute may vary but is typically one of {PE, SIMD},
    and has a certain allowed-maximum value and divisibility constraints,
    which SetFolding will take into account.

    If style is set to 'optimizer', an optimization algorithm based on a target function
    and an optimization objective is employed.
     
    If padding is set to more than 0, folding factor restrictions are
    drastically relaxed by adding padding to all relevant nodes if this helps
    achieve the optimal folding. Special padding & cropping DWCs are also inserted where
    necessary.

    In the returned model, each node's
    cycles_estimate attribute will be set to its estimated number of cycles.

    """

    def __init__(
        self,
        target_cycles_per_frame=1000,
        mvau_wwidth_max=1024,
        two_pass_relaxation=True,
        style="optimizer",
        folding_maximum_padding=0,
        folding_max_attempts=1,
        platform="Pynq-Z1",
        folding_effort=250,
        enable_folding_dwc_heuristic=1,
        enable_folding_fifo_heuristic=1,
        folding_pad_io_nodes=False,
        devices=1,
        verbose=False,
        max_parameters_per_partition=4,

        # the strategy should ideally be analytic fifo sizing only.
        # RTLSIM-based sizing would make the folding time
        # quickly explode
        auto_fifo_strategy="analytic",
    ):
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
        self.two_pass_relaxation = two_pass_relaxation
        self.max_attempts = folding_max_attempts
        self.padding = folding_maximum_padding
        self.devices = devices
        self.platform = platform
        self.fpgapart = part_map[self.platform]
        self.verbose=verbose
        self.pad_io_nodes = folding_pad_io_nodes
        # either "naive" or "optimizer"
        self.style = style

        # maximum function calls / parameter
        # recommended in the range of 50-200 depending on the network size
        # and how long the user is willing to wait for this step
        # ~20 parameters with <30 possible values per parameter @ 200 effort = <30s
        self.effort = folding_effort

        # self.optimization_parameters = ["SIMD","PE"]
        self.optimization_parameters = ["SIMD", "PE", "parallel_window", "ram_style", "resType","ram_style_thresholds"]
        self.hard_constraint_target = "max_cycles"
        self.optimize_folding = True
        self.optimize_resource_types = False
        self.insert_dwcs = False
        self.consider_dwc_costs = True

        self.max_parameters_per_partition = max_parameters_per_partition

        # WARNING: if set to true, this flag
        # can result in an enormous increase in 
        # the time it takes to run this transformation
        # relative to the time it takes to run
        # set_fifo_depths times (folding_max_attempts-1)
        # Recommended to only run if analytic FIFO sizing
        # is also enabled (experimental feature)
        self.enable_folding_fifo_heuristic = enable_folding_fifo_heuristic

        self.auto_fifo_strategy = auto_fifo_strategy
        self.enable_folding_dwc_heuristic = enable_folding_dwc_heuristic
        
        self.target_resources = ["LUT","BRAM_18K","DSP","URAM"]



    def apply_optimized_folding(self, model):
        """
        Optimization algorithm-based folding transformation 
        using an iterative optimization algorithm and a target function
        to find optimal folding values for each node in the FINN graph,
        by default minimizing resource consumption while making sure to meet
        the target max_cycles (throughput) rate
        """

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        targets = {}
        targets["max_cycles"] = self.target_cycles_per_frame
        current_throughput_target = self.target_cycles_per_frame

        # fetch all parameters and bounds from the model by
        # running the optimizer once without minimizing cost

        init_model = copy.deepcopy(model)
        opt1 = Optimizer(
            init_model,
            "defaultOPT_for_parameter_extraction",
            targets,
            self.hard_constraint_target,
            target_cycles_per_frame=self.target_cycles_per_frame,
            padding=0,
            fpgapart=self.fpgapart,
            maxfun_per_parameter=self.effort,
            parameters_to_apply=["SIMD", "PE"],
            enable_folding_dwc_heuristic=self.enable_folding_dwc_heuristic,
            verbose=self.verbose,
            mvau_wwidth_max=self.mvau_wwidth_max,
            init_run=True,
            pad_io_nodes=self.pad_io_nodes,
            optimization_parameters=self.optimization_parameters
            max_parameters_per_partition=self.max_parameters_per_partition,
        )

        opt1.targets = targets
        opt1.generate_parameter_set()  # generate full param list
        param_set_default = opt1.params

        param_values_min = param_set_default.get_min_vals()
        param_values_max = param_set_default.get_max_vals()

        param_set_default.add_all_params_to_index_list()

        # create copies of the minimum and maximum parameters
        # for folding to use as upper and lower bounds for
        # optimization

        param_set_min = copy.deepcopy(param_set_default)
        param_set_min.set_values(param_values_min)

        param_set_max = copy.deepcopy(param_set_default)
        param_set_max.set_values(param_values_max)

        # run once to initialize all the lists and objects
        param_set_min.apply_updates(self.optimization_parameters)
        param_set_max.apply_updates(self.optimization_parameters)


        param_set_min.assign_involved_nodes()
        param_set_max.assign_involved_nodes()

       
        # assign maximum throughput achievable
        opt1.optimize(max_nodes_in_partition=1, target_parameters=["SIMD", "PE","parallel_window"])
        init_model = init_model.transform(AnnotateCycles())
        maximum_achievable_throughput = init_model.analysis(dataflow_performance)["max_cycles"]

       # assert True==False
        limits = DEFAULT_RES_LIMITS
        self.max_luts = limits[0] * sum(
            [r["LUT"] for r in platforms[self.platform](self.devices).resource_count_dict.values()]
        )
        self.max_bram = limits[2] * sum(
            [
                r["BRAM_18K"]
                for r in platforms[self.platform](self.devices).resource_count_dict.values()
            ]
        )
        self.max_uram = limits[3] * sum(
            [r["URAM"] for r in platforms[self.platform](self.devices).resource_count_dict.values()]
        )
        self.max_dsp = limits[4] * sum(
            [r["DSP"] for r in platforms[self.platform](self.devices).resource_count_dict.values()]
        )

        targets["LUT"] = max(self.max_luts, 0.001)
        targets["BRAM_18K"] = max(self.max_bram, 0.001)
        targets["DSP"] = max(self.max_dsp, 0.001)
        targets["URAM"] = max(self.max_uram, 0.001)

        opt2 = Optimizer(
            model,
            "padded OPT",
            targets,
            self.hard_constraint_target,
            target_cycles_per_frame=current_throughput_target,
            padding=self.padding,
            fpgapart=self.fpgapart,
            maxfun_per_parameter=self.effort,
            parameters_to_apply=self.optimization_parameters,
            enable_folding_dwc_heuristic=self.enable_folding_dwc_heuristic,
            verbose=self.verbose,
            mvau_wwidth_max=self.mvau_wwidth_max,
            init_run=False,
            pad_io_nodes=self.pad_io_nodes,
            optimization_parameters=self.optimization_parameters,
            max_parameters_per_partition=self.max_parameters_per_partition,
        )

        opt2.targets = targets
        opt2.generate_parameter_set()  # generate full param list

        # First pass which deals with folding factors only

        optimization_attempts = 0
        last_limited = False
        fifos_in_the_loop = True
        last_successful_throughput_target = self.target_cycles_per_frame
        
        current_step = 1
        min_step = 0.05

        opt2_tmp = copy.deepcopy(opt2)
        pre_folding_model = copy.deepcopy(opt2.model)
        last_good_model = copy.deepcopy(opt2.model)
        print("MAX ATTEMPTS: ")
        print(self.max_attempts)
        print("entering global optimization passes")
        while current_step > min_step and optimization_attempts < self.max_attempts:
            targets["max_cycles"] = current_throughput_target
            opt2 = copy.deepcopy(opt2_tmp)
            #opt2.model = copy.deepcopy(pre_folding_model)
            opt2.targets = targets
            opt2.generate_parameter_set()  # generate full param list
            print(f'current_step: {current_step}, max_cycles: {targets["max_cycles"]}, attempts:{optimization_attempts}')
            # dont optimize if throughput request is impossible to meet


            opt2.target_cycles_per_frame=current_throughput_target
            if self.optimize_folding is True:
                opt2.optimize(max_nodes_in_partition=3, target_parameters=["SIMD", "PE"])

            # Second pass which adjusts ram style for memory and resource types for compute
            if self.optimize_resource_types is True:
                opt2.optimize(
                    max_nodes_in_partition=min(len(model.graph.node), 8),
                    target_parameters=["ram_style", "resType","ram_style_thresholds"],
                )

            model = opt2.model

            # generate extra model with fifos and dwcs for the final estimate
                
            if self.consider_dwc_costs:
                model = model.transform(InsertDWC())
                model = model.transform(SpecializeLayers(self.fpgapart))
            
            if self.enable_folding_fifo_heuristic and self.max_attempts != 1:

                # store model to use in the builder
                model_dir = "folded_model.onnx"

                model = insert_and_size_fifos(model_dir,model,self.platform,
                                              self.fpgapart,
                                              self.consider_dwc_costs, 
                                              self.auto_fifo_strategy)
                model = model.transform(SpecializeLayers(self.fpgapart))

            resources = {}
            for n in opt2.model.graph.node:
                node_inst = getCustomOp(n)
                resources[node_inst] = node_inst.node_res_estimation(self.fpgapart)
            metrics = aggregate_dict_keys(resources)

            # extract costs
            overshot = False
            for resource in self.target_resources:
                if metrics[resource] > targets[resource]:
                    print(f"{resource}: {metrics[resource]} > {targets[resource]}")
                    overshot = True

            if overshot:
                # if we overshot, we try again, but with half the step size
                print(f"overshot, new target: {current_throughput_target}")
                print(f"step decreasing from {current_step} to {current_step/2}")
                print(f"target changing from {current_throughput_target} to {int(last_successful_throughput_target - last_successful_throughput_target*(current_step/2))} by decreasing on {last_successful_throughput_target} by a lower step")                
                current_step /= 2
                current_throughput_target = int(last_successful_throughput_target - last_successful_throughput_target*current_step)
                
                if self.max_attempts == 1:
                    last_good_model = copy.deepcopy(opt2.model)


            else:
                print(f"did not overshoot, still halving step size and repeating")
                if last_limited:
                    current_step /= 2

                else:
                    for resource in self.target_resources:
                        budget_left = 1 - (metrics[resource] / targets[resource])
                        print(f"budget: {budget_left} from {metrics[resource]} / {targets[resource]} ratio for {resource}")
                        current_step = min(current_step,budget_left)
                        print(f"new step: {current_step}")
      
                new_throughput = int(last_successful_throughput_target - last_successful_throughput_target*current_step)
                print(f"did NOT overshoot, new target: {current_throughput_target} from {last_successful_throughput_target}")
                last_good_model = copy.deepcopy(opt2.model)
                last_successful_throughput_target = copy.copy(current_throughput_target)
                current_throughput_target = new_throughput
                print(f"new step for the attempt: {current_step}")
            if current_throughput_target < maximum_achievable_throughput:
                print("requested beyond maximal folding, limiting")
                last_limited = True
                
                current_throughput_target = maximum_achievable_throughput
            else:
                last_limited = False
            optimization_attempts += 1

        model = last_good_model

        print("optimizer FINAL padding ratio: ",opt2.padding_result)

        if self.insert_dwcs:
            # In case future steps do not insert DWCs
            model = model.transform(InsertDWC())
            model = model.transform(SpecializeLayers(self.fpgapart))

        # necessary final transformation
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        #assert True == False

        # perform input and output tensor shape adjustment
        # this is only going to have effects if padding was performed
        # The tensor shape change on the IOs will require modification
        # of the host, with either padding of the input or cropping of the output
        # to get an equivalent result.


        if self.pad_io_nodes:
            print("Padding IO nodes!")
            input_mw_padded = getCustomOp(model.graph.node[0]).get_normal_input_shape()
            output_mh_padded = getCustomOp(model.graph.node[-1]).get_normal_output_shape()
            x = np.zeros(input_mw_padded, dtype=np.float32)
            y = np.zeros(output_mh_padded, dtype=np.float32)

            input_name = model.graph.input[0].name
            output_name = model.graph.output[0].name

            if len(model.graph.input) != 0:
                model.graph.input.remove(model.graph.input[0])
            input_x = helper.make_tensor_value_info(
                model.graph.node[0].input[0], TensorProto.FLOAT, [*input_mw_padded]
            )
            model.graph.input.append(input_x)

            if len(model.graph.output) != 0:
                model.graph.output.remove(model.graph.output[0])
            output_y = helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, [*output_mh_padded]
            )
            model.graph.output.append(output_y)

        return (model, False)



    def divisors(self,num):
        for x in range(1, num + 1):
            if (num % x) == 0:
                yield x

    def optimize_attribute_val(self, node_inst, max_val, attr_name):
        node_inst.set_nodeattr(attr_name, 1)
        for val in self.divisors(max_val):
            node_inst.set_nodeattr(attr_name, val)
            cyc = node_inst.get_exp_cycles()
            if cyc < self.target_cycles_per_frame:
                # finish if target met
                break

    def apply_naive_folding(self, model):
        """
        A naive folding optimizer implementation

        If two_pass_relaxation is enabled,
        SetFolding will internally run a second time if the target cycles from the
        first pass could not be achieved, instead using the achievable target (which
        may be constrained by a single node) to obtain a balanced pipeline.

        Notable exceptions and special behavior:

        When folding dense convolution/FC compute engines ("MVAU"/MatrixVectorActivation),
        which have two attributes (PE and SIMD):

        * first increases SIMD while weight stream width per PE is <= mvau_wwidth_max
        (configurable in the SetFolding initializer, defaults to 36)
        * then increases PE until the target is met or max PE reached

        When folding depthwise convolutions ("VVAU"/VectorVectorActivation)
        or spatial reduction ops (Pool_Batch):

        * the producer of the node is expected to be a ConvolutionInputGenerator
        with depthwise=1, whose SIMD value will be set equal to the PE value of
        its consumer node
        * the VVAU also supports SIMD ("input window") parallelism next to
        PE ("channels"), but current ConvInpGen limitations require PE to be fully
        unfolded before SIMD is increased
        """

        graph = model.graph
        # these ops use PE parallelism, up to a max value of NumChannels
        pe_ops = [
            "AddStreams_hls",
            "ChannelwiseOp_hls",
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Thresholding_hls",
            "Thresholding_rtl",
            "StreamingMaxPool_hls",
        ]
        # these ops use SIMD parallelism, up to a max value of NumChannels
        # ConvolutionInputGenerator* has a special case when depthwise=1
        # ConvolutionInputGenerator_rtl supports additional parallelism by
        # setting parallel_window=1 mode after maxing out SIMD
        simd_ops = [
            "DownSampler_hls",
            "FMPadding_hls",
            "FMPadding_Pixel_hls",
            "ConvolutionInputGenerator_hls",
            "ConvolutionInputGenerator_rtl",
        ]
        # these ops are preceded by depthwise SWG and have special behavior,
        # as explained in the SetFolding docstring
        depthwise_op_exceptions = ["VVAU_hls", "VVAU_rtl", "Pool_hls"]
        for node in graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            op_type = node.op_type
            node_inst = getCustomOp(node)
            if op_type in ["MVAU_hls", "MVAU_rtl"]:
                max_simd = node_inst.get_nodeattr("MW")
                max_pe = node_inst.get_nodeattr("MH")
                node_inst.set_nodeattr("PE", 1)
                node_inst.set_nodeattr("SIMD", 1)
                # increase SIMD until either we meet
                # the target or weight stream becomes
                # too wide
                for simd_val in self.divisors(max_simd):
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    node_inst.set_nodeattr("SIMD", simd_val)
                    cyc = node_inst.get_exp_cycles()
                    if cyc < self.target_cycles_per_frame and simd_val > (max_simd / 1024):
                        # finish if target met and simd value is not too low
                        break
                    if (
                        node_inst.get_input_datatype(1).bitwidth() * node_inst.get_nodeattr("SIMD")
                        > self.mvau_wwidth_max
                    ):
                        # revert if we've gone above width threshold
                        node_inst.set_nodeattr("SIMD", prev_simd_val)
                        break
                # increase PE until target met or reached max_pe
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in pe_ops:
                max_pe = node_inst.get_nodeattr("NumChannels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type == "LabelSelect_hls":
                max_pe = node_inst.get_nodeattr("Labels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in depthwise_op_exceptions:
                # init/reset SIMD of VVAU
                if op_type in ["VVAU_hls", "VVAU_rtl"]:
                    node_inst.set_nodeattr("SIMD", 1)
                max_pe = node_inst.get_nodeattr("Channels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                # increase SIMD for VVAU once PE is exhausted
                pe = node_inst.get_nodeattr("PE")
                cyc = node_inst.get_exp_cycles()
                if (
                    op_type in ["VVAU_hls", "VVAU_rtl"]
                    and pe == max_pe
                    and cyc > self.target_cycles_per_frame
                ):
                    max_simd = np.prod(node_inst.get_nodeattr("Kernel"))
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                # also set the folding of the upsteam DW SWU
                # which must be identical to this node
                swu_node = model.find_producer(node.input[0])
                if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                    swu_node_inst = getCustomOp(swu_node)
                    swu_node_inst.set_nodeattr("SIMD", pe)
                    # enable parallel_window mode of RTL SWG if needed
                    if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                        if op_type.startswith("VVAU") and node_inst.get_nodeattr("SIMD") > 1:
                            swu_node_inst.set_nodeattr("parallel_window", 1)
                        else:
                            swu_node_inst.set_nodeattr("parallel_window", 0)
                else:
                    if op_type in ["VVAU_hls", "VVAU_rtl"]:
                        ksize = np.prod(node_inst.get_nodeattr("Kernel"))
                    elif op_type == "Pool_hls":
                        ksize = node_inst.get_nodeattr("KernelSize")
                    else:
                        raise Exception("Undefined edge case for %s" % op_type)
                    if ksize != 1:  # pointwise vvau/pool lack a SWU
                        raise Exception("Expected SWU on DW op input, found " + swu_node.op_type)
            elif op_type in simd_ops:
                if op_type.startswith("ConvolutionInputGenerator"):
                    depthwise = node_inst.get_nodeattr("depthwise")
                    if depthwise == 0:
                        max_simd = node_inst.get_nodeattr("IFMChannels")
                        # init/reset parallel_window mode of RTL SWG
                        if op_type == "ConvolutionInputGenerator_rtl":
                            node_inst.set_nodeattr("parallel_window", 0)
                        self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                        # enable parallel_window mode of RTL SWG if needed
                        simd = node_inst.get_nodeattr("SIMD")
                        cyc = node_inst.get_exp_cycles()
                        if (
                            op_type == "ConvolutionInputGenerator_rtl"
                            and simd == max_simd
                            and cyc > self.target_cycles_per_frame
                        ):
                            node_inst.set_nodeattr("parallel_window", 1)
                    else:
                        # depthwise SWGs are handled separately
                        continue
                else:
                    max_simd = node_inst.get_nodeattr("NumChannels")
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
            else:
                warnings.warn("SetFolding doesn't know how to handle op_type " + op_type)

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        if self.two_pass_relaxation:
            perf_dict = model.analysis(dataflow_performance)
            if perf_dict["max_cycles"] > self.target_cycles_per_frame:
                # run again, but with lower target (that we managed) -- this
                # may be coming from a single node's constraints, but we want
                # to balance the entire dataflow pipeline instead
                # no two_pass_relaxation this time -- no guarantee we'll
                # converge otherwise
                warnings.warn(
                    "Node %s is bottleneck with %d cycles, running second pass"
                    % (perf_dict["max_cycles_node_name"], perf_dict["max_cycles"])
                )
                model = model.transform(
                    SetFolding(
                        target_cycles_per_frame=perf_dict["max_cycles"],
                        mvau_wwidth_max=self.mvau_wwidth_max,
                        two_pass_relaxation=False,
                        padding=0,
                    )
                )

        # necessary final transforms
        if self.insert_dwcs:
            model.transform(InsertDWC())
            #model.transform(InsertDWC())

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        return (model, False)

    def apply(self, model):
        if self.style == "naive":
            return self.apply_naive_folding(model)
        else:
            return self.apply_optimized_folding(model)
