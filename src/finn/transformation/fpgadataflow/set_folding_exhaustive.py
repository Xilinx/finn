from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.util.fpgadataflow import is_fpgadataflow_node

from copy import deepcopy

boards = {
    "Pynq-Z1": {"LUT": 53200, "BRAM_18K": 280, "URAM": 0, "DSP": 220},
    "ZCU104": {"LUT": 230000, "BRAM_18K": 610, "URAM": 92, "DSP": 1728},
}


def get_sorted_attrs(attrs, op_type):
    if op_type == "StreamingFCLayer_Batch":
        return sorted(
            attrs.items(),
            key=lambda k_v: (k_v[1].get("cycles"), k_v[0][0]),
            reverse=True,
        )
    else:
        return sorted(attrs.items(), key=lambda k_v: k_v[1].get("cycles"), reverse=True)


def get_luts(model):
    luts = 0
    for node in model.graph.node:
        node_inst = getCustomOp(node)
        luts += node_inst.lut_estimation()
    return luts


class SetFoldingExhaustive(Transformation):
    def __init__(
        self,
        target_cycles_per_frame,
        clk_ns,
        board=None,
        mvau_wwidth_max=36,
        scale_ratio=0.85,
        from_scratch=True,
    ):
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.clk_ns = clk_ns
        self.board = board if board is not None else None
        self.mvau_wwidth_max = mvau_wwidth_max
        self.from_scratch = from_scratch
        self.max_luts = (
            scale_ratio * boards[board]["LUT"] if board is not None else float("inf")
        )
        # self.max_bram = scale_ratio * boards[board]["BRAM_18K"]
        # self.max_uram = scale_ratio * boards[board]["URAM"]
        # self.max_dsp = scale_ratio * boards[board]["DSP"]
        self.pe_ops = [
            "AddStreams_Batch",
            "ChannelwiseOp_Batch",
            "DuplicateStreams_Batch",
            "GlobalAccPool_Batch",
            "Thresholding_Batch",
        ]
        self.simd_ops = ["DownSampler", "FMPadding_Batch", "ConvolutionInputGenerator"]
        self.depthwise_op_exceptions = ["Vector_Vector_Activate_Batch", "Pool_Batch"]

    def get_attrs(self, model):
        """
        Performs an exhaustive search over the possible SIMD / PE / (SIMD, PE)
        configurations for each node
        """

        def _iterate_node_attrs(node_inst, val, attr_name):
            node_inst.set_nodeattr(attr_name, val)

            return {
                "inWidth": node_inst.get_instream_width(),
                "outWidth": node_inst.get_outstream_width(),
                "lut": node_inst.lut_estimation(),
                "cycles": node_inst.get_exp_cycles(),
                "viability": True,
            }

        attr_model = deepcopy(model)
        graph = attr_model.graph
        all_attrs = dict()

        for node in graph.node:
            if not is_fpgadataflow_node(node):
                continue
            children = model.find_consumers(node.output[0])
            producer = model.find_producer(node.input[0])

            all_attrs[node.name] = dict()
            all_attrs[node.name]["child_layer"] = children[0].name if children else None
            all_attrs[node.name]["parent_layer"] = producer.name if producer else None

            all_attrs[node.name]["possible_attrs"] = dict()

            node_inst = getCustomOp(node)
            # Dealing with SIMD Ops
            if node.op_type in self.simd_ops:
                if node.op_type == "ConvolutionInputGenerator":
                    depthwise = node_inst.get_nodeattr("depthwise")
                    if depthwise == 0:
                        max_simd = node_inst.get_nodeattr("IFMChannels")
                    else:
                        # SIMD value will be set equal to the PE value of its consumer node
                        continue
                else:
                    max_simd = node_inst.get_nodeattr("NumChannels")

                min_simd = node_inst.get_nodeattr("SIMD")

                while min_simd <= max_simd:
                    if not max_simd % min_simd:
                        all_attrs[node.name]["possible_attrs"][
                            min_simd
                        ] = _iterate_node_attrs(node_inst, min_simd, "SIMD")
                    min_simd += 1

            # Dealing with PE and depthwise exception Ops
            elif (
                node.op_type in [*self.pe_ops, *self.depthwise_op_exceptions]
                or node.op_type == "LabelSelect_Batch"
            ):
                if node.op_type in self.pe_ops:
                    max_pe = node_inst.get_nodeattr("NumChannels")
                if node.op_type in self.depthwise_op_exceptions:
                    max_pe = node_inst.get_nodeattr("Channels")
                if node.op_type == "LabelSelect_Batch":
                    max_pe = node_inst.get_nodeattr("Labels")

                min_pe = node_inst.get_nodeattr("PE")

                while min_pe <= max_pe:
                    if not max_pe % min_pe:
                        all_attrs[node.name]["possible_attrs"][
                            min_pe
                        ] = _iterate_node_attrs(node_inst, min_pe, "PE")
                    min_pe += 1

            # Dealing with FClayers - using both (SIMD, PE)
            elif node.op_type == "StreamingFCLayer_Batch":
                max_simd = node_inst.get_nodeattr("MW")
                max_pe = node_inst.get_nodeattr("MH")

                min_simd = node_inst.get_nodeattr("SIMD")
                old_pe = node_inst.get_nodeattr("PE")
                min_pe = node_inst.get_nodeattr("PE")

                while min_simd <= max_simd:
                    min_pe = old_pe
                    while min_pe <= max_pe:
                        if (
                            not max_simd % min_simd
                            and not max_pe % min_pe
                            and not node_inst.get_weight_datatype().bitwidth()
                            * min_simd
                            > self.mvau_wwidth_max
                        ):
                            node_inst.set_nodeattr("SIMD", min_simd)
                            node_inst.set_nodeattr("PE", min_pe)

                            all_attrs[node.name]["possible_attrs"][
                                (min_simd, min_pe)
                            ] = {
                                "inWidth": node_inst.get_instream_width(),
                                "outWidth": node_inst.get_outstream_width(),
                                "lut": node_inst.lut_estimation(),
                                "cycles": node_inst.get_exp_cycles(),
                                "difference": abs(min_simd - min_pe),
                                "viability": True,
                            }

                        min_pe += 1
                    min_simd += 1
        return all_attrs

    def inst_model(self, model):
        """
        If from_scratch = True, sets node attributes to 1. Logs LUTs and expected cycles of each node.
        Instantiates the among_slowest dict with the slowest node, which is used to update slow nodes
        per iteration.
        """
        new_model = deepcopy(model)
        graph = new_model.graph
        values = dict()

        for node in graph.node:
            node_inst = getCustomOp(node)

            if self.from_scratch:
                if node.op_type == "StreamingFCLayer_Batch":
                    node_inst.set_nodeattr("SIMD", 1)
                    node_inst.set_nodeattr("PE", 1)

                elif node.op_type in [*self.pe_ops, *self.depthwise_op_exceptions]:
                    node_inst.set_nodeattr("PE", 1)

                elif node.op_type in self.simd_ops:
                    node_inst.set_nodeattr("SIMD", 1)

            if hasattr(node_inst, "lut_estimation"):
                node_values = {
                    "lut": node_inst.lut_estimation(),
                    "cycles": node_inst.get_exp_cycles(),
                }
            else:
                # TODO: Not sure how to optimize layers with no LUT estimation
                # For now, assume that LUT usage is insignificant and decrease
                # cycle count if necessary.
                node_values = {"lut": 0, "cycles": node_inst.get_exp_cycles()}

            values[node.name] = node_values

        node_cycles_sorted = sorted(
            values.items(), key=lambda node: node[1].get("cycles"), reverse=True
        )
        among_slowest = {node_cycles_sorted[0][0]: node_cycles_sorted[0][1]}

        return new_model, values, among_slowest

    def get_next_val(self, node, current_attrs, total_luts):
        """
        Try to fetch next folding config for a given node. Next folding config
        is chosen s.t. the incremental change in cycle count is the smallest
        s.t. we can obtain a finer granularity in optimization.
        """
        node_inst = getCustomOp(node)
        child_layer = current_attrs.get(node.name).get("child_layer")
        done = False

        if child_layer:
            child_layer_attrs = current_attrs.get(child_layer)

        if node.op_type == "StreamingFCLayer_Batch":
            node_key = (node_inst.get_nodeattr("SIMD"), node_inst.get_nodeattr("PE"))
            node_luts = node_inst.lut_estimation()
        elif node.op_type in [*self.pe_ops, *self.depthwise_op_exceptions]:
            node_key = node_inst.get_nodeattr("PE")
            if hasattr(node_inst, "lut_estimation"):
                node_luts = node_inst.lut_estimation()
            else:
                node_luts = 0
        elif node.op_type in self.simd_ops:
            node_key = node_inst.get_nodeattr("SIMD")
            if hasattr(node_inst, "lut_estimation"):
                node_luts = node_inst.lut_estimation()
            else:
                node_luts = 0

        node_attrs = current_attrs.get(node.name).get("possible_attrs")
        sorted_attrs = get_sorted_attrs(node_attrs, node.op_type)

        for idx, v in enumerate(sorted_attrs):
            if v[0] == node_key:
                next_opt = idx + 1
                # If the current setting for the node is the highest setting, we're done optimizing
                if next_opt > len(sorted_attrs):
                    done = True
                    return sorted_attrs[idx][0], current_attrs, done

                # Multiple folding settings have the same cycle count. During optimization
                # we want to choose the next folding configuration that has a lower cycle count
                while node_inst.get_exp_cycles() <= sorted_attrs[next_opt][1].get(
                    "cycles"
                ):
                    next_opt += 1

                # Try to find a viable config with same cycles
                if not sorted_attrs[next_opt][1].get("viability"):

                    # Return if we are at end of possible configs
                    if next_opt > len(sorted_attrs):
                        done = True
                        return sorted_attrs[idx][0], current_attrs, done

                    # Multiple folding configurations have same cycle count,
                    # but some of them may not be viable.
                    # Iterative over them until we find a viable configuration
                    while sorted_attrs[next_opt][1].get("cycles") == sorted_attrs[
                        next_opt + 1
                    ][1].get("cycles"):
                        if not sorted_attrs[next_opt + 1][1].get("viability"):
                            next_opt += 1
                        else:
                            next_opt += 1
                            break

                # We've found a folding configuration that satisfies constraints and is within budget
                if (sorted_attrs[next_opt][1].get("viability")) and (
                    sorted_attrs[next_opt][1].get("lut") - node_luts + total_luts
                ) < self.max_luts:

                    # update viability of next layer
                    if child_layer:
                        for k, v in child_layer_attrs.get("possible_attrs").items():
                            if sorted_attrs[next_opt][1].get("outWidth") > v.get(
                                "inWidth"
                            ):
                                if not sorted_attrs[next_opt][1].get(
                                    "outWidth"
                                ) % v.get("inWidth"):
                                    v["viability"] = True
                                else:
                                    v["viability"] = False
                            elif sorted_attrs[next_opt][1].get("outWidth") <= v.get(
                                "inWidth"
                            ):
                                if not v.get("inWidth") % sorted_attrs[idx + 1][1].get(
                                    "outWidth"
                                ):
                                    v["viability"] = True
                                else:
                                    v["viability"] = False

                    return sorted_attrs[next_opt][0], current_attrs, done
                else:
                    # Didn't find a viable folding configuration for cycle count at 'level' above
                    # the current cycle count. Maybe possible to find viable configurations with lower
                    # cycle count and still stay within LUT budget?

                    # If we have the slowest layer and find no possible configs we need to stop optimization
                    optimizable = False

                    for attr in sorted_attrs[next_opt:]:
                        attr_luts = attr[1].get("lut")
                        if (
                            attr_luts - node_luts + total_luts
                        ) < self.max_luts and attr[1].get("viability"):
                            if child_layer:
                                for k, v in child_layer_attrs.get(
                                    "possible_attrs"
                                ).items():
                                    if attr[1].get("outWidth") > v.get("inWidth"):
                                        if not attr[1].get("outWidth") % v.get(
                                            "inWidth"
                                        ):
                                            v["viability"] = True
                                        else:
                                            v["viability"] = False
                                    else:
                                        if not v.get("inWidth") % attr[1].get(
                                            "outWidth"
                                        ):
                                            v["viability"] = True
                                        else:
                                            v["viability"] = False

                            return attr[0], current_attrs, done
                        elif (attr_luts - node_luts + total_luts) > self.max_luts:
                            done = True
                        elif attr[1].get("Viable"):
                            optimizable = True
                    if not optimizable:
                        done = True

                    return sorted_attrs[idx][0], current_attrs, done

    def incr_folding(self, model, among_slowest, attrs):
        """
        Loops over the nodes of the model and updates the folding of the nodes in `among_slowest`
        incrementally.
        """
        done = False
        new_model = deepcopy(model)
        graph = new_model.graph

        total_luts = get_luts(new_model)

        def _incr_node_attr(node, attrs, total_luts, among_slowest, attr_name):
            node_inst = getCustomOp(node)
            val, attrs, done = self.get_next_val(node, attrs, total_luts)

            old_lut = (
                node_inst.lut_estimation()
                if hasattr(node_inst, "lut_estimation")
                else 0
            )
            prev_attr_val = node_inst.get_nodeattr(attr_name)
            node_inst.set_nodeattr(attr_name, val)
            new_lut = (
                node_inst.lut_estimation()
                if hasattr(node_inst, "lut_estimation")
                else 0
            )

            if (total_luts - old_lut + new_lut) > self.max_luts:
                node_inst.set_nodeattr(attr_name, prev_attr_val)
                done = True
            else:
                among_slowest[node.name]["cycles"] = node_inst.get_exp_cycles()

                # update upstream ConvInpGen node
                if node.op_type in self.depthwise_op_exceptions:
                    swu_node = new_model.find_producer(node.input[0])
                    if swu_node.op_type == "ConvolutionInputGenerator":
                        swu_node_inst = getCustomOp(swu_node)
                        swu_node_inst.set_nodeattr("SIMD", val)

            total_luts -= old_lut
            total_luts += new_lut

            return total_luts, among_slowest, done

        for node in graph.node:
            if node.name in among_slowest.keys() and not among_slowest[node.name].get(
                "skip"
            ):
                node_inst = getCustomOp(node)
                op_type = node.op_type

                if op_type == "StreamingFCLayer_Batch":
                    old_lut = node_inst.lut_estimation()
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    prev_pe_val = node_inst.get_nodeattr("PE")

                    vals, attrs, done = self.get_next_val(node, attrs, total_luts)
                    new_simd_val, new_pe_val = vals

                    node_inst.set_nodeattr("SIMD", new_simd_val)
                    node_inst.set_nodeattr("PE", new_pe_val)

                    new_lut = node_inst.lut_estimation()

                    if (total_luts - old_lut + new_lut) > self.max_luts:
                        node_inst.set_nodeattr("SIMD", prev_simd_val)
                        node_inst.set_nodeattr("PE", prev_pe_val)
                        done = True

                    else:
                        among_slowest[node.name]["cycles"] = node_inst.get_exp_cycles()

                        total_luts -= old_lut
                        total_luts += new_lut

                elif op_type == "ConvolutionInputGenerator":
                    # If child is not in depthwise_op_exceptions, update ConvInpGen node
                    child_node = new_model.find_consumers(node.output[0])[0]
                    if child_node.op_type in self.depthwise_op_exceptions:
                        continue
                    else:
                        total_luts, among_slowest, done = _incr_node_attr(
                            node, attrs, total_luts, among_slowest, "SIMD"
                        )

                elif op_type in self.depthwise_op_exceptions:
                    total_luts, among_slowest, done = _incr_node_attr(
                        node, attrs, total_luts, among_slowest, "PE"
                    )

                elif op_type in self.simd_ops:
                    total_luts, among_slowest, done = _incr_node_attr(
                        node, attrs, total_luts, among_slowest, "SIMD"
                    )

                elif op_type in self.pe_ops:
                    total_luts, among_slowest, done = _incr_node_attr(
                        node, attrs, total_luts, among_slowest, "PE"
                    )

        return new_model, attrs, among_slowest, done

    def update_slowest(
        self, model, among_slowest, prev_slowest_cycles=0, prev_slowest_name=None
    ):
        """
        Updates the list of slowest nodes. Upstream DW SWUs are always removed as updates to their
        downstream counterparts will also update them.
        """
        slowest_layer = sorted(
            among_slowest.items(), key=lambda item: item[1].get("cycles"), reverse=True
        )[0]

        added = False
        for node in model.graph.node:
            node_inst = getCustomOp(node)

            if node_inst.get_exp_cycles() > slowest_layer[1].get("cycles"):
                # remove ConvInpGen nodes if they have depthwise_op_exception node children
                if (
                    node.op_type == "ConvolutionInputGenerator"
                    and model.find_consumers(node.output[0])[0].op_type
                    in self.depthwise_op_exceptions
                ):
                    among_slowest.pop(node.name, None)
                else:
                    among_slowest[node.name] = {"cycles": node_inst.get_exp_cycles()}
                    added = True

        new_among_slowest = dict()

        # Skip updating nodes if they still have lower cycle count than slowest node(s) updated
        # in previous iteration
        for node, values in among_slowest.items():
            if values.get("cycles") < prev_slowest_cycles:
                values["skip"] = True
                new_among_slowest[node] = values
            else:
                values["skip"] = False
                new_among_slowest[node] = values

        # In some cases, an iteration of updates will only
        if prev_slowest_name:
            for node, vals in new_among_slowest.items():
                if vals["cycles"] > prev_slowest_cycles:
                    new_among_slowest[prev_slowest_name]["skip"] = True
                    break

        return model, new_among_slowest, added

    def fold_iteratively(self, model, among_slowest, attrs):
        added = False
        done = False
        model_luts = 0
        while model_luts < self.max_luts and not done:
            model_luts = get_luts(model)
            if not added:

                model, attrs, among_slowest, done = self.incr_folding(
                    model, among_slowest, attrs
                )

                sorted_among_slowest = sorted(
                    among_slowest.items(),
                    key=lambda item: item[1].get("cycles"),
                    reverse=True,
                )
                prev_slowest_cycles, prev_slowest_name = (
                    sorted_among_slowest[0][1].get("cycles"),
                    sorted_among_slowest[0][0],
                )

            model, among_slowest, added = self.update_slowest(
                model, among_slowest, prev_slowest_cycles, prev_slowest_name
            )

            slowest_layer = sorted(
                among_slowest.items(),
                key=lambda item: item[1].get("cycles"),
                reverse=True,
            )[0]

            if slowest_layer[1].get("cycles") < self.target_cycles_per_frame:
                print(f"Reached target of {self.target_cycles_per_frame} cycles")
                break

        return model, among_slowest, attrs

    def apply(self, model):
        model, model_values, among_slowest = self.inst_model(model)

        all_attrs = self.get_attrs(model)

        new_model, among_slowest, attrs = self.fold_iteratively(
            model, among_slowest, all_attrs
        )

        return (new_model, False)
