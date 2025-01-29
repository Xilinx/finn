
import math
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    gen_finn_dt_tensor,
    qonnx_make_model
)
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from bench_base import bench

class bench_mvau(bench):

    def _make_single_mvau_model(
        self,
        W,
        numInputVectors,
        pe,
        simd,
        m,
        wdt,
        idt,
        odt,
        T=None,
        tdt=None,
        mem_mode="const",
        ram_style="auto",
        ram_style_thresholds="auto",
    ):
        mw = W.shape[0]
        mh = W.shape[1]

        # there are two ways to implement bipolar weights and inputs for
        # MatrixVectorActivation:
        # - specify their datatypes as such
        # - specify their datatypes as BINARY as use binaryXnorMode
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            # we'll internally convert weights/inputs to binary and specify the
            # datatypes as such, and also set the binaryXnorMode attribute to 1
            export_wdt = DataType["BINARY"]
            export_idt = DataType["BINARY"]
            binary_xnor_mode = 1
        else:
            export_wdt = wdt
            export_idt = idt
            binary_xnor_mode = 0

        # numInputVectors for dense = [N]
        # numInputVectors for conv  = [N, H, W]
        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, numInputVectors + [mw])
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, numInputVectors + [mh])
        if T is not None:
            no_act = 0
            node_inp_list = ["inp", "weights", "thresh"]
            if odt == DataType["BIPOLAR"]:
                actval = 0
            else:
                actval = odt.min()
        else:
            # no thresholds
            node_inp_list = ["inp", "weights"]
            actval = 0
            no_act = 1
        mvau_node = helper.make_node(
            "MVAU_hls", #TODO: add rtl support (configurable as param)
            node_inp_list,
            ["outp"],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            MW=mw,
            MH=mh,
            SIMD=simd,
            PE=pe,
            M=m,
            numInputVectors=numInputVectors,
            inputDataType=export_idt.name,
            weightDataType=export_wdt.name,
            outputDataType=odt.name,
            ActVal=actval,
            binaryXnorMode=binary_xnor_mode,
            noActivation=no_act,
            resType="lut",
            mem_mode=mem_mode,
            ram_style=ram_style,
            ram_style_thresholds=ram_style_thresholds,
            runtime_writeable_weights=0,
        )

        graph = helper.make_graph(nodes=[mvau_node], name="mvau_graph", inputs=[inp], outputs=[outp])
        model = qonnx_make_model(graph, producer_name="mvau-model")
        model = ModelWrapper(model)

        model.set_tensor_datatype("inp", idt)
        model.set_tensor_datatype("outp", odt)
        model.set_tensor_datatype("weights", wdt)
        # model.set_tensor_shape("weights", (channels, 1, k_h, k_w)) from VVAU
        if binary_xnor_mode:
            # convert bipolar to binary
            model.set_initializer("weights", (W + 1) / 2)
        else:
            model.set_initializer("weights", W)
        if T is not None:
            model.set_tensor_datatype("thresh", tdt)
            model.set_initializer("thresh", T)

        # Minimize weight & accumulator width to obtain realistic resource consumption
        # model = model.transform(InferShapes())
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(InferDataTypes())

        return model

    def step_make_model(self):
        # Read params
        idt = self.params["idt"]
        wdt = self.params["wdt"]
        act = self.params["act"]

        numInputVectors = self.params["nhw"]
        mw = self.params["mw"]
        mh = self.params["mh"]
        sf = self.params["sf"]
        nf = self.params["nf"]
        m = self.params["m"]

        mem_mode = self.params["mem_mode"]
        ram_style = self.params["ram_style"]
        ram_style_thr = self.params["ram_style_thr"]

        output_dict = {}

        # convert string to FINN DataType
        idt = DataType[idt]
        wdt = DataType[wdt]
        if act is not None:
            act = DataType[act]

        # Determine and log folding
        if sf == -1:
            sf = mw
        simd = mw // sf
        if nf == -1:
            nf = mh
        pe = mh // nf
        if mw % simd != 0 or mh % pe != 0:
            print("Invalid simd/pe configuration, skipping")
            return
        if m > 1 and (simd != mw or pe != mh):
            print("M > 1 not possible for non-max simd/pe, skipping")
            return
        output_dict["simd"] = simd
        output_dict["pe"] = pe

        # Generate weights
        np.random.seed(123456)  # TODO: verify or switch to modern numpy random generation

        W = gen_finn_dt_tensor(wdt, (mw, mh))

        if "sparsity_type" in self.params:
            sparsity_type = self.params["sparsity_type"]
        else:
            sparsity_type = "none"

        if sparsity_type == "none":
            if "sparsity_amount" in self.params:
                if self.params["sparsity_amount"] > 0:
                    print("sparsity amount > 0 not applicable for none sparsity, skipping")
                    return
        else:
            if self.params["sparsity_amount"] == 0:
                print("sparsity amount = 0 not applicable for selected sparsity, skipping")
                return
            if sparsity_type == "unstructured":
                idx = np.random.choice(
                    mw * mh, size=int(self.params["sparsity_amount"] * mw * mh), replace=False
                )
                W = np.reshape(W, -1)
                W[idx] = 0.0
                W = np.reshape(W, (mw, mh))
            elif sparsity_type == "rows_random":
                idx_mw = np.random.choice(mw, size=int(self.params["sparsity_amount"] * mw), replace=False)
                W[idx_mw, :] = 0.0
            elif sparsity_type == "cols_random":
                idx_mh = np.random.choice(mh, size=int(self.params["sparsity_amount"] * mh), replace=False)
                W[:, idx_mh] = 0.0
            elif sparsity_type == "rows_regular":
                if self.params["sparsity_amount"] == 0.25:
                    idx_mw = np.arange(0, mw, step=4)
                elif self.params["sparsity_amount"] == 0.5:
                    idx_mw = np.arange(0, mw, step=2)
                elif self.params["sparsity_amount"] == 0.75:
                    idx_mw = np.concatenate(
                        (np.arange(0, mw, step=4), np.arange(1, mw, step=4), np.arange(2, mw, step=4))
                    )
                else:
                    print("regular sparsity only applicable for amount 0.25/0.5/0.75, skipping")
                    return
                W[idx_mw, :] = 0.0
            elif sparsity_type == "cols_regular":
                if self.params["sparsity_amount"] == 0.25:
                    idx_mh = np.arange(0, mh, step=4)
                elif self.params["sparsity_amount"] == 0.5:
                    idx_mh = np.arange(0, mh, step=2)
                elif self.params["sparsity_amount"] == 0.75:
                    idx_mh = np.concatenate(
                        (np.arange(0, mh, step=4), np.arange(1, mh, step=4), np.arange(2, mh, step=4))
                    )
                else:
                    print("regular sparsity only applicable for amount 0.25/0.5/0.75, skipping")
                    return
                W[:, idx_mh] = 0.0

            else:
                print("ERROR: unknown sparsity type")
                raise Exception("ERROR: unknown sparsity type")

        # TODO: implement enforce option which prevents naturally occurring sparsity
        # params["sparsity_enforce"]
        # TODO: implement distribution option which selects between uniform/normal/??
        # params["sparsity_distribution"]

        # log resulting sparsity statistics
        # could be higher than selected due to naturally occurring sparsity
        num_zeros = (W == 0).sum()
        num_ones = (W == 1).sum() + (W == -1).sum()
        num_p2 = 0
        for w in np.nditer(W):
            if w != 0 and w != 1 and w != -1:
                if w > 0:
                    if math.log2(w).is_integer():
                        num_p2 = num_p2 + 1
                else:
                    if math.log2(-w).is_integer():
                        num_p2 = num_p2 + 1
        output_dict["zero_weights"] = round(num_zeros / W.size, 2)
        output_dict["easy_weights"] = round((num_zeros + num_ones + num_p2) / W.size, 2)

        # Generate thresholds
        if act is None:
            # no activation, produce accumulators
            T = None
            tdt = None
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                odt = DataType["UINT32"]
            else:
                odt = DataType["INT32"]
        else:
            odt = act
            # set range for threshold values according to worst-case accumulator range (not weight value specific)
            # this could result in some thresholds being clipped by MinimizeAccumulatorWidth
            # lower_range = calculate_matvec_accumulator_range(wdt.min() * np.ones_like(W), idt)
            # upper_range = calculate_matvec_accumulator_range(wdt.max() * np.ones_like(W), idt)
            # acc_min = min(min(lower_range), min(upper_range))
            # acc_max = max(max(lower_range), max(upper_range))
            # set range for threshold values according to actual accumulator range for the generated weights
            (acc_min, acc_max) = calculate_matvec_accumulator_range(W, idt)
            n_steps = act.get_num_possible_values() - 1
            T = np.random.randint(acc_min, acc_max - 1, (mh, n_steps)).astype(np.float32)
            # provide non-decreasing thresholds
            T = np.sort(T, axis=1)
            # generate thresholds for activation
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                tdt = DataType["UINT32"]
                # bias thresholds to be positive
                T = np.ceil((T + mw) / 2)
                assert (T >= 0).all()
            else:
                tdt = DataType["INT32"]

        # Create model
        model = self._make_single_mvau_model(
            W, numInputVectors, pe, simd, m, wdt, idt, odt, T, tdt, mem_mode, ram_style, ram_style_thr
        )
        model = model.transform(GiveUniqueNodeNames())
        node = model.get_nodes_by_op_type("MVAU_hls")[0]
        inst = getCustomOp(node)

        self.target_node = "MVAU_hls" # display results of analysis passes only for the first occurence of this op type
        return model, output_dict

    def run(self):
        self.steps_simple_model_flow()
