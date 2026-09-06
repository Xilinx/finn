import numpy as np
import warnings
from onnx import TensorProto, helper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import auto_pad_to_explicit_padding, get_by_name


class InferPixelPaddingDeconv(Transformation):
    """
    Lowering and conversion of ConvTranspose (NCHW) nodes to
    FMPadding_Pixel + Im2Col + MatMul (NHWC) surrounded by Transpose nodes
    note: this transformation produces a mix of hw layers and non hw layers
    to implement this on an FPGA the Im2Col and MatMul nodes need to be converted to hw layers
    after applying this transformation and the resulting transpose nodes need to be streamlined.
    See deconv test case under tests/fpgadataflow for an example.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "ConvTranspose":
                # conversion currently only supported for group=1
                group = get_by_name(n.attribute, "group").i
                if group != 1:
                    warnings.warn(
                        "%s : Only group=1 is currently supported. Can't infer PixelPaddingDeconv."
                        % n.name
                    )
                    continue
                deconv_input = n.input[0]
                deconv_output = n.output[0]
                idt = model.get_tensor_datatype(deconv_input)
                odt = model.get_tensor_datatype(deconv_output)
                k_h = get_by_name(n.attribute, "kernel_shape").ints[0]
                k_w = get_by_name(n.attribute, "kernel_shape").ints[1]
                stride_h = get_by_name(n.attribute, "strides").ints[0]
                stride_w = get_by_name(n.attribute, "strides").ints[1]
                weight_name = n.input[1]
                W_conv = model.get_initializer(weight_name)
                ifm_ch = model.get_tensor_shape(n.input[0])[1]  # assume NCHW
                ofm_ch = model.get_tensor_shape(n.output[0])[1]  # assume NCHW
                ifm_dim_h = model.get_tensor_shape(n.input[0])[2]  # assume NCHW
                ifm_dim_w = model.get_tensor_shape(n.input[0])[3]
                ofm_dim_h = model.get_tensor_shape(n.output[0])[2]  # assume NCHW
                ofm_dim_w = model.get_tensor_shape(n.output[0])[3]
                dilation_attr = get_by_name(n.attribute, "dilations")
                if dilation_attr is not None:
                    dilation = dilation_attr.ints
                else:
                    dilation = [1, 1]  # default value
                # handle both auto_pad and explicit padding
                auto_pad = get_by_name(n.attribute, "auto_pad")
                if auto_pad is not None:
                    # find equivalent specified padding
                    auto_pad = auto_pad.s.decode("utf-8")
                    if auto_pad == "NOTSET":
                        # use specified padding
                        pad = get_by_name(n.attribute, "pads").ints
                    else:
                        pad = auto_pad_to_explicit_padding(
                            auto_pad,
                            ifm_dim_h,
                            ifm_dim_w,
                            k_h,
                            k_w,
                            stride_h,
                            stride_w,
                            len(model.get_tensor_shape(n.input[0])) - 2,
                        )
                else:
                    # use specified padding
                    pad = get_by_name(n.attribute, "pads").ints

                # If len(pad) == 2, assume no padding for other dimension
                if len(pad) == 2:  # only one dimension should be padded
                    assert (
                        ifm_dim_h == 1 or ifm_dim_w == 1
                    ), "Padding is assumed to be 1D, image is 2D"
                # reuse ConvTranspose weights for new matmul weights
                # conv weights are [IFM][OFM][k][k]
                # We need to rotate the weights and make them [OFM][IFM][k][k]
                # for pixel padding deconv to remain mathematically equivalent
                # and then convert to [OFM][k][k][IFM] (to remain compatible
                # with finn-hlslib and how it does im2col/sliding window)
                W_conv = np.rot90(W_conv, 2, [2, 3])
                W_conv = np.moveaxis(W_conv, 0, 1)
                W_matmul = W_conv.transpose(0, 2, 3, 1)  # W_conv = [OFM, IFM, k_H, k_W]
                # reshape into [OFM][k*k*IFM] matrix
                W_matmul = W_matmul.reshape(ofm_ch, ifm_ch * k_h * k_w)
                # transpose to get ONNX-compatible [k*k*IFM][OFM] matrix
                W_matmul = W_matmul.T
                model.set_initializer(weight_name, W_matmul)

                # Compute intermediate parameters
                padded_odim_h = ifm_dim_h + (ifm_dim_h - 1) * (stride_h - 1)
                padded_odim_w = ifm_dim_w + (ifm_dim_w - 1) * (stride_w - 1)
                conv_padding = [dilation[0] * (k_h - 1) - pad[0]] * 4

                # create new intermediate values
                inp_trans_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ifm_dim_h, ifm_dim_w, ifm_ch),  # NHWC
                )
                padding_pixel_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, padded_odim_h, padded_odim_w, ifm_ch),  # NHWC
                )
                graph.value_info.append(inp_trans_out)
                graph.value_info.append(padding_pixel_out)
                inp_trans_out = inp_trans_out.name
                padding_pixel_out = padding_pixel_out.name
                model.set_tensor_datatype(inp_trans_out, idt)
                model.set_tensor_datatype(padding_pixel_out, idt)

                need_im2col = True
                if all(p == 0 for p in conv_padding):
                    padding = 0

                # k_h=k_w==1: pointwise convolution, thus no im2col needed
                if k_h == 1 and k_w == 1 and padding == 0 and stride_h == 1 and stride_w == 1:
                    need_im2col = False

                if need_im2col:
                    im2col_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, ofm_dim_h, ofm_dim_w, ifm_ch * k_h * k_w),
                    )
                    graph.value_info.append(im2col_out)
                    im2col_out = im2col_out.name
                    model.set_tensor_datatype(im2col_out, idt)

                matmul_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ofm_dim_h, ofm_dim_w, ofm_ch),
                )
                graph.value_info.append(matmul_out)
                matmul_out = matmul_out.name
                model.set_tensor_datatype(matmul_out, odt)

                # create new nodes

                # NCHW -> NHWC
                inp_trans_node = helper.make_node(
                    "Transpose", [deconv_input], [inp_trans_out], perm=[0, 2, 3, 1]
                )
                # Pixel Padding
                fmpadding_pixel_node = helper.make_node(
                    "FMPadding_Pixel",
                    [inp_trans_out],
                    [padding_pixel_out],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ImgDim=(ifm_dim_h, ifm_dim_w),
                    Stride=[stride_h, stride_w],
                    NumChannels=ifm_ch,
                    inputDataType=str(idt.name),
                    numInputVectors=1,
                    SIMD=1,
                )
                # lower input tensor
                matmul_input = padding_pixel_out
                if need_im2col:
                    matmul_input = im2col_out
                    im2col_node = helper.make_node(
                        "Im2Col",
                        [padding_pixel_out],
                        [im2col_out],
                        domain="qonnx.custom_op.general",
                        stride=[1, 1],
                        kernel_size=[k_h, k_w],
                        pad_amount=conv_padding,
                        input_shape="(1,{},{},{})".format(padded_odim_h, padded_odim_w, ifm_ch),
                        depthwise=False,
                        dilations=dilation,
                    )

                # do matmul
                matmul_node = helper.make_node("MatMul", [matmul_input, weight_name], [matmul_out])
                # NHWC -> NCHW
                out_trans_node = helper.make_node(
                    "Transpose", [matmul_out], [deconv_output], perm=[0, 3, 1, 2]
                )
                # insert nodes where the conv is to preserve topological ordering
                graph.node.insert(node_ind, inp_trans_node)
                if need_im2col:
                    graph.node.insert(node_ind + 1, fmpadding_pixel_node)
                    graph.node.insert(node_ind + 2, im2col_node)
                    graph.node.insert(node_ind + 3, matmul_node)
                    graph.node.insert(node_ind + 4, out_trans_node)
                else:
                    graph.node.insert(node_ind + 1, fmpadding_pixel_node)
                    graph.node.insert(node_ind + 2, matmul_node)
                    graph.node.insert(node_ind + 3, out_trans_node)
                # remove old nodes
                graph.node.remove(n)

        return (model, graph_modified)
