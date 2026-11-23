import numpy as np
import warnings
from onnx import TensorProto, helper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import auto_pad_to_explicit_padding, get_by_name


class InferPixelPaddingDeconv(Transformation):
    """
    Lowering and conversion of ConvTranspose (NCHW) nodes to
    FMPadding_Pixel + Im2Col + MatMul (NHWC) surrounded by Transpose nodes.
    If the ConvTranspose pads exceed the dilated kernel extent (pads > dilation * (k - 1)),
    the equivalent convolution crops its input, which is done by a Crop node between
    FMPadding_Pixel and Im2Col.
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
                    pad = [pad[0]] * 4
                output_padding_attr = get_by_name(n.attribute, "output_padding")
                if output_padding_attr is not None:
                    output_padding = output_padding_attr.ints
                else:
                    output_padding = [0, 0]
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
                # The ConvTranspose equals a stride-1 Conv of the pixel-padded input with
                # the rotated weights and the padding [H_begin, W_begin, H_end, W_end]
                # below. ConvTranspose pads larger than the dilated kernel extent make it
                # negative, i.e. the Conv crops its input instead of padding it.
                conv_padding = [
                    dilation[0] * (k_h - 1) - pad[0],
                    dilation[1] * (k_w - 1) - pad[1],
                    dilation[0] * (k_h - 1) - pad[2] + output_padding[0],
                    dilation[1] * (k_w - 1) - pad[3] + output_padding[1],
                ]
                crop = [max(0, -p) for p in conv_padding]
                conv_padding = [max(0, p) for p in conv_padding]
                need_crop = any(c > 0 for c in crop)
                cropped_odim_h = padded_odim_h - crop[0] - crop[2]
                cropped_odim_w = padded_odim_w - crop[1] - crop[3]

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

                if need_crop:
                    crop_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, cropped_odim_h, cropped_odim_w, ifm_ch),  # NHWC
                    )
                    graph.value_info.append(crop_out)
                    crop_out = crop_out.name
                    model.set_tensor_datatype(crop_out, idt)

                need_im2col = True
                padding = any(p != 0 for p in conv_padding)

                # k_h=k_w==1: pointwise convolution, thus no im2col needed
                if k_h == 1 and k_w == 1 and not padding and stride_h == 1 and stride_w == 1:
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
                new_nodes = [inp_trans_node, fmpadding_pixel_node]
                conv_input = padding_pixel_out
                if need_crop:
                    crop_node = helper.make_node(
                        "Crop",
                        [padding_pixel_out],
                        [crop_out],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        DataType=idt.name,
                        ImgDim=[padded_odim_h, padded_odim_w],
                        NumChannels=ifm_ch,
                        CropNorth=crop[0],
                        CropWest=crop[1],
                        CropSouth=crop[2],
                        CropEast=crop[3],
                        SIMD=1,
                        numInputVectors=[1],
                    )
                    new_nodes.append(crop_node)
                    conv_input = crop_out
                # lower input tensor
                matmul_input = conv_input
                if need_im2col:
                    matmul_input = im2col_out
                    im2col_node = helper.make_node(
                        "Im2Col",
                        [conv_input],
                        [im2col_out],
                        domain="qonnx.custom_op.general",
                        stride=[1, 1],
                        kernel_size=[k_h, k_w],
                        pad_amount=conv_padding,
                        input_shape="(1,{},{},{})".format(cropped_odim_h, cropped_odim_w, ifm_ch),
                        depthwise=False,
                        dilations=dilation,
                    )
                    new_nodes.append(im2col_node)

                # do matmul
                matmul_node = helper.make_node("MatMul", [matmul_input, weight_name], [matmul_out])
                # NHWC -> NCHW
                out_trans_node = helper.make_node(
                    "Transpose", [matmul_out], [deconv_output], perm=[0, 3, 1, 2]
                )
                new_nodes += [matmul_node, out_trans_node]
                # insert nodes where the conv is to preserve topological ordering
                for i, new_node in enumerate(new_nodes):
                    graph.node.insert(node_ind + i, new_node)
                # remove old nodes
                graph.node.remove(n)

        return (model, graph_modified)
