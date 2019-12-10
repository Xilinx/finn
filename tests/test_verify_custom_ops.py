from onnx import helper
from finn.custom_op.verify_custom_op_construct import CustomOp_Construct

def test_verify_layout_custom_ops():
    # MultiThreshold
    m_node = helper.make_node(
        "MultiThreshold",
        ["v", "thresholds"],
        ["out"],
        domain="finn",
        out_scale=2.0,
        out_bias=-1.0,
        out_dtype="",
    ) 

    inst = CustomOp_Construct[m_node.op_type] 
    inst.verify_construct(m_node) 

    # XnorPopcountMatMul
    xnor_node = helper.make_node(
        "XnorPopcountMatMul", ["x", "W"], ["out"], domain="finn"
    )

    inst = CustomOp_Construct[xnor_node.op_type]
    inst.verify_construct(xnor_node)

    # StreamingMaxPool_Batch
    MaxPool_batch_node = helper.make_node(
        "StreamingMaxPool_Batch",
        ["in"],
        ["out"],
        domain="finn",
        backend="fpgadataflow",
        code_gen_dir="",
        executable_path="",
        ImgDim=4,
        PoolDim=2,
        NumChannels=2,
    )

    inst = CustomOp_Construct[MaxPool_batch_node.op_type]
    inst.verify_construct(MaxPool_batch_node)
    
    
    
    # StreamingFCLayer_Batch - no activation
    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["in", "weights"],
        ["out"],
        domain="finn",
        backend="fpgadataflow",
        code_gen_dir="",
        executable_path="",
        resType="ap_resource_lut()",
        MW=8,
        MH=8,
        SIMD=4,
        PE=4,
        inputDataType="<FINN DataType>",
        weightDataType="<FINN DataType>",
        outputDataType="<FINN DataType>",
        ActVal=0,
        binaryXnorMode=1,
        noActivation=1,
    )
    
    inst = CustomOp_Construct[FCLayer_node.op_type]
    inst.verify_construct(FCLayer_node)

    # StreamingFCLayer_Batch - with activation
    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["in", "weights", "threshs"],
        ["out"],
        domain="finn",
        backend="fpgadataflow",
        code_gen_dir="",
        executable_path="",
        resType="ap_resource_lut()",
        MW=8,
        MH=8,
        SIMD=4,
        PE=4,
        inputDataType="<FINN DataType>",
        weightDataType="<FINN DataType>",
        outputDataType="<FINN DataType>",
        ActVal=0,
        binaryXnorMode=1,
        noActivation=0
    )
    inst = CustomOp_Construct[FCLayer_node.op_type]
    inst.verify_construct(FCLayer_node)

