import os
import xml.etree.ElementTree as ET

import finn.core.utils as util
import finn.custom_op.registry as registry


def hls_synth_res_estimation(model):
    """Extracts the results from the vivado synthesis.
    Returns {node name : resource estimation}"""

    res_dict = {}
    for node in model.graph.node:
        if node.domain == "finn":
            backend_attribute = util.get_by_name(node.attribute, "backend")
            if backend_attribute is None:
                continue
            backend_value = backend_attribute.s.decode("UTF-8")
            if backend_value == "fpgadataflow":
                op_type = node.op_type
                inst = registry.custom_op[op_type](node)
                code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
                if code_gen_dir == "":
                    raise Exception(
                        """Please run "CodeGen_ipgen" transformation and
                            "HLSSynth_IPGen" first to generate the report files"""
                    )
                else:
                    xmlfile = "{}/project_{}/sol1/syn/report/{}_csynth.xml".format(
                        code_gen_dir, node.name, node.name
                    )

                    if os.path.isfile(xmlfile):
                        res_dict[node.name] = []
                        tree = ET.parse(xmlfile)
                        root = tree.getroot()
                        for item in root.findall("AreaEstimates/Resources"):
                            for child in item:
                                res_dict[node.name].append(
                                    ["{} : {}".format(child.tag, child.text)]
                                )
                    else:
                        raise Exception(
                            """Please run "HLSSynth_IPGen" first
                                to generate the report files"""
                        )

    return res_dict
