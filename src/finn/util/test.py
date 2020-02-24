import torch
from models.CNV import CNV
from models.LFC import LFC
from models.SFC import SFC
from models.TFC import TFC


def get_trained_checkpoint(netname, wbits, abits):
    """Returns the weights and activations from the FINN Brevitas test networks
    for given netname and the number of bits for weights and activations"""
    # TODO get from config instead, hardcoded to Docker path for now
    nname = "%s_%dW%dA" % (netname, wbits, abits)
    root = "/workspace/brevitas_cnv_lfc/pretrained_models/%s/checkpoints/best.tar"
    return root % nname


def get_test_model_def_fxn(netname):
    """Returns the PyTorch model instantation function related to netname."""
    model_def_map = {"LFC": LFC, "SFC": SFC, "TFC": TFC, "CNV": CNV}
    return model_def_map[netname]


def get_test_model_trained(netname, wbits, abits):
    """Returns the pretrained model specified by input arguments loaded with weights 
    and activations from the FINN Brevitas test networks."""
    model_def_fxn = get_test_model_def_fxn(netname)
    checkpoint_loc = get_trained_checkpoint(netname, wbits, abits)
    fc = model_def_fxn(weight_bit_width=wbits, act_bit_width=abits, in_bit_width=abits)
    checkpoint = torch.load(checkpoint_loc, map_location="cpu")
    fc.load_state_dict(checkpoint["state_dict"])
    return fc.eval()


def get_test_model_untrained(netname, wbits, abits):
    """Returns untrained model specified by input arguments."""
    model_def_fxn = get_test_model_def_fxn(netname)
    fc = model_def_fxn(weight_bit_width=wbits, act_bit_width=abits, in_bit_width=abits)
    return fc.eval()
