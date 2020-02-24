import torch
from models.CNV import CNV
from models.LFC import LFC
from models.SFC import SFC
from models.TFC import TFC


def get_trained_checkpoint(netname, wbits, abits):
    # TODO get from config instead, hardcoded to Docker path for now
    nname = "%s_%dW%dA" % (netname, wbits, abits)
    root = "/workspace/brevitas_cnv_lfc/pretrained_models/%s/checkpoints/best.tar"
    return root % nname


def get_test_model_def_fxn(netname):
    model_def_map = {"LFC": LFC, "SFC": SFC, "TFC": TFC, "CNV": CNV}
    return model_def_map[netname]


def get_test_model_trained(netname, wbits, abits):
    model_def_fxn = get_test_model_def_fxn(netname)
    checkpoint_loc = get_trained_checkpoint(netname, wbits, abits)
    fc = model_def_fxn(weight_bit_width=wbits, act_bit_width=abits, in_bit_width=abits)
    checkpoint = torch.load(checkpoint_loc, map_location="cpu")
    fc.load_state_dict(checkpoint["state_dict"])
    return fc.eval()


def get_test_model_untrained(netname, wbits, abits):
    model_def_fxn = get_test_model_def_fxn(netname)
    fc = model_def_fxn(weight_bit_width=wbits, act_bit_width=abits, in_bit_width=abits)
    return fc.eval()
