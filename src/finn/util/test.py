import torch
from models.LFC import LFC
from models.SFC import SFC
from models.TFC import TFC


def get_trained_fc_checkpoint(size, wbits, abits):
    # TODO get from config instead, hardcoded to Docker path for now
    nname = "%s_%dW%dA" % (size, wbits, abits)
    root = "/workspace/brevitas_cnv_lfc/pretrained_models/%s/checkpoints/best.tar"
    return root % nname


def get_fc_model_def_fxn(size):
    model_def_map = {"LFC": LFC, "SFC": SFC, "TFC": TFC}
    return model_def_map[size]


def get_fc_model_trained(size, wbits, abits):
    model_def_fxn = get_fc_model_def_fxn(size)
    checkpoint_loc = get_trained_fc_checkpoint(size, wbits, abits)
    fc = model_def_fxn(weight_bit_width=wbits, act_bit_width=abits, in_bit_width=abits)
    checkpoint = torch.load(checkpoint_loc, map_location="cpu")
    fc.load_state_dict(checkpoint["state_dict"])
    return fc


def get_fc_model_untrained(size, wbits, abits):
    model_def_fxn = get_fc_model_def_fxn(size)
    fc = model_def_fxn(weight_bit_width=wbits, act_bit_width=abits, in_bit_width=abits)
    return fc
