from finn.core.utils import pack_innermost_dim_as_hex_string


def numpy_to_hls_code(ndarray, dtype, pack_innermost_dim=True):
    hls_dtype = dtype.get_hls_datatype_str()
    if pack_innermost_dim:
        idimlen = ndarray.shape[-1]
        ndarray = pack_innermost_dim_as_hex_string(ndarray, dtype, idimlen)
        hls_dtype = "ap_uint<%d>" % (idimlen * dtype.bitwidth())
    return hls_dtype
