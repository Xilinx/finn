import os
import pyxsi_utils
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

try:
    ldlp = os.environ["LD_LIBRARY_PATH"]
    if not ("Vivado" in ldlp):
        assert False, "Must be launched with LD_LIBRARY_PATH=$(XILINX_VIVADO)/lib/lnx64.o"
except KeyError:
    assert False, "Must be launched with LD_LIBRARY_PATH=$(XILINX_VIVADO)/lib/lnx64.o"


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2",)


# Create server
port = 8000
with SimpleXMLRPCServer(("localhost", port), requestHandler=RequestHandler) as server:
    server.register_introspection_functions()
    server.register_instance(pyxsi_utils)
    print(f"pyxsi RPC server is now running at {port}")
    server.serve_forever()
