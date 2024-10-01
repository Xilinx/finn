# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of pyxsi nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import pyxsi_utils
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

# since simulation with XSI requires a certain LD_LIBRARY_PATH setting
# which breaks other applications, we launch the simulation in its
# own executable with this env.var. setting, and use xmlrpc to access it

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
