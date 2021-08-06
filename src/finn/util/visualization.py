# Copyright (c) 2020, Xilinx
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
# * Neither the name of FINN nor the names of its
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

import inspect
import netron
import os
from IPython.display import IFrame


def showSrc(what):
    print("".join(inspect.getsourcelines(what)[0]))


def showInNetron(model_filename: str, localhost_url: str = None, port: int = None):
    """Shows a ONNX model file in the Jupyter Notebook using Netron.

    :param model_filename: The path to the ONNX model file.
    :type model_filename: str

    :param localhost_url: The IP address used by the Jupyter IFrame to show the model.
     Defaults to localhost.
    :type localhost_url: str, optional

    :param port: The port number used by Netron and the Jupyter IFrame to show
     the ONNX model.  Defaults to 8081.
    :type port: int, optional

    :return: The IFrame displaying the ONNX model.
    :rtype: IPython.lib.display.IFrame
    """
    if port is None:
        try:
            port = int(os.environ["NETRON_PORT"])
        except KeyError:
            port = 8081
    if localhost_url is None:
        localhost_url = os.getenv("LOCALHOST_URL", default="localhost")
    netron.start(model_filename, address=("0.0.0.0", port), browse=False)
    return IFrame(src=f"http://{localhost_url}:{port}/", width="100%", height=400)
