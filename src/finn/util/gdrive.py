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

import gspread
import os
import warnings
from datetime import datetime

from finn.util.basic import get_finn_root


def upload_to_end2end_dashboard(data_dict):
    gdrive_key = get_finn_root() + "/gdrive-key/service_account.json"
    if not os.path.isfile(gdrive_key):
        warnings.warn("Google Drive key not found, skipping dashboard upload")
        return
    gc = gspread.service_account(filename=gdrive_key)
    spreadsheet = gc.open("finn-end2end-dashboard")
    worksheet = spreadsheet.get_worksheet(0)
    keys = list(data_dict.keys())
    vals = list(data_dict.values())
    # check against existing header
    existing_keys = worksheet.row_values(1)
    if not set(existing_keys).issuperset(set(keys)):
        # create new worksheet
        dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet = spreadsheet.add_worksheet(
            title="Dashboard " + dtstr, rows=10, cols=len(keys), index=0
        )
        # create header row with keys
        worksheet.update("A1:1", [keys])
        # freeze and make header bold
        worksheet.freeze(rows=1)
        worksheet.format("A1:1", {"textFormat": {"bold": True}})
    # insert values into new row at appropriate positions
    worksheet.insert_row([], index=2)
    for i in range(len(keys)):
        colind = existing_keys.index(keys[i])
        col_letter = chr(ord("A") + colind)
        worksheet.update("%s2" % col_letter, vals[i])
