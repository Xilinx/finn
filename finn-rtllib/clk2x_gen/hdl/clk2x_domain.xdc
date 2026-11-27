# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Implementation constraints of the module clk2x_domain, read for each of its
# instances (SCOPED_TO_REF clk2x_domain, USED_IN implementation; see
# CreateStitchedIP).
#
# The 1x and 2x clock nets form one CLOCK_DELAY_GROUP, so that the router
# balances their clock trees. Otherwise, their skew easily takes a large part of
# the 2x period from the synchronous crossings between both domains. A group
# balances the clocks of one clock source: it is named after the instance
# (current_instance), so that every generator gets its own.
set_property CLOCK_DELAY_GROUP [current_instance .] [get_nets -of_objects [get_pins {clk_buf/O clk2x_buf/O}]]
