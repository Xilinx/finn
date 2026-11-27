# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Implementation constraints of the module thresholding, read for each of its
# instances (SCOPED_TO_REF thresholding, USED_IN implementation; see
# HWCustomOp.get_xdc_files).
#
# The search pointers of the pipeline stages (genStages[*].genPE[*].P_reg[ptr],
# with DEEP_PIPELINE also Pf_reg[ptr]) are identical across the PEs in their
# channel fold bits. Synthesis merges them into one register per stage, which
# drives the stage memory addresses of all PEs (LUTRAM address and MUXF7/F8 select
# pins, up to some 1500). Replicating it per clock region during placement keeps
# these nets local. The constraint goes onto the parent nets, which carry the
# whole fanout.
#
# No -quiet: Thresholding_rtl adds this file only in the configurations that have
# such registers (numSteps > 1 or deep_pipeline), so that a miss, e.g. after a
# change of thresholding.sv, shows up as a warning.
set_property MAX_FANOUT_MODE CLOCK_REGION [get_nets -parent_net [get_nets -of_objects [get_pins -filter {REF_PIN_NAME == Q} -of_objects [get_cells -hierarchical -filter {NAME =~ *genStages[*].genPE[*].P*_reg[ptr]*}]]]]
