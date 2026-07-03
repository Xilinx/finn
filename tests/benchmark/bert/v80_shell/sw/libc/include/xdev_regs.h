/*
 * This file is part of the Xilinx DMA IP Core driver for Linux
 *
 * Copyright (c) 2018-2022, Xilinx, Inc. All rights reserved.
 * Copyright (c) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#ifndef __XDEV_REGS_H__
#define __XDEV_REGS_H__

#ifdef __cplusplus
	extern "C" {
#endif

struct xreg_info {
	const char name[32];
	uint32_t addr;
	unsigned int repeat;
	unsigned int step;
	unsigned char shift;
	unsigned char len;
	unsigned char filler[2];
};

static struct xreg_info qdma_config_regs[] = {

	/* QDMA_TRQ_SEL_GLBL1 (0x00000) */
	{"CFG_BLOCK_ID",				0x00, 0, 0, 0, 0,},
	{"CFG_BUSDEV",					0x04, 0, 0, 0, 0,},
	{"CFG_PCIE_MAX_PL_SZ",				0x08, 0, 0, 0, 0,},
	{"CFG_PCIE_MAX_RDRQ_SZ",			0x0C, 0, 0, 0, 0,},
	{"CFG_SYS_ID",					0x10, 0, 0, 0, 0,},
	{"CFG_MSI_EN",					0x14, 0, 0, 0, 0,},
	{"CFG_PCIE_DATA_WIDTH",				0x18, 0, 0, 0, 0,},
	{"CFG_PCIE_CTRL",				0x1C, 0, 0, 0, 0,},
	{"CFG_AXI_USR_MAX_PL_SZ",			0x40, 0, 0, 0, 0,},
	{"CFG_AXI_USR_MAX_RDRQ_SZ",			0x44, 0, 0, 0, 0,},
	{"CFG_MISC_CTRL",				0x4C, 0, 0, 0, 0,},
	{"CFG_SCRATCH_REG",				0x80, 8, 0, 0, 0,},
	{"QDMA_RAM_SBE_MSK_A",				0xF0, 0, 0, 0, 0,},
	{"QDMA_RAM_SBE_STS_A",				0xF4, 0, 0, 0, 0,},
	{"QDMA_RAM_DBE_MSK_A",				0xF8, 0, 0, 0, 0,},
	{"QDMA_RAM_DBE_STS_A",				0xFC, 0, 0, 0, 0,},

	/* QDMA_TRQ_SEL_GLBL2 (0x00100) */
	{"GLBL2_ID",					0x100, 0, 0, 0, 0,},
	{"GLBL2_PF_BL_INT",				0x104, 0, 0, 0, 0,},
	{"GLBL2_PF_VF_BL_INT",				0x108, 0, 0, 0, 0,},
	{"GLBL2_PF_BL_EXT",				0x10C, 0, 0, 0, 0,},
	{"GLBL2_PF_VF_BL_EXT",				0x110, 0, 0, 0, 0,},
	{"GLBL2_CHNL_INST",				0x114, 0, 0, 0, 0,},
	{"GLBL2_CHNL_QDMA",				0x118, 0, 0, 0, 0,},
	{"GLBL2_CHNL_STRM",				0x11C, 0, 0, 0, 0,},
	{"GLBL2_QDMA_CAP",				0x120, 0, 0, 0, 0,},
	{"GLBL2_PASID_CAP",				0x128, 0, 0, 0, 0,},
	{"GLBL2_FUNC_RET",				0x12C, 0, 0, 0, 0,},
	{"GLBL2_SYS_ID",				0x130, 0, 0, 0, 0,},
	{"GLBL2_MISC_CAP",				0x134, 0, 0, 0, 0,},
	{"GLBL2_DBG_PCIE_RQ",				0x1B8, 2, 0, 0, 0,},
	{"GLBL2_DBG_AXIMM_WR",				0x1C0, 2, 0, 0, 0,},
	{"GLBL2_DBG_AXIMM_RD",				0x1C8, 2, 0, 0, 0,},

	/* QDMA_TRQ_SEL_GLBL (0x00200) */
	{"GLBL_RNGSZ",					0x204, 16, 0, 0, 0,},
	{"GLBL_ERR_STAT",				0x248, 0,  0, 0, 0,},
	{"GLBL_ERR_MASK",				0x24C, 0,  0, 0, 0,},
	{"GLBL_DSC_CFG",				0x250, 0,  0, 0, 0,},
	{"GLBL_DSC_ERR_STS",				0x254, 0,  0, 0, 0,},
	{"GLBL_DSC_ERR_MSK",				0x258, 0,  0, 0, 0,},
	{"GLBL_DSC_ERR_LOG",				0x25C, 2,  0, 0, 0,},
	{"GLBL_TRQ_ERR_STS",				0x264, 0,  0, 0, 0,},
	{"GLBL_TRQ_ERR_MSK",				0x268, 0,  0, 0, 0,},
	{"GLBL_TRQ_ERR_LOG",				0x26C, 0,  0, 0, 0,},
	{"GLBL_DSC_DBG_DAT",				0x270, 2,  0, 0, 0,},
	{"GLBL_DSC_ERR_LOG2",				0x27C, 0,  0, 0, 0,},
	{"GLBL_INTERRUPT_CFG",				0x288, 0,  0, 0, 0,},

	/* QDMA_TRQ_SEL_FMAP (0x00400 - 0x7FC) * TODO: max 256, display 4 for now */
	{"TRQ_SEL_FMAP",				0x400, 4, 0, 0, 0,},

	/* QDMA_TRQ_SEL_IND (0x00800) */
	{"IND_CTXT_DATA",				0x804, 8, 0, 0, 0,},
	{"IND_CTXT_MASK",				0x824, 8, 0, 0, 0,},
	{"IND_CTXT_CMD",				0x844, 0, 0, 0, 0,},

	/* QDMA_TRQ_SEL_C2H (0x00A00) */
	{"C2H_TIMER_CNT",				0xA00, 16, 0, 0, 0,},
	{"C2H_CNT_THRESH",				0xA40, 16, 0, 0, 0,},
	{"C2H_STAT_S_AXIS_C2H_ACCEPTED",		0xA88, 0, 0, 0, 0,},
	{"C2H_STAT_S_AXIS_CMPT_ACCEPTED",		0xA8C, 0, 0, 0, 0,},
	{"C2H_STAT_DESC_RSP_PKT_ACCEPTED",		0xA90, 0, 0, 0, 0,},
	{"C2H_STAT_AXIS_PKG_CMP",			0xA94, 0, 0, 0, 0,},
	{"C2H_STAT_DESC_RSP_ACCEPTED",			0xA98, 0, 0, 0, 0,},
	{"C2H_STAT_DESC_RSP_CMP",			0xA9C, 0, 0, 0, 0,},
	{"C2H_STAT_WRQ_OUT",				0xAA0, 0, 0, 0, 0,},
	{"C2H_STAT_WPL_REN_ACCEPTED",			0xAA4, 0, 0, 0, 0,},
	{"C2H_STAT_TOTAL_WRQ_LEN",			0xAA8, 0, 0, 0, 0,},
	{"C2H_STAT_TOTAL_WPL_LEN",			0xAAC, 0, 0, 0, 0,},
	{"C2H_BUF_SZ",					0xAB0, 16, 0, 0, 0,},
	{"C2H_ERR_STAT",				0xAF0, 0, 0, 0, 0,},
	{"C2H_ERR_MASK",				0xAF4, 0, 0, 0, 0,},
	{"C2H_FATAL_ERR_STAT",				0xAF8, 0, 0, 0, 0,},
	{"C2H_FATAL_ERR_MASK",				0xAFC, 0, 0, 0, 0,},
	{"C2H_FATAL_ERR_ENABLE",			0xB00, 0, 0, 0, 0,},
	{"GLBL_ERR_INT",				0xB04, 0, 0, 0, 0,},
	{"C2H_PFCH_CFG",				0xB08, 0, 0, 0, 0,},
	{"C2H_INT_TIMER_TICK",				0xB0C, 0, 0, 0, 0,},
	{"C2H_STAT_DESC_RSP_DROP_ACCEPTED",		0xB10, 0, 0, 0, 0,},
	{"C2H_STAT_DESC_RSP_ERR_ACCEPTED",		0xB14, 0, 0, 0, 0,},
	{"C2H_STAT_DESC_REQ",				0xB18, 0, 0, 0, 0,},
	{"C2H_STAT_DEBUG_DMA_ENG",			0xB1C, 4, 0, 0, 0,},
	{"C2H_DBG_PFCH_ERR_CTXT",			0xB2C, 0, 0, 0, 0,},
	{"C2H_FIRST_ERR_QID",				0xB30, 0, 0, 0, 0,},
	{"STAT_NUM_CMPT_IN",				0xB34, 0, 0, 0, 0,},
	{"STAT_NUM_CMPT_OUT",				0xB38, 0, 0, 0, 0,},
	{"STAT_NUM_CMPT_DRP",				0xB3C, 0, 0, 0, 0,},
	{"STAT_NUM_STAT_DESC_OUT",			0xB40, 0, 0, 0, 0,},
	{"STAT_NUM_DSC_CRDT_SENT",			0xB44, 0, 0, 0, 0,},
	{"STAT_NUM_FCH_DSC_RCVD",			0xB48, 0, 0, 0, 0,},
	{"STAT_NUM_BYP_DSC_RCVD",			0xB4C, 0, 0, 0, 0,},
	{"C2H_CMPT_COAL_CFG",				0xB50, 0, 0, 0, 0,},
	{"C2H_INTR_H2C_REQ",				0xB54, 0, 0, 0, 0,},
	{"C2H_INTR_C2H_MM_REQ",				0xB58, 0, 0, 0, 0,},
	{"C2H_INTR_ERR_INT_REQ",			0xB5C, 0, 0, 0, 0,},
	{"C2H_INTR_C2H_ST_REQ",				0xB60, 0, 0, 0, 0,},
	{"C2H_INTR_H2C_ERR_MM_MSIX_ACK",		0xB64, 0, 0, 0, 0,},
	{"C2H_INTR_H2C_ERR_MM_MSIX_FAIL",		0xB68, 0, 0, 0, 0,},
	{"C2H_INTR_H2C_ERR_MM_NO_MSIX",			0xB6C, 0, 0, 0, 0,},
	{"C2H_INTR_H2C_ERR_MM_CTXT_INVAL",		0xB70, 0, 0, 0, 0,},
	{"C2H_INTR_C2H_ST_MSIX_ACK",			0xB74, 0, 0, 0, 0,},
	{"C2H_INTR_C2H_ST_MSIX_FAIL",			0xB78, 0, 0, 0, 0,},
	{"C2H_INTR_C2H_ST_NO_MSIX",			0xB7C, 0, 0, 0, 0,},
	{"C2H_INTR_C2H_ST_CTXT_INVAL",			0xB80, 0, 0, 0, 0,},
	{"C2H_STAT_WR_CMP",				0xB84, 0, 0, 0, 0,},
	{"C2H_STAT_DEBUG_DMA_ENG_4",			0xB88, 0, 0, 0, 0,},
	{"C2H_STAT_DEBUG_DMA_ENG_5",			0xB8C, 0, 0, 0, 0,},
	{"C2H_DBG_PFCH_QID",				0xB90, 0, 0, 0, 0,},
	{"C2H_DBG_PFCH",				0xB94, 0, 0, 0, 0,},
	{"C2H_INT_DEBUG",				0xB98, 0, 0, 0, 0,},
	{"C2H_STAT_IMM_ACCEPTED",			0xB9C, 0, 0, 0, 0,},
	{"C2H_STAT_MARKER_ACCEPTED",			0xBA0, 0, 0, 0, 0,},
	{"C2H_STAT_DISABLE_CMP_ACCEPTED",		0xBA4, 0, 0, 0, 0,},
	{"C2H_C2H_PAYLOAD_FIFO_CRDT_CNT",		0xBA8, 0, 0, 0, 0,},

	{"C2H_INTR_DYN_REQ",				0xBAC, 0, 0, 0, 0,},
	{"C2H_INTR_DYN_MSIX",				0xBB0, 0, 0, 0, 0,},
	{"C2H_DROP_LEN_MISMATCH",			0xBB4, 0, 0, 0, 0,},
	{"C2H_DROP_DESC_RSP_LEN",			0xBB8, 0, 0, 0, 0,},
	{"C2H_DROP_QID_FIFO_LEN",			0xBBC, 0, 0, 0, 0,},
	{"C2H_DROP_PAYLOAD_CNT",			0xBC0, 0, 0, 0, 0,},
	{"QDMA_C2H_CMPT_FORMAT",			0xBC4, 7, 0, 0, 0,},
	{"C2H_PFCH_CACHE_DEPTH",			0xBE0, 0, 0, 0, 0,},
	{"C2H_CMPT_COAL_BUF_DEPTH",			0xBE4, 0, 0, 0, 0,},
	{"C2H_PFCH_CRDT",				0xBE8, 0, 0, 0, 0,},

	/* QDMA_TRQ_SEL_H2C(0x00E00) Register Space*/
	{"H2C_ERR_STAT",				0xE00, 0, 0, 0, 0,},
	{"H2C_ERR_MASK",				0xE04, 0, 0, 0, 0,},
	{"H2C_FIRST_ERR_QID",				0xE08, 0, 0, 0, 0,},
	{"H2C_DBG_REG",					0xE0C, 5, 0, 0, 0,},
	{"H2C_FATAL_ERR_EN",				0xE20, 0, 0, 0, 0,},
	{"H2C_REQ_THROT",				0xE24, 0, 0, 0, 0,},
	{"H2C_ALN_DBG_REG0",				0xE28, 0, 0, 0, 0,},

	/* QDMA_TRQ_SEL_C2H_MM (0x1000) */
	{"C2H_MM_CONTROL",				0x1004, 3, 0, 0, 0,},
	{"C2H_MM_STATUS",				0x1040, 2, 0, 0, 0,},
	{"C2H_MM_CMPL_DSC_CNT",				0x1048, 0, 0, 0, 0,},
	{"C2H_MM_ERR_CODE_EN_MASK",			0x1054, 0, 0, 0, 0,},
	{"C2H_MM_ERR_CODE",				0x1058, 0, 0, 0, 0,},
	{"C2H_MM_ERR_INFO",				0x105C, 0, 0, 0, 0,},
	{"C2H_MM_PERF_MON_CTRL",			0x10C0, 0, 0, 0, 0,},
	{"C2H_MM_PERF_MON_CY_CNT",			0x10C4, 2, 0, 0, 0,},
	{"C2H_MM_PERF_MON_DATA_CNT",			0x10CC, 2, 0, 0, 0,},
	{"C2H_MM_DBG_INFO",				0x10E8, 2, 0, 0, 0,},

	/* QDMA_TRQ_SEL_H2C_MM (0x1200)*/
	{"H2C_MM_CONTROL",				0x1204, 3, 0, 0, 0,},
	{"H2C_MM_STATUS",				0x1240, 0, 0, 0, 0,},
	{"H2C_MM_CMPL_DSC_CNT",				0x1248, 0, 0, 0, 0,},
	{"H2C_MM_ERR_CODE_EN_MASK",			0x1254, 0, 0, 0, 0,},
	{"H2C_MM_ERR_CODE",				0x1258, 0, 0, 0, 0,},
	{"H2C_MM_ERR_INFO",				0x125C, 0, 0, 0, 0,},
	{"H2C_MM_PERF_MON_CTRL",			0x12C0, 0, 0, 0, 0,},
	{"H2C_MM_PERF_MON_CY_CNT",			0x12C4, 2, 0, 0, 0,},
	{"H2C_MM_PERF_MON_DATA_CNT",			0x12CC, 2, 0, 0, 0,},
	{"H2C_MM_DBG_INFO",				0x12E8, 0, 0, 0, 0,},
	{"H2C_MM_REQ_THROT",				0x12EC, 0, 0, 0, 0,},

	/* QDMA_PF_MAILBOX (0x2400) */
	{"FUNC_STATUS",					0x2400, 0, 0, 0, 0,},
	{"FUNC_CMD",					0x2404, 0, 0, 0, 0,},
	{"FUNC_INTR_VEC",				0x2408, 0, 0, 0, 0,},
	{"TARGET_FUNC",					0x240C, 0, 0, 0, 0,},
	{"INTR_CTRL",					0x2410, 0, 0, 0, 0,},
	{"PF_ACK",					0x2420, 8, 0, 0, 0,},
	{"FLR_CTRL_STATUS",				0x2500, 0, 0, 0, 0,},
	{"MSG_IN",					0x2800, 32, 0, 0, 0,},
	{"MSG_OUT",					0x2C00, 32, 0, 0, 0,},

	{"", 0, 0, 0 }
};

static struct xreg_info qdma_user_regs[] = {
	{"ST_C2H_QID",					0x0, 0, 0, 0, 0,},
	{"ST_C2H_PKTLEN",				0x4, 0, 0, 0, 0,},
	{"ST_C2H_CONTROL",				0x8, 0, 0, 0, 0,},
	/*  ST_C2H_CONTROL:
	 *	[1] : start C2H
	 *	[2] : immediate data
	 *	[3] : every packet statrs with 00 instead of continuous data
	 *	      stream until # of packets is complete
	 *	[31]: gen_user_reset_n
	 */
	{"ST_H2C_CONTROL",				0xC, 0, 0, 0, 0,},
	/*  ST_H2C_CONTROL:
	 *	[0] : clear match for H2C transfer
	 */
	{"ST_H2C_STATUS",				0x10, 0, 0, 0, 0,},
	{"ST_H2C_XFER_CNT",				0x14, 0, 0, 0, 0,},
	{"ST_C2H_PKT_CNT",				0x20, 0, 0, 0, 0,},
	{"ST_C2H_CMPT_DATA",				0x30, 8, 0, 0, 0,},
	{"ST_C2H_CMPT_SIZE",				0x50, 0, 0, 0, 0,},
	{"ST_SCRATCH_REG",				0x60, 2, 0, 0, 0,},
	{"ST_C2H_PKT_DROP",				0x88, 0, 0, 0, 0,},
	{"ST_C2H_PKT_ACCEPT",				0x8C, 0, 0, 0, 0,},
	{"DSC_BYPASS_LOOP",				0x90, 0, 0, 0, 0,},
	{"USER_INTERRUPT",				0x94, 0, 0, 0, 0,},
	{"USER_INTERRUPT_MASK",				0x98, 0, 0, 0, 0,},
	{"USER_INTERRUPT_VEC",				0x9C, 0, 0, 0, 0,},
	{"DMA_CONTROL",					0xA0, 0, 0, 0, 0,},
	{"VDM_MSG_READ",				0xA4, 0, 0, 0, 0,},

	{"", 0, 0, 0 }
};

#ifdef __cplusplus
	}
#endif

#endif
