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

#ifndef QDMA_NL_H__
#define QDMA_NL_H__

#ifdef __cplusplus
	extern "C" {
#endif

/**
 * @file
 * @brief This file contains the declarations for qdma netlink interfaces
 *
 ** physical function name (no more than 15 characters) */
#define XNL_NAME_PF		"xnl_pf"
/** virtual function name */
#define XNL_NAME_VF		"xnl_vf"
/** qdma netlink interface version number */
#define XNL_VERSION		0x1

/** qdma nl interface minimum response buffer length*/
#define XNL_RESP_BUFLEN_MIN	 256
/** qdma nl interface maximum response buffer length*/
#define XNL_RESP_BUFLEN_MAX	 (2048 * 10)
/** qdma nl interface error buffer length*/
#define XNL_ERR_BUFLEN		 64
/** qdma nl command parameter length*/
#define XNL_STR_LEN_MAX		 20

/** Q parameter: value to indicate invalid qid*/
#define XNL_QIDX_INVALID	0xFFFF
/** Q parameter: streaming mode*/
#define XNL_F_QMODE_ST	        0x00000001
/** Q parameter: memory management mode*/
#define XNL_F_QMODE_MM	        0x00000002
/** Q parameter: queue in h2c direction*/
#define XNL_F_QDIR_H2C	        0x00000004
/** Q parameter: queue in c2h direction*/
#define XNL_F_QDIR_C2H	        0x00000008
/** Q parameter: queue in both directions*/
#define XNL_F_QDIR_BOTH         (XNL_F_QDIR_H2C | XNL_F_QDIR_C2H)
/** Q parameter: queue in prefetch mode*/
#define XNL_F_PFETCH_EN         0x00000010
/** Q parameter: enable the bypass for the queue*/
#define XNL_F_DESC_BYPASS_EN	0x00000020
/** Q parameter: fetch credits*/
#define XNL_F_FETCH_CREDIT      0x00000040
/** Q parameter: enable writeback accumulation*/
#define XNL_F_CMPL_STATUS_ACC_EN        0x00000080
/** Q parameter: enable writeback*/
#define XNL_F_CMPL_STATUS_EN            0x00000100
/** Q parameter: enable writeback pending check*/
#define XNL_F_CMPL_STATUS_PEND_CHK      0x00000200
/** Q parameter: enable writeback status descriptor*/
#define XNL_F_CMPL_STATUS_DESC_EN  0x00000400
/** Q parameter: enable queue completion interrupt*/
#define XNL_F_C2H_CMPL_INTR_EN  0x00000800
/** Q parameter: enable user defined data*/
#define XNL_F_CMPL_UDD_EN       0x00001000
/** Q parameter: enable the pfetch bypass for the queue */
#define XNL_F_PFETCH_BYPASS_EN  0x00002000
/** Q parameter: disable CMPT overflow check */
#define XNL_F_CMPT_OVF_CHK_DIS	0x00004000
/** Q parameter: Completion Queue? */
#define XNL_F_Q_CMPL         0x00008000

/** maximum number of queue flags to control queue configuration*/
#define MAX_QFLAGS 17

/** maximum number of interrupt ring entries*/
#define QDMA_MAX_INT_RING_ENTRIES 512

/**
 * xnl_attr_t netlink attributes for qdma(variables):
 * the index in this enum is used as a reference for the type,
 * userspace application has to indicate the corresponding type
 * the policy is used for security considerations
 */
enum xnl_attr_t {
	XNL_ATTR_GENMSG,		/**< generatl message */
	XNL_ATTR_DRV_INFO,		/**< device info */

	XNL_ATTR_DEV_IDX,		/**< device index */
	XNL_ATTR_PCI_BUS,		/**< pci bus number */
	XNL_ATTR_PCI_DEV,		/**< pci device number */
	XNL_ATTR_PCI_FUNC,		/**< pci function id */

	XNL_ATTR_DEV_STAT_MMH2C_PKTS1,	/**< number of MM H2C packets */
	XNL_ATTR_DEV_STAT_MMH2C_PKTS2,	/**< number of MM H2C packets */
	XNL_ATTR_DEV_STAT_MMC2H_PKTS1,	/**< number of MM C2H packets */
	XNL_ATTR_DEV_STAT_MMC2H_PKTS2,	/**< number of MM C2H packets */
	XNL_ATTR_DEV_STAT_STH2C_PKTS1,	/**< number of ST H2C packets */
	XNL_ATTR_DEV_STAT_STH2C_PKTS2,	/**< number of ST H2C packets */
	XNL_ATTR_DEV_STAT_STC2H_PKTS1,	/**< number of ST C2H packets */
	XNL_ATTR_DEV_STAT_STC2H_PKTS2,	/**< number of ST C2H packets */

	XNL_ATTR_DEV_CFG_BAR,		/**< device config bar number */
	XNL_ATTR_DEV_USR_BAR,		/**< device AXI Master Lite(user bar) number */
	XNL_ATTR_DEV_QSET_MAX,		/**< max queue sets */
	XNL_ATTR_DEV_QSET_QBASE,	/**< queue base start */

	XNL_ATTR_VERSION_INFO,		/**< version info */
	XNL_ATTR_DEVICE_TYPE,		/**< device type */
	XNL_ATTR_IP_TYPE,		/**< ip type */
	XNL_ATTR_DEV_NUMQS,		/**< num of queues */
	XNL_ATTR_DEV_NUM_PFS,		/**< num of PFs */
	XNL_ATTR_DEV_MM_CHANNEL_MAX,	/**< mm channels */
	XNL_ATTR_DEV_MAILBOX_ENABLE,	/**< mailbox enable */
	XNL_ATTR_DEV_FLR_PRESENT,	/**< flr present */
	XNL_ATTR_DEV_ST_ENABLE,		/**< device st capability */
	XNL_ATTR_DEV_MM_ENABLE,		/**< device mm capability */
	XNL_ATTR_DEV_MM_CMPT_ENABLE,	/**< device mm cmpt capability */

	XNL_ATTR_REG_BAR_NUM,		/**< register bar number */
	XNL_ATTR_REG_ADDR,		/**< register address */
	XNL_ATTR_REG_VAL,		/**< register value */

	XNL_ATTR_CSR_INDEX,		/**< csr index */
	XNL_ATTR_CSR_COUNT,		/**< csr count */

	XNL_ATTR_QIDX,			/**< queue index */
	XNL_ATTR_NUM_Q,			/**< number of queues */
	XNL_ATTR_QFLAG,			/**< queue config flags */

	XNL_ATTR_CMPT_DESC_SIZE,	/**< completion descriptor size */
	XNL_ATTR_SW_DESC_SIZE,		/**< software descriptor size */
	XNL_ATTR_QRNGSZ_IDX,		/**< queue ring index */
	XNL_ATTR_C2H_BUFSZ_IDX,		/**< c2h buffer idex */
	XNL_ATTR_CMPT_TIMER_IDX,	/**< completion timer index */
	XNL_ATTR_CMPT_CNTR_IDX,		/**< completion counter index */
	XNL_ATTR_CMPT_TRIG_MODE,	/**< completion trigger mode */
	XNL_ATTR_MM_CHANNEL,		/**< mm channel */
	XNL_ATTR_CMPT_ENTRIES_CNT,      /**< completion entries count */

	XNL_ATTR_RANGE_START,		/**< range start */
	XNL_ATTR_RANGE_END,		/**< range end */

	XNL_ATTR_INTR_VECTOR_IDX,	/**< interrupt vector index */
	XNL_ATTR_INTR_VECTOR_START_IDX, /**< interrupt vector start index */
	XNL_ATTR_INTR_VECTOR_END_IDX,	/**< interrupt vector end index */
	XNL_ATTR_RSP_BUF_LEN,		/**< response buffer length */
	XNL_ATTR_GLOBAL_CSR,		/**< global csr data */
	XNL_ATTR_PIPE_GL_MAX,		/**< max no. of gl for pipe */
	XNL_ATTR_PIPE_FLOW_ID,          /**< pipe flow id */
	XNL_ATTR_PIPE_SLR_ID,           /**< pipe slr id */
	XNL_ATTR_PIPE_TDEST,            /**< pipe tdest */
	XNL_ATTR_DEV_STM_BAR,		/**< device STM bar number */
	XNL_ATTR_Q_STATE,
	XNL_ATTR_ERROR,
	XNL_ATTR_PING_PONG_EN,
	XNL_ATTR_APERTURE_SZ,
	XNL_ATTR_DEV_STAT_PING_PONG_LATMIN1,
	XNL_ATTR_DEV_STAT_PING_PONG_LATMIN2,
	XNL_ATTR_DEV_STAT_PING_PONG_LATMAX1,
	XNL_ATTR_DEV_STAT_PING_PONG_LATMAX2,
	XNL_ATTR_DEV_STAT_PING_PONG_LATAVG1,
	XNL_ATTR_DEV_STAT_PING_PONG_LATAVG2,
	XNL_ATTR_DEV,
	XNL_ATTR_DEBUG_EN,	/** Debug Regs Capability*/
	XNL_ATTR_DESC_ENGINE_MODE, /** Descriptor Engine Capability */
#ifdef ERR_DEBUG
	XNL_ATTR_QPARAM_ERR_INFO,	/**< queue param info */
#endif
	XNL_ATTR_NUM_REGS,			/**< number of regs */
	XNL_ATTR_MAX,
};

/**
 * xnl_st_c2h_cmpt_desc_size
 * c2h writeback descriptor sizes
 */
enum xnl_st_c2h_cmpt_desc_size {
	XNL_ST_C2H_CMPT_DESC_SIZE_8B,	/**< 8B descriptor */
	XNL_ST_C2H_CMPT_DESC_SIZE_16B,	/**< 16B descriptor */
	XNL_ST_C2H_CMPT_DESC_SIZE_32B,	/**< 32B descriptor */
	XNL_ST_C2H_CMPT_DESC_SIZE_64B,	/**< 64B descriptor */
	XNL_ST_C2H_NUM_CMPT_DESC_SIZES	/**< Num of desc sizes */
};

enum xnl_qdma_rngsz_idx {
	XNL_QDMA_RNGSZ_2048_IDX,
	XNL_QDMA_RNGSZ_64_IDX,
	XNL_QDMA_RNGSZ_128_IDX,
	XNL_QDMA_RNGSZ_192_IDX,
	XNL_QDMA_RNGSZ_256_IDX,
	XNL_QDMA_RNGSZ_384_IDX,
	XNL_QDMA_RNGSZ_512_IDX,
	XNL_QDMA_RNGSZ_768_IDX,
	XNL_QDMA_RNGSZ_1024_IDX,
	XNL_QDMA_RNGSZ_1536_IDX,
	XNL_QDMA_RNGSZ_3072_IDX,
	XNL_QDMA_RNGSZ_4096_IDX,
	XNL_QDMA_RNGSZ_6144_IDX,
	XNL_QDMA_RNGSZ_8192_IDX,
	XNL_QDMA_RNGSZ_12288_IDX,
	XNL_QDMA_RNGSZ_16384_IDX,
	XNL_QDMA_RNGSZ_IDXS
};


static const char *xnl_attr_str[XNL_ATTR_MAX + 1] = {
	"GENMSG",		        /**< XNL_ATTR_GENMSG */
	"DRV_INFO",		        /**< XNL_ATTR_DRV_INFO */
	"DEV_IDX",		        /**< XNL_ATTR_DEV_IDX */
	"DEV_PCIBUS",			/**< XNL_ATTR_PCI_BUS */
	"DEV_PCIDEV",			/**< XNL_ATTR_PCI_DEV */
	"DEV_PCIFUNC",			/**< XNL_ATTR_PCI_FUNC */
	"DEV_STAT_MMH2C_PKTS1",		/**< number of MM H2C packkts */
	"DEV_STAT_MMH2C_PKTS2",		/**< number of MM H2C packkts */
	"DEV_STAT_MMC2H_PKTS1",		/**< number of MM C2H packkts */
	"DEV_STAT_MMC2H_PKTS2",		/**< number of MM C2H packkts */
	"DEV_STAT_STH2C_PKTS1",		/**< number of ST H2C packkts */
	"DEV_STAT_STH2C_PKTS2",		/**< number of ST H2C packkts */
	"DEV_STAT_STC2H_PKTS1",		/**< number of ST C2H packkts */
	"DEV_STAT_STC2H_PKTS2",		/**< number of ST C2H packkts */
	"DEV_CFG_BAR",			/**< XNL_ATTR_DEV_CFG_BAR */
	"DEV_USR_BAR",			/**< XNL_ATTR_DEV_USER_BAR */
	"DEV_QSETMAX",			/**< XNL_ATTR_DEV_QSET_MAX */
	"DEV_QBASE",			/**< XNL_ATTR_DEV_QSET_QBASE */
	"VERSION_INFO",			/**< XNL_ATTR_VERSION_INFO */
	"DEVICE_TYPE",			/**< XNL_ATTR_DEVICE_TYPE */
	"IP_TYPE",			/**< XNL_ATTR_IP_TYPE */
	"DEV_NUMQS",			/**<XNL_ATTR_DEV_NUMQS */
	"DEV_NUM_PFS",			/**<XNL_ATTR_DEV_NUM_PFS */
	"DEV_MM_CHANNEL_MAX",		/**<XNL_ATTR_DEV_MM_CHANNEL_MAX */
	"DEV_MAILBOX_ENABLE",		/**<XNL_ATTR_DEV_MAILBOX_ENABLE */
	"DEV_FLR_PRESENT",		/**<XNL_ATTR_DEV_FLR_PRESENT */
	"DEV_ST_ENABLE",		/**<XNL_ATTR_DEV_ST_ENABLE */
	"DEV_MM_ENABLE",		/**<XNL_ATTR_DEV_MM_ENABLE */
	"DEV_MM_CMPT_ENABLE",		/**<XNL_ATTR_DEV_MM_CMPT_ENABLE */
	"REG_BAR",		        /**< XNL_ATTR_REG_BAR_NUM */
	"REG_ADDR",		        /**< XNL_ATTR_REG_ADDR */
	"REG_VAL",		        /**< XNL_ATTR_REG_VAL */
	"CSR_INDEX",			/**< XNL_ATTR_CSR_INDEX*/
	"CSR_COUNT",			/**< XNL_ATTR_CSR_COUNT*/
	"QIDX",			        /**< XNL_ATTR_QIDX */
	"NUM_Q",		        /**< XNL_ATTR_NUM_Q */
	"QFLAG",		        /**< XNL_ATTR_QFLAG */
	"CMPT_DESC_SZ",			/**< XNL_ATTR_CMPT_DESC_SIZE */
	"SW_DESC_SIZE",			/**< XNL_ATTR_SW_DESC_SIZE */
	"QRINGSZ_IDX",			/**< XNL_ATTR_QRNGSZ */
	"C2H_BUFSZ_IDX",		/**< XNL_ATTR_QBUFSZ */
	"CMPT_TIMER_IDX",		/**< XNL_ATTR_CMPT_TIMER_IDX */
	"CMPT_CNTR_IDX",		/**< XNL_ATTR_CMPT_CNTR_IDX */
	"CMPT_TRIG_MODE",		/**< XNL_ATTR_CMPT_TRIG_MODE */
	"RANGE_START",			/**< XNL_ATTR_RANGE_START */
	"RANGE_END",			/**< XNL_ATTR_RANGE_END */
	"INTR_VECTOR_IDX",		/**< XNL_ATTR_INTR_VECTOR_IDX */
	"INTR_VECTOR_START_IDX",	/**< XNL_ATTR_INTR_VECTOR_START_IDX */
	"INTR_VECTOR_END_IDX",		/**< XNL_ATTR_INTR_VECTOR_END_IDX */
	"RSP_BUF_LEN",			/**< XNL_ATTR_RSP_BUF_LEN */
	"GLOBAL_CSR",			/**< global csr data */
	"PIPE_GL_MAX",			/**< max no. of gl for pipe */
	"PIPE_FLOW_ID",			/**< pipe flow id */
	"PIPE_SLR_ID",			/**< pipe slr id */
	"PIPE_TDEST",			/**< pipe tdest */
	"DEV_STM_BAR",			/**< device STM bar number */
	"Q_STATE",			/**< XNL_ATTR_Q_STATE*/
	"ERROR",			/**< XNL_ATTR_ERROR */
	"PING_PONG_EN",		/**< XNL_PING_PONG_EN */
	"DEV_ATTR",			/**< XNL_ATTR_DEV */
	"XNL_ATTR_DEBUG_EN",	/** XNL_ATTR_DEBUG_EN */
	"XNL_ATTR_DESC_ENGINE_MODE",	/** XNL_ATTR_DESC_ENGINE_MODE */
#ifdef ERR_DEBUG
	"QPARAM_ERR_INFO",		/**< queue param info */
#endif
	"ATTR_MAX",

};



/* commands, 0 ~ 0x7F */
/**
 * xnl_op_t - XNL command types
 */
enum xnl_op_t {
	XNL_CMD_DEV_LIST,	/**< list all the qdma devices */
	XNL_CMD_DEV_INFO,	/**< dump the device information */
	XNL_CMD_DEV_STAT,	/**< dump the device statistics */
	XNL_CMD_DEV_STAT_CLEAR,	/**< reset the device statistics */

	XNL_CMD_REG_DUMP,	/**< dump the register information */
	XNL_CMD_REG_RD,		/**< read a register value */
	XNL_CMD_REG_WRT,	/**< write value to a register */

	XNL_CMD_Q_LIST,		/**< list all the queue present in the system */
	XNL_CMD_Q_ADD,		/**< add a queue */
	XNL_CMD_Q_START,	/**< start a queue */
	XNL_CMD_Q_STOP,		/**< stop a queue */
	XNL_CMD_Q_DEL,		/**< delete a queue */
	XNL_CMD_Q_DUMP,		/**< dump queue information*/
	XNL_CMD_Q_DESC,		/**< dump descriptor information*/
	XNL_CMD_Q_CMPT,		/**< dump writeback descriptor information*/
	XNL_CMD_Q_RX_PKT,	/**< dump packet information*/
	XNL_CMD_Q_CMPT_READ,	/**< read the cmpt data */
#ifdef ERR_DEBUG
	XNL_CMD_Q_ERR_INDUCE,	/**< induce an error*/
#endif

	XNL_CMD_INTR_RING_DUMP,	/**< dump interrupt ring information*/
	XNL_CMD_Q_UDD,		/**< dump the user defined data */
	XNL_CMD_GLOBAL_CSR,	/**< get all global csr register values */
	XNL_CMD_DEV_CAP,	/**< list h/w capabilities , hw and sw version */
	XNL_CMD_GET_Q_STATE,	/**< get the queue state */
	XNL_CMD_REG_INFO_READ,  /**< read register info */
#ifdef TANDEM_BOOT_SUPPORTED
	XNL_CMD_EN_ST,  	/**< Enable Streaming */
#endif
	XNL_CMD_MAX,		/**< max number of XNL commands*/
};

/**
 * XNL command operation type
 */
static const char *xnl_op_str[XNL_CMD_MAX] = {
	"DEV_LIST",		/** XNL_CMD_DEV_LIST */
	"DEV_INFO",		/** XNL_CMD_DEV_INFO */
	"DEV_STAT",		/** XNL_CMD_DEV_STAT */
	"DEV_STAT_CLEAR",	/** XNL_CMD_DEV_STAT_CLEAR */

	"REG_DUMP",		/** XNL_CMD_REG_DUMP */
	"REG_READ",		/** XNL_CMD_REG_RD */
	"REG_WRITE",		/** XNL_CMD_REG_WRT */

	"Q_LIST",		/** XNL_CMD_Q_LIST */
	"Q_ADD",		/** XNL_CMD_Q_ADD */
	"Q_START",		/** XNL_CMD_Q_START */
	"Q_STOP",		/** XNL_CMD_Q_STOP */
	"Q_DEL",		/** XNL_CMD_Q_DEL */
	"Q_DUMP",		/** XNL_CMD_Q_DUMP */
	"Q_DESC",		/** XNL_CMD_Q_DESC */
	"Q_CMPT",		/** XNL_CMD_Q_CMPT */
	"Q_RX_PKT",		/** XNL_CMD_Q_RX_PKT */
	"Q_CMPT_READ",		/** XNL_CMD_Q_CMPT_READ */
#ifdef ERR_DEBUG
	"Q_ERR_INDUCE",		/** XNL_CMD_Q_ERR_INDUCE */
#endif
	"INTR_RING_DUMP",	/** XNL_CMD_INTR_RING_DUMP */
	"Q_UDD_DUMP",		/** XNL_CMD_Q_UDD */
	"GLOBAL_CSR",		/** XNL_CMD_GLOBAL_CSR*/
	"DEV_CAP",			/** XNL_CMD_DEV_CAP */
	"GET_Q_STATE",		/** XNL_CMD_GET_Q_STATE */
	"REG_INFO_READ",		/** XNL_CMD_REG_INFO_READ */
#ifdef TANDEM_BOOT_SUPPORTED
	"EN_ST"				/** XNL_CMD_EN_ST */
#endif
};

enum qdma_queue_state {
	QUEUE_DISABLED,
	QUEUE_ENABLED,
	QUEUE_ONLINE
};


/**
 * qdma_err_idx - Induce error
 */
enum qdma_err_idx {
	err_ram_sbe,
	err_ram_dbe,
	err_dsc,
	err_trq,
	err_h2c_mm_0,
	err_h2c_mm_1,
	err_c2h_mm_0,
	err_c2h_mm_1,
	err_c2h_st,
	ind_ctxt_cmd_err,
	err_bdg,
	err_h2c_st,
	poison,
	ur_ca,
	param,
	addr,
	tag,
	flr,
	timeout,
	dat_poison,
	flr_cancel,
	dma,
	dsc,
	rq_cancel,
	dbe,
	sbe,
	unmapped,
	qid_range,
	vf_access_err,
	tcp_timeout,
	mty_mismatch,
	len_mismatch,
	qid_mismatch,
	desc_rsp_err,
	eng_wpl_data_par_err,
	msi_int_fail,
	err_desc_cnt,
	portid_ctxt_mismatch,
	portid_byp_in_mismatch,
	cmpt_inv_q_err,
	cmpt_qfull_err,
	cmpt_cidx_err,
	cmpt_prty_err,
	fatal_mty_mismatch,
	fatal_len_mismatch,
	fatal_qid_mismatch,
	timer_fifo_ram_rdbe,
	fatal_eng_wpl_data_par_err,
	pfch_II_ram_rdbe,
	cmpt_ctxt_ram_rdbe,
	pfch_ctxt_ram_rdbe,
	desc_req_fifo_ram_rdbe,
	int_ctxt_ram_rdbe,
	cmpt_coal_data_ram_rdbe,
	tuser_fifo_ram_rdbe,
	qid_fifo_ram_rdbe,
	payload_fifo_ram_rdbe,
	wpl_data_par_err,
	zero_len_desc_err,
	csi_mop_err,
	no_dma_dsc_err,
	sb_mi_h2c0_dat,
	sb_mi_c2h0_dat,
	sb_h2c_rd_brg_dat,
	sb_h2c_wr_brg_dat,
	sb_c2h_rd_brg_dat,
	sb_c2h_wr_brg_dat,
	sb_func_map,
	sb_dsc_hw_ctxt,
	sb_dsc_crd_rcv,
	sb_dsc_sw_ctxt,
	sb_dsc_cpli,
	sb_dsc_cpld,
	sb_pasid_ctxt_ram,
	sb_timer_fifo_ram,
	sb_payload_fifo_ram,
	sb_qid_fifo_ram,
	sb_tuser_fifo_ram,
	sb_wrb_coal_data_ram,
	sb_int_qid2vec_ram,
	sb_int_ctxt_ram,
	sb_desc_req_fifo_ram,
	sb_pfch_ctxt_ram,
	sb_wrb_ctxt_ram,
	sb_pfch_ll_ram,
	sb_h2c_pend_fifo,
	db_mi_h2c0_dat,
	db_mi_c2h0_dat,
	db_h2c_rd_brg_dat,
	db_h2c_wr_brg_dat,
	db_c2h_rd_brg_dat,
	db_c2h_wr_brg_dat,
	db_func_map,
	db_dsc_hw_ctxt,
	db_dsc_crd_rcv,
	db_dsc_sw_ctxt,
	db_dsc_cpli,
	db_dsc_cpld,
	db_pasid_ctxt_ram,
	db_timer_fifo_ram,
	db_payload_fifo_ram,
	db_qid_fifo_ram,
	db_tuser_fifo_ram,
	db_wrb_coal_data_ram,
	db_int_qid2vec_ram,
	db_int_ctxt_ram,
	db_desc_req_fifo_ram,
	db_pfch_ctxt_ram,
	db_wrb_ctxt_ram,
	db_pfch_ll_ram,
	db_h2c_pend_fifo,
	qdma_errs
};


#ifdef __cplusplus
}
#endif

#endif /* ifndef QDMA_NL_H__ */
