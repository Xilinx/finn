/*
 * This file is part of the QDMA userspace application
 * to enable the user to execute the QDMA functionality
 *
 * Copyright (c) 2019 - 2020,  Xilinx, Inc.
 * All rights reserved.
 * Copyright (c) 2022-2024,  Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under BSD-style license (found in the
 * LICENSE file in the root directory of this source tree)
 */
#ifndef QDMAUTILS_H
#define QDMAUTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/** @QDMA_GLOBAL_CSR_ARRAY_SZ: QDMA Global CSR array size */
#define QDMA_GLOBAL_CSR_ARRAY_SZ        16

/**
 * enum dmautils_io_dir - DMA direction to perform
 */
enum qdmautils_io_dir {
	/** @DMAXFER_IO_READ: DMA card to host */
	DMAXFER_IO_READ,
	/** @DMAXFER_IO_WRITE: DMA host to card */
	DMAXFER_IO_WRITE,
};


enum qdma_q_fetch_credit {
	/** @NONE: Disable Fetch Credit**/
	Q_DISABLE_FETCH_CREDIT = 0,
	/** @Q_H2C: Enable fetch credit for H2C direction */
	Q_ENABLE_H2C_FETCH_CREDIT,
	/** @Q_C2H: Enable fetch credit for C2H direction */
	Q_ENABLE_C2H_FETCH_CREDIT,
	/** @Q_H2C_C2H: Enable fetch credit for H2C and C2H direction */
	Q_ENABLE_H2C_C2H_FETCH_CREDIT,
};


/**
 * enum qdma_q_state - QDMA q state
 */
enum qdma_q_state {
	/** @Q_STATE_DISABLED: Queue is not taken */
	Q_STATE_DISABLED = 0,
	/** @Q_STATE_ENABLED: Assigned/taken. Partial config is done */
	Q_STATE_ENABLED,
	/** @Q_STATE_ONLINE: Resource/context is initialized for the queue and is available for
	 *  data consumption
	 */
	Q_STATE_ONLINE,
};

/** @set_qparam - helper macro to set  q param
 *  @xcmd: pointer to xcmd_info
 *  @var_ptr: pointer to the variable to be updated in xcmd->u.qparm
 *  @param_type: type of the q param that is being updated
 *  @val: calue to be updated in @var_ptr
 */
#define set_qparam(xcmd, var_ptr, param_type, val) \
        do {\
        	*var_ptr = val; \
		xcmd->sflags |= (1 << param_type); \
	} while (0);

/**
 * enum qdma_q_parm_type - QDMA q param type to indicate what all params are
 *                         provided
 */
enum qdma_q_parm_type {
	/** @QPARM_IDX: q index param */
	QPARM_IDX,
	/** @QPARM_MODE: q mode param */
	QPARM_MODE,
	/** @QPARM_DIR: q direction param */
	QPARM_DIR,
	/** @QPARM_DESC: q desc request param */
	QPARM_DESC,
	/** @QPARM_CMPT: q cmpt desc request param */
	QPARM_CMPT,
	/** @QPARM_CMPTSZ: q cmpt size param */
	QPARM_CMPTSZ,
	/** @QPARM_SW_DESC_SZ: q sw desc size param */
	QPARM_SW_DESC_SZ,
	/** @QPARM_RNGSZ_IDX: q ring size idx param */
	QPARM_RNGSZ_IDX,
	/** @QPARM_C2H_BUFSZ_IDX: q c2h buf size idx param */
	QPARM_C2H_BUFSZ_IDX,
	/** @QPARM_CMPT_TMR_IDX: q cmpt timer idx param */
	QPARM_CMPT_TMR_IDX,
	/** @QPARM_CMPT_CNTR_IDX: q cmpt counter idx param */
	QPARM_CMPT_CNTR_IDX,
	/** @QPARM_CMPT_TRIG_MODE: q cmpt trigger mode param  */
	QPARM_CMPT_TRIG_MODE,
	/** @QPARM_PING_PONG_EN: ping pong param  */
	QPARM_PING_PONG_EN,
	/** @KEYHOLE_PARAM: keyhole feature aperture */
	QPARM_KEYHOLE_EN,
	/** @QPARM_MM_CHANNEL: q mm channel enable param */
	QPARM_MM_CHANNEL,
	/** @QPARM_MAX: max q param */
	QPARM_MAX,
};

/**
 * struct xcmd_q_parm - QDMA q param information
 */
struct xcmd_q_parm {
	/** @sflags: flgas to indicate the presence of parms with
	 *           @qdma_q_parm_type */
	unsigned int sflags;
	/** @sflags: flgas to indicate the presence of features */
	unsigned int flags;
	/** @idx:  index of queue */
	unsigned int idx;
	/** @num_q: num of queue */
	unsigned int num_q;
	/** @range_start:  start of range */
	unsigned int range_start;
	/** @range_end:  end of range */
	unsigned int range_end;
	/** @sw_desc_sz: SW desc size */
	unsigned char sw_desc_sz;
	/** @cmpt_entry_size: completion desc size */
	unsigned char cmpt_entry_size;
	/** @qrngsz_idx: ring size idx */
	unsigned char qrngsz_idx;
	/** @c2h_bufsz_idx: c2h buf size index */
	unsigned char c2h_bufsz_idx;
	/** @cmpt_tmr_idx: cmpt timer index */
	unsigned char cmpt_tmr_idx;
	/** @cmpt_cntr_idx: cmpt counter index */
	unsigned char cmpt_cntr_idx;
	/** @cmpt_trig_mode: cmpt trigger mode */
	unsigned char cmpt_trig_mode;
	/** @mm_channel: mm channel enable */
	unsigned char mm_channel;
	/** @fetch_credit: fetch credit enable */
	unsigned char fetch_credit;
	/** @is_qp: queue pair */
	unsigned char is_qp;
	/** @ping_pong_en: ping pong en */
	unsigned char ping_pong_en;
	/** @aperture_sz: aperture_size for keyhole transfers*/
	unsigned int aperture_sz;
};

/**
 * struct xcmd_intr - QDMA dev interrupt ring dump
 */
struct xcmd_intr {
	/** @vector: vector number */
	unsigned int vector;
	/** @start_idx: start idx of interrupt ring descriptor */
	int start_idx;
	/** @end_idx: end idx of interrupt ring descriptor */
	int end_idx;
};

/** @QDMA_VERSION_INFO_STR_LENGTH: QDMA version string length */
#define QDMA_VERSION_INFO_STR_LENGTH            256

/**
 * struct xcmd_dev_cap - QDMA device capabilites
 */
struct xcmd_dev_cap {
	/** @version_str: qdma version string */
	char version_str[QDMA_VERSION_INFO_STR_LENGTH];
	/** @num_qs: HW qmax */
	unsigned int num_qs;
	/** @num_pfs: HW number of pf */
	unsigned int num_pfs;
	/** @flr_present: flr support */
	unsigned int flr_present;
	/** @mm_en: MM mode support */
	unsigned int mm_en;
	/** @mm_cmpt_en: MM CMPT mode support */
	unsigned int mm_cmpt_en;
	/** @st_en: ST mode support */
	unsigned int st_en;
	/** @mailbox_en: Mailbox support */
	unsigned int mailbox_en;
	/** @mm_channel_max: Max MM channel */
	unsigned int mm_channel_max;
	/** @debug_mode: Debug Mode*/
	unsigned int debug_mode;
	/** @desc_eng_mode: Descriptor Engine Mode*/
	unsigned int desc_eng_mode;
};

/**
 * struct xcmd_reg - register access command info
 */
struct xcmd_reg {
	/** @sflags: flags to indicate parameter presence */
	unsigned int sflags;
	/** @XCMD_REG_F_BAR_SET: bar param set */
#define XCMD_REG_F_BAR_SET	0x1
	/** @XCMD_REG_F_REG_SET: reg param set */
#define XCMD_REG_F_REG_SET	0x2
	/** @XCMD_REG_F_VAL_SET: val param set */
#define XCMD_REG_F_VAL_SET	0x4
	/** @bar: bar number */
	unsigned int bar;
	/** @reg: register offset */
	unsigned int reg;
	/** @val: value */
	unsigned int val;
	/** @range_start: range start */
	unsigned int range_start;
	/** @range_end: range end */
	unsigned int range_end;
};

/**
 * struct xnl_dev_info - device information
 */
struct xnl_dev_info {
	/** @pci_bus: pci bus */
	unsigned char pci_bus;
	/** @pci_dev: pci device */
	unsigned char pci_dev;
	/** @dev_func: pci function */
	unsigned char dev_func;
	/** @config_bar: config bar */
	unsigned char config_bar;
	/** @user_bar: AXI Master Lite(user bar) */
	unsigned char user_bar;
	/** @qmax: SW qmax */
	unsigned int qmax;
	/** @qbase: HW q base */
	unsigned int qbase;
};

/**
 * struct xnl_dev_stat - device statistics
 */
struct xnl_dev_stat {
	/** @mm_h2c_pkts: MM H2C packets processed */
	unsigned long long mm_h2c_pkts;
	/** @mm_c2h_pkts: MM C2H packets processed */
	unsigned long long mm_c2h_pkts;
	/** @st_h2c_pkts: ST H2C packets processed */
	unsigned long long st_h2c_pkts;
	/** @st_c2h_pkts: ST C2H packets processed */
	unsigned long long st_c2h_pkts;
	/** max ping_pong latency */
	unsigned long long ping_pong_lat_max;
	/** min ping_pong latency */
	unsigned long long ping_pong_lat_min;
	/** avg ping_pong latency */
	unsigned long long ping_pong_lat_avg;
};

/**
 * struct xnl_q_info - q state information
 */
struct xnl_q_info {
	/** @flags: feature flags enabled */
	unsigned int flags;
	/** @qidx: index of queue */
	unsigned int qidx;
	/** @state: queue state */
	enum qdma_q_state state;
};

/**
 * struct global_csr_conf - Global CSR information
 */
struct global_csr_conf {
	/** @ring_sz: Descriptor ring size ie. queue depth */
	unsigned int ring_sz[QDMA_GLOBAL_CSR_ARRAY_SZ];
	/** @c2h_timer_cnt: C2H timer count  list */
	unsigned int c2h_timer_cnt[QDMA_GLOBAL_CSR_ARRAY_SZ];
	/** @c2h_cnt_th: C2H counter threshold list*/
	unsigned int c2h_cnt_th[QDMA_GLOBAL_CSR_ARRAY_SZ];
	/** @c2h_buf_sz: C2H buffer size list */
	unsigned int c2h_buf_sz[QDMA_GLOBAL_CSR_ARRAY_SZ];
	/** @wb_intvl: Writeback interval */
	unsigned int wb_intvl;
};


/**
 * struct xcmd_info - command information
 */
struct xcmd_info {
	/** @vf: is VF */
	unsigned char vf:1;
	/** @op: operation */
	unsigned char op:7;
	/** @req: union of other information for request */
	union {
		/** @intr: interrupt ring info */
		struct xcmd_intr intr;
		/** @reg: reg command info */
		struct xcmd_reg reg;
		/** @qparm: q command info */
		struct xcmd_q_parm qparm;
	} req;
	/** @resp: union of information from response */
	union {
		/** @cap: device capabilities response */
		struct xcmd_dev_cap cap;
		/** @dev_stat: device stat response */
		struct xnl_dev_stat dev_stat;
		/** @dev_info: device info response */
		struct xnl_dev_info dev_info;
		/** @q_info: queue info response */
		struct xnl_q_info q_info;
		/** @csr: CSR reponse */
		struct global_csr_conf csr;
	} resp;
	/** @if_bdf: interface BDF */
	unsigned int if_bdf;
	/** @log_msg_dump: log messages dumping function */
	void (*log_msg_dump)(const char *resp_str);
};

/** @QDMAUTILS_VERSION_LEN: qdmautils version string length */
#define QDMAUTILS_VERSION_LEN   10
/**
 * struct qdmautils_version - qdmautils version information
 */
struct qdmautils_version {
	char version[QDMAUTILS_VERSION_LEN];
};

/*****************************************************************************/
/**
 * qdmautils_get_version() - Get qdmautils version
 *
 * @ver:	Pointer to the output data structure where version is to be
 *              updated
 *
 * Return:	Nothing
 *
 *****************************************************************************/
void qdmautils_get_version(struct qdmautils_version *ver);

/*****************************************************************************/
/**
 * qdma_dev_list_dump() - dump device list
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_dev_list_dump(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_dev_info() - get device information provided by cmd->u.dev_info
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_dev_info(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_dev_cap() - get device capabilities provided by cmd->u.dev_cap
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_dev_cap(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_dev_stat() - get device statistics provided by cmd->u.dev_stat
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_dev_stat(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_dev_stat_clear() - clear device statistics
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_dev_stat_clear(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_dev_intr_ring_dump() - dump device interrupt ring
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_dev_intr_ring_dump(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_dev_get_global_csr() - get global CSR provided by cmd->u.csr
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_dev_get_global_csr(struct xcmd_info *cmd);


/*****************************************************************************/
/**
 * qdma_dev_q_list_dump() - dump device q list
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_dev_q_list_dump(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_q_add() - add a queue
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_q_add(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_q_del() - delete a queue
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_q_del(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_q_start() - start a queue
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_q_start(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_q_stop() - stop a queue
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_q_stop(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_q_get_state() - get q state information provided by cmd->u.q_info
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_q_get_state(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_q_dump() - dump q context
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_q_dump(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_q_desc_dump() - dump q sw/cmpt descriptors
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_q_desc_dump(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_q_cmpt_read() - dump last cmpt descriptor of the queue
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_q_cmpt_read(struct xcmd_info *cmd);


/*****************************************************************************/
/**
 * qdma_reg_read() - read register
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_reg_read(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_reg_info_read() - read register fields information
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_reg_info_read(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_reg_write() - write register
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_reg_write(struct xcmd_info *cmd);

/*****************************************************************************/
/**
 * qdma_reg_dump() - dump registers
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_reg_dump(struct xcmd_info *cmd);


/*****************************************************************************/
/**
 * qdmautils_ioctl_nomemcpy() - set no memcpy from ioctl for the cdev
 *
 * @filename:	filename on which ioctl needs to be done
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdmautils_ioctl_nomemcpy(char *filename);

/*****************************************************************************/
/**
 * qdmautils_sync_xfer() - sync transfer
 *
 * @filename:	filename on which transfer needs to be done
 * @dir:	direction oif operation
 * @buf:        pointer to buffer
 * @xfer_len:   size of transfer
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdmautils_sync_xfer(char *filename, enum qdmautils_io_dir dir, void *buf,
		   unsigned int xfer_len);

/*****************************************************************************/
/**
 * qdmautils_async_xfer() - async trasnfer
 *
 * @filename:	filename on which transfer needs to be done
 * @dir:	direction oif operation
 * @buf:        pointer to buffer
 * @xfer_len:   size of transfer
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdmautils_async_xfer(char *filename, enum qdmautils_io_dir dir, void *buf,
		   unsigned int xfer_len);

#ifdef TANDEM_BOOT_SUPPORTED
/*****************************************************************************/
/**
 * qdma_en_st() - Enable Streaming
 *
 * @cmd:	command information
 *
 * Return:	>=0 for success and <0 for error
 *
 *****************************************************************************/
int qdma_en_st(struct xcmd_info *cmd);
#endif

#ifdef __cplusplus
}
#endif

#endif /* QDMAUTILS_H */
