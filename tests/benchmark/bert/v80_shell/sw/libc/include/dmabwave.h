// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is subject to the Xilinx Design License Agreement located
// in the LICENSE.md file in the root directory of this repository.
//
// This file contains confidential and proprietary information of Xilinx, Inc.
// and is protected under U.S. and international copyright and other
// intellectual property laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any rights to the materials
// distributed herewith. Except as otherwise provided in a valid license issued to
// you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
// MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
// DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
// FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
// in contract or tort, including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature related to, arising
// under or in connection with these materials, including for any direct, or any
// indirect, special, incidental, or consequential loss or damage (including loss
// of data, profits, goodwill, or any type of loss or damage suffered as a result
// of any action brought by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the possibility of the
// same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-safe, or for use in
// any application requiring failsafe performance, such as life-support or safety
// devices or systems, Class III medical devices, nuclear facilities, applications
// related to the deployment of airbags, or any other applications that could lead
// to death, personal injury, or severe property or environmental damage
// (individually and collectively, "Critical Applications"). Customer assumes the
// sole risk and liability of any use of Xilinx products in Critical Applications,
// subject only to applicable laws and regulations governing limitations on product
// liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.
// CS

#ifndef DMABWAVE_H
#define DMABWAVE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <stdbool.h>
#include <linux/types.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>
#include <ctype.h>
#include <errno.h>
#include <error.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/sysinfo.h>
#include "version.h"
#include "dmautils.h"
#include "qdma_nl.h"
#include "dmaxfer.h"
#include "time.h"
#include "math.h"
#include <sys/time.h>

// PCIe
// -------------------------------------------------------
#define BAR_SIZE_DEFAULT (256 * 1024 * 1024)
#define H2C_TRANSFER (0)
#define C2H_TRANSFER (1)
#define BEAT_BYTES (64)
#define CSRBW_OFFS (0x02000000)
#define CSRBW_SIZE (8 * 1024)

#define RW_MAX_SIZE	0x7ffff000
#define GB_DIV 1000000000
#define MB_DIV 1000000
#define KB_DIV 1000
#define NSEC_DIV 1000000000

// Banks
// -------------------------------------------------------
#define SRC_ADDRESS (0x4000000000UL) // Source bank offset
#define DST_ADDRESS (0x4080000000UL) // Destination bank offset
#define IBF_ADDRESS (0x4100000000UL) // Intermediate buffers
#define WGT_ADDRESS (0x4180000000UL) // Weight buffers
#define WGT_RANGE 	(0x80000000UL) 	 // Weight range

// Queues
// -------------------------------------------------------
#define QDMA_Q_NAME_LEN     100
#define QDMA_ST_MAX_PKT_SIZE 0x7000
#define QDMA_RW_MAX_SIZE	0x7ffff000
#define QDMA_GLBL_MAX_ENTRIES  (16)

extern struct queue_info *q_info;
extern int q_count;

enum qdma_q_dir {
    QDMA_Q_DIR_H2C,
	QDMA_Q_DIR_C2H,
	QDMA_Q_DIR_BIDI
};

enum qdma_q_mode {
	QDMA_Q_MODE_MM,
	QDMA_Q_MODE_ST
};

struct queue_info {
	char *q_name;
	int qid;
	int pf;
	enum qdmautils_io_dir dir;
};

static enum qdma_q_mode mode;
static enum qdma_q_dir dir;

static char cfg_name[64];
static unsigned int pkt_sz;
static unsigned int pci_bus;
static unsigned int pci_dev;
static char pci_resource[64];
static int pci_fd;
static int fun_id = -1;
static int is_vf;
static unsigned int q_start;
static unsigned int num_q;
static unsigned int idx_rngsz;
static unsigned int idx_tmr;
static unsigned int idx_cnt;
static unsigned int pfetch_en;
static unsigned int cmptsz;
static char input_file[128];
static char output_file[128];
static int io_type;
static char trigmode_str[10];
static unsigned char trig_mode;

// QDMA functions
// -------------------------------------------------------
int setup_qdma(const char *cfg_fname);
void clean_qdma();
volatile uint64_t* map_csr();

int dma_xfer(char *qname, char *buffer, size_t size, uint64_t base, int8_t dir);

// Util
// -------------------------------------------------------
int read_io_ref(char *filename, int32_t *buffer, size_t buffer_size);
float bits_to_float(uint32_t bits);
int compare_floats(float a, float b, float atol);
void fill_random(void *buffer, size_t size);
void timespec_sub(struct timespec *t1, struct timespec *t2);

#ifdef __cplusplus
}
#endif

#endif // DMABWAVE_H