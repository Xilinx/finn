/*
 * This file is part of the QDMA userspace application
 * to enable the user to execute the QDMA functionality
 *
 * Copyright (c) 2019 - 2022,  Xilinx, Inc.
 * All rights reserved.
 * Copyright (c) 2022-2024,  Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under BSD-style license (found in the
 * LICENSE file in the root directory of this source tree)
 */

#ifndef __DMAXFER_H__
#define __DMAXFER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "dmautils.h"
#define DMAPERF_PAGE_SIZE          4096
#define DMAPERF_PAGE_SHIFT         12


enum dmaxfer_io_type {
	DMAXFER_IO_SYNC,
	DMAXFER_IO_ASYNC
};

struct dmaxfer_io_info {
	char *file_name;
	unsigned char write;
	int num_jobs;
	int pkt_sz;
	int pkt_burst;
	int runtime;
	unsigned int max_req_outstanding;
	unsigned long long int pps;
	int (*app_env_init)(unsigned long *handle);
	int (*app_env_exit)(unsigned long *handle);
	unsigned long handle;
};

int dmaxfer_perf_run(struct dmaxfer_io_info *list,
		unsigned int num_entries);
void dmaxfer_perf_stop(struct dmaxfer_io_info *list,
			unsigned int num_entries);
ssize_t dmaxfer_iosubmit(char *fname, unsigned char write,
			 enum dmaxfer_io_type io_type, char *buffer,
			 uint64_t size);


#ifdef __cplusplus
}
#endif

#endif /* __DMAXFER_H__ */
