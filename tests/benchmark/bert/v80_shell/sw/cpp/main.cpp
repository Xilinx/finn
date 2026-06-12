/*
 * This file is part of the QDMA userspace application
 * to enable the user to execute the QDMA functionality
 *
 * Copyright (c) 2018-2022, Xilinx, Inc. All rights reserved.
 * Copyright (c) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under BSD-style license (found in the
 * LICENSE file in the root directory of this source tree)
 */

#define _DEFAULT_SOURCE
#define _XOPEN_SOURCE 500

#include <assert.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>

// -------------------------------------------------------

#include "dmabwave.h"
#include "extract_sys.hpp"

// Runtime
// -------------------------------------------------------
#define DEF_SIZE (1024)

// This is an example main.cpp, adapt as necessary
// Python is used for the deployment of Brainsmith cores
// -------------------------------------------------------
int main(int argc, char *argv[]) 
{
    int ret = 0;
    struct timespec ts_start, ts_end;
	double time_c2h, time_h2c;

	// Args
	uint64_t size = DEF_SIZE;
	
	// Setup
	ret = setup_qdma(CONFIG_PATH); if(ret < 0) {  return EXIT_FAILURE; }
	volatile uint64_t *csr = map_csr(); if(!csr) { return EXIT_FAILURE; }
	
    // Allocate buffers
	int32_t *buffer = NULL;
	ret = posix_memalign((void **)&buffer, 4096 , size + 4096);
	if(ret) { return EXIT_FAILURE; }

    // Offload
	clock_gettime(CLOCK_MONOTONIC, &ts_start);

    ret = dma_xfer(q_info->q_name, (char*) buffer, size, 0, H2C_TRANSFER);
    if(ret < 0) { goto err_h2c; }

    ret = clock_gettime(CLOCK_MONOTONIC, &ts_end);
	timespec_sub(&ts_end, &ts_start);
	time_h2c = (ts_end.tv_sec + ((double)ts_end.tv_nsec/NSEC_DIV));
	printf("** H2C time %f sec\n", time_h2c);

    // Sync
	clock_gettime(CLOCK_MONOTONIC, &ts_start);
	
    ret = dma_xfer(q_info->q_name, (char*) buffer, size, 0, C2H_TRANSFER);
    if(ret < 0) { goto err_c2h; }
	
	ret = clock_gettime(CLOCK_MONOTONIC, &ts_end);
	timespec_sub(&ts_end, &ts_start);
	time_c2h = (ts_end.tv_sec + ((double)ts_end.tv_nsec/NSEC_DIV));
	printf("** C2H time %f sec\n", time_c2h);
    
	printf("\n");

err_c2h:
err_h2c:
    free(buffer); 
	
	ret = EXIT_SUCCESS;
}