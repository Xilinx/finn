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
#include "dmautils.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "qdma_nl.h"
#include "dmaxfer.h"
#include "version.h"

#define QDMA_Q_NAME_LEN 100

enum qdma_reg_op {
	QDMA_Q_REG_DUMP,
	QDMA_Q_REG_RD,
	QDMA_Q_REG_WR,
};


int qdmautils_sync_xfer(char *filename, enum qdmautils_io_dir dir, void *buf,
			unsigned int xfer_len)
{
	return dmaxfer_iosubmit(filename, dir, DMAXFER_IO_SYNC, buf, xfer_len);
}

int qdmautils_async_xfer(char *filename, enum qdmautils_io_dir dir, void *buf,
			unsigned int xfer_len)
{
	return dmaxfer_iosubmit(filename, dir, DMAXFER_IO_ASYNC, buf, xfer_len);
}
