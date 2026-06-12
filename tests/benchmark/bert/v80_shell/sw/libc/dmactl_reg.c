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

#include <endian.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include "qdma_nl.h"
#include "dmactl_internal.h"

#define QDMA_CFG_BAR_SIZE 0xB400
#define QDMA_USR_BAR_SIZE 0x100

/* Macros for reading device capability */
#define QDMA_GLBL2_MISC_CAP                       0x134
#define     QDMA_GLBL2_MM_CMPT_EN_MASK            0x4
#define QDMA_GLBL2_CHANNEL_MDMA                   0x118
#define     QDMA_GLBL2_ST_C2H_MASK                0x10000
#define     QDMA_GLBL2_ST_H2C_MASK                0x20000
#define     QDMA_GLBL2_MM_C2H_MASK                0x100
#define     QDMA_GLBL2_MM_H2C_MASK                0x1

#define QDMA_MM_EN_SHIFT          0
#define QDMA_CMPT_EN_SHIFT        1
#define QDMA_ST_EN_SHIFT          2
#define QDMA_MAILBOX_EN_SHIFT     3

#define QDMA_MM_MODE              (1 << QDMA_MM_EN_SHIFT)
#define QDMA_COMPLETION_MODE      (1 << QDMA_CMPT_EN_SHIFT)
#define QDMA_ST_MODE              (1 << QDMA_ST_EN_SHIFT)
#define QDMA_MAILBOX              (1 << QDMA_MAILBOX_EN_SHIFT)


#define QDMA_MM_ST_MODE \
	(QDMA_MM_MODE | QDMA_COMPLETION_MODE | QDMA_ST_MODE)

#define GET_CAPABILITY_MASK(mm_en, st_en, mm_cmpt_en, mailbox_en)  \
	((mm_en << QDMA_MM_EN_SHIFT) | \
			((mm_cmpt_en | st_en) << QDMA_CMPT_EN_SHIFT) | \
			(st_en << QDMA_ST_EN_SHIFT) | \
			(mailbox_en << QDMA_MAILBOX_EN_SHIFT))

struct xreg_info {
	const char name[32];
	uint32_t addr;
	uint32_t repeat;
	uint32_t step;
	uint8_t shift;
	uint8_t len;
	uint8_t mode;
};

static struct xreg_info qdma_user_regs[] = {
	{"ST_C2H_QID", 0x0, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"ST_C2H_PKTLEN", 0x4, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"ST_C2H_CONTROL", 0x8, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	/*  ST_C2H_CONTROL:
	 *	[1] : start C2H
	 *	[2] : immediate data
	 *	[3] : every packet statrs with 00 instead of continuous data
	 *	      stream until # of packets is complete
	 *	[31]: gen_user_reset_n
	 */
	{"ST_H2C_CONTROL", 0xC, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	/*  ST_H2C_CONTROL:
	 *	[0] : clear match for H2C transfer
	 */
	{"ST_H2C_STATUS", 0x10, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"ST_H2C_XFER_CNT", 0x14, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"ST_C2H_PKT_CNT", 0x20, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"ST_C2H_CMPT_DATA", 0x30, 8, 0, 0, 0, QDMA_MM_ST_MODE},
	{"ST_C2H_CMPT_SIZE", 0x50, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"ST_SCRATCH_REG", 0x60, 2, 0, 0, 0, QDMA_MM_ST_MODE},
	{"ST_C2H_PKT_DROP", 0x88, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"ST_C2H_PKT_ACCEPT", 0x8C, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"DSC_BYPASS_LOOP", 0x90, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"USER_INTERRUPT", 0x94, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"USER_INTERRUPT_MASK", 0x98, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"USER_INTERRUPT_VEC", 0x9C, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"DMA_CONTROL", 0xA0, 0, 0, 0, 0, QDMA_MM_ST_MODE},
	{"VDM_MSG_READ", 0xA4, 0, 0, 0, 0, QDMA_MM_ST_MODE},

	{"", 0, 0, 0 }
};


static struct xreg_info qdma_cpm_dmap_regs[] = {
/* QDMA_TRQ_SEL_QUEUE_PF (0x6400) */
	{"DMAP_SEL_INT_CIDX", 0x6400, 512, 0x10, 0, 0, QDMA_MM_ST_MODE},
	{"DMAP_SEL_H2C_DSC_PIDX", 0x6404, 512, 0x10, 0, 0,  QDMA_MM_ST_MODE},
	{"DMAP_SEL_C2H_DSC_PIDX", 0x6408, 512, 0x10, 0, 0, QDMA_MM_ST_MODE},
	{"DMAP_SEL_CMPT_CIDX", 0x640C, 512, 0x10, 0, 0, QDMA_MM_ST_MODE},
	{"", 0, 0, 0 }
};

static struct xreg_info qdma_dmap_regs[] = {
/* QDMA_TRQ_SEL_QUEUE_PF (0x6400) */
	{"DMAP_SEL_INT_CIDX", 0x6400, 512, 0x10, 0, 0, QDMA_MM_ST_MODE},
	{"DMAP_SEL_H2C_DSC_PIDX", 0x6404, 512, 0x10, 0, 0,  QDMA_MM_ST_MODE},
	{"DMAP_SEL_C2H_DSC_PIDX", 0x6408, 512, 0x10, 0, 0, QDMA_MM_ST_MODE},
	{"DMAP_SEL_CMPT_CIDX", 0x640C, 512, 0x10, 0, 0, QDMA_MM_ST_MODE},
	{"", 0, 0, 0 }
};

/*
 * Register I/O through mmap of BAR0.
 */

/* /sys/bus/pci/devices/0000:<bus>:<dev>.<func>/resource<bar#> */
#define get_syspath_bar_mmap(s, bus,dev,func,bar) \
	snprintf(s, sizeof(s), \
		"/sys/bus/pci/devices/0000:%02x:%02x.%x/resource%u", \
		bus, dev, func, bar)

static uint32_t *mmap_bar(char *fname, size_t len, int prot)
{
	int fd;
	uint32_t *bar;

	fd = open(fname, (prot & PROT_WRITE) ? O_RDWR : O_RDONLY);
	if (fd < 0)
		return NULL;

	bar = mmap(NULL, len, prot, MAP_SHARED, fd, 0);
	close(fd);

	return bar == MAP_FAILED ? NULL : bar;
}

static int32_t reg_read_mmap(struct xnl_dev_info *dev_info,
			     unsigned char barno,
			     struct xcmd_info *xcmd)
{
	uint32_t *bar;
	char fname[256];
	int rv = 0;
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	get_syspath_bar_mmap(fname, dev_info->pci_bus, dev_info->pci_dev,
			     dev_info->dev_func, barno);

	bar = mmap_bar(fname, xcmd->req.reg.reg + 4, PROT_READ);
	if (!bar) {
		if (xcmd->op == XNL_CMD_REG_WRT)
			xcmd->op = XNL_CMD_REG_RD;

		rv  = xnl_common_msg_send(xcmd, attrs);

		return rv;
	}

	xcmd->req.reg.val = le32toh(bar[xcmd->req.reg.reg / 4]);
	munmap(bar, xcmd->req.reg.reg + 4);

	return rv;
}

static int32_t reg_write_mmap(struct xnl_dev_info *dev_info,
			      unsigned char barno,
			      struct xcmd_info *xcmd)
{
	uint32_t *bar;
	char fname[256];
	int rv = 0;
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	get_syspath_bar_mmap(fname, dev_info->pci_bus, dev_info->pci_dev,
			     dev_info->dev_func, barno);

	bar = mmap_bar(fname, xcmd->req.reg.reg + 4, PROT_WRITE);
	if (!bar) {
		rv  = xnl_common_msg_send(xcmd, attrs);

		return rv;
	}

	bar[xcmd->req.reg.reg / 4] = htole32(xcmd->req.reg.val);
	munmap(bar, xcmd->req.reg.reg + 4);
	return 0;
}

static void print_repeated_reg(uint32_t *bar, struct xreg_info *xreg,
		unsigned start, unsigned limit, struct xnl_dev_info *dev_info,
		struct xcmd_info *xcmd, void (*log_reg)(const char *))
{
	int i;
	int end = start + limit;
	int step = xreg->step ? xreg->step : 4;
	uint32_t val;
	int32_t rv = 0;
	char reg_dump[100];

	for (i = start; i < end; i++) {
		uint32_t addr = xreg->addr + (i * step);
		char name[45];
		snprintf(name, 45, "%s_%d",
				xreg->name, i);

		if (xcmd == NULL) {
			val = le32toh(bar[addr / 4]);
		} else {
			xcmd->req.reg.reg = addr;
			rv = reg_read_mmap(dev_info, xcmd->req.reg.bar, xcmd);
			if (rv < 0) {
				snprintf(reg_dump, 100, "\n");
				if (log_reg)
					log_reg(reg_dump);
				continue;
			}
		}
		snprintf(reg_dump, 100, "[%#7x] %-47s %#-10x %u\n",
			addr, name, val, val);
		if (log_reg)
			log_reg(reg_dump);
	}
}

static void read_config_regs(uint32_t *bar, struct xnl_dev_info *xdev,
			     struct xcmd_info *xcmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	/* Get the capabilities of the Device */
	xcmd->op = XNL_CMD_REG_DUMP;
	xnl_common_msg_send(xcmd, attrs);
}

static void read_regs(uint32_t *bar, struct xreg_info *reg_list,
		      struct xnl_dev_info *xdev, struct xcmd_info *xcmd,
		      void (*log_reg)(const char *))
{
	struct xreg_info *xreg = reg_list;
	uint32_t val;
	int32_t rv = 0;
	char reg_dump[100] = {'\0'};

	for (xreg = reg_list; strlen(xreg->name); xreg++) {

		if (!xreg->len) {
			if (xreg->repeat) {
				if (xcmd == NULL)
					print_repeated_reg(bar, xreg, 0,
							   xreg->repeat, NULL,
							   NULL, log_reg);
				else
					print_repeated_reg(NULL, xreg, 0,
							   xreg->repeat, xdev,
							   xcmd, log_reg);
			} else {
				uint32_t addr = xreg->addr;
				if (xcmd == NULL) {
					val = le32toh(bar[addr / 4]);
				} else {
					xcmd->req.reg.reg = addr;
					rv = reg_read_mmap(xdev,
							   xcmd->req.reg.bar,
							   xcmd);
					if (rv < 0)
						continue;
				}
				snprintf(reg_dump, 100,
					 "[%#7x] %-47s %#-10x %u\n",
					 addr, xreg->name, val, val);
				if (log_reg)
					log_reg(reg_dump);
			}
		} else {
			uint32_t addr = xreg->addr;
			uint32_t val = 0;
			if (xcmd == NULL) {
				val = le32toh(bar[addr / 4]);
			} else {
				xcmd->req.reg.reg = addr;
				rv = reg_read_mmap(xdev, xcmd->req.reg.bar,
						   xcmd);
				if (rv < 0)
					continue;
			}

			uint32_t v = (val >> xreg->shift) &
					((1 << xreg->len) - 1);

			snprintf(reg_dump, 100,
				 "    %*u:%d %-47s %#-10x %u\n",
				 xreg->shift < 10 ? 3 : 2,
				 xreg->shift + xreg->len - 1,
				 xreg->shift, xreg->name, v, v);
			if (log_reg)
				log_reg(reg_dump);
		}
		memset(reg_dump, '\0', 100);
	}
}

static void reg_dump_mmap(struct xnl_dev_info *dev_info, unsigned char barno,
			struct xreg_info *reg_list, unsigned int max,
			struct xcmd_info *xcmd)
{
	uint32_t *bar;
	char fname[256];

	get_syspath_bar_mmap(fname, dev_info->pci_bus, dev_info->pci_dev,
			     dev_info->dev_func, barno);

	if ((barno == dev_info->config_bar) && (reg_list == NULL))
		read_config_regs(NULL, dev_info, xcmd);
	else {
		bar = mmap_bar(fname, max, PROT_READ);
		if (!bar) {
			xcmd->req.reg.bar = barno;
			xcmd->op = XNL_CMD_REG_RD;
			read_regs(NULL, reg_list, dev_info, xcmd,
					  xcmd->log_msg_dump);
		} else {
			read_regs(bar, reg_list, NULL, NULL, xcmd->log_msg_dump);
			munmap(bar, max);
		}
	}
	return;
}


static void reg_dump_range(struct xnl_dev_info *dev_info, unsigned char barno,
			 unsigned int max, struct xreg_info *reg_list,
			 unsigned int start, unsigned int limit,
			 struct xcmd_info *xcmd)
{
	struct xreg_info *xreg = reg_list;
	uint32_t *bar;
	char fname[256];

	get_syspath_bar_mmap(fname, dev_info->pci_bus, dev_info->pci_dev,
			     dev_info->dev_func, barno);

	bar = mmap_bar(fname, max, PROT_READ);
	if (!bar) {
		xcmd->op = XNL_CMD_REG_RD;
		xcmd->req.reg.bar = barno;
		for (xreg = reg_list; strlen(xreg->name); xreg++) {
			print_repeated_reg(NULL, xreg, start, limit, dev_info,
					   xcmd, xcmd->log_msg_dump);
		}
		return;
	}

	for (xreg = reg_list; strlen(xreg->name); xreg++) {
		print_repeated_reg(bar, xreg, start, limit, NULL, NULL,
				   xcmd->log_msg_dump);
	}

	munmap(bar, max);
}

static inline void print_seperator(struct xcmd_info *xcmd)
{
	char buffer[81];

	memset(buffer, '#', 80);
	buffer[80] = '\0';

	if (xcmd && xcmd->log_msg_dump)
		xcmd->log_msg_dump(buffer);
}

int proc_reg_cmd(struct xcmd_info *xcmd)
{
	struct xcmd_reg *regcmd;
	struct xcmd_info dev_xcmd;
	int32_t rv = 0;
	char reg_dump[100];
	unsigned int barno;
	unsigned char version[QDMA_VERSION_INFO_STR_LENGTH];
	int32_t v;
	unsigned char op;
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	memcpy(&dev_xcmd, xcmd, sizeof(dev_xcmd));

	dev_xcmd.op = XNL_CMD_DEV_CAP;
	rv = qdma_dev_cap(&dev_xcmd);
	if (rv < 0)
		return rv;

	strcpy(version, dev_xcmd.resp.cap.version_str);

	op = xcmd->op;
	xcmd->op = XNL_CMD_DEV_INFO;
	rv = qdma_dev_info(xcmd);
	if (rv < 0)
		return rv;
	xcmd->op = op;
	regcmd = &xcmd->req.reg;
	barno = (regcmd->sflags & XCMD_REG_F_BAR_SET) ?
			 regcmd->bar : xcmd->resp.dev_info.config_bar;
	regcmd->bar = barno;

	switch (xcmd->op) {
	case XNL_CMD_REG_RD:
		rv = reg_read_mmap(&xcmd->resp.dev_info, barno, xcmd);
		if (rv < 0)
			return rv;
		break;
	case XNL_CMD_REG_WRT:
		v = reg_write_mmap(&xcmd->resp.dev_info, barno, xcmd);
		if (v < 0)
			return rv;
		rv = reg_read_mmap(&xcmd->resp.dev_info, barno, xcmd);
		if (rv < 0)
			return rv;
		break;
	case XNL_CMD_REG_DUMP:
		print_seperator(xcmd);

		if (xcmd->vf)
			snprintf(reg_dump, 100, "\n###\t\tqdmavf%05x, pci %02x:%02x.%02x, reg dump\n",
				xcmd->if_bdf, xcmd->resp.dev_info.pci_bus,
				xcmd->resp.dev_info.pci_dev, xcmd->resp.dev_info.dev_func);
		else
			snprintf(reg_dump, 100, "\n###\t\tqdma%05x, pci %02x:%02x.%02x, reg dump\n",
				xcmd->if_bdf, xcmd->resp.dev_info.pci_bus,
				xcmd->resp.dev_info.pci_dev, xcmd->resp.dev_info.dev_func);
		if (xcmd->log_msg_dump)
			xcmd->log_msg_dump(reg_dump);

		print_seperator(xcmd);
		snprintf(reg_dump, 100, "\nAXI Master Lite Bar #%d\n",
			 xcmd->resp.dev_info.user_bar);
		if (xcmd->log_msg_dump)
			xcmd->log_msg_dump(reg_dump);

		reg_dump_mmap(&xcmd->resp.dev_info, xcmd->resp.dev_info.user_bar,
			      qdma_user_regs, QDMA_USR_BAR_SIZE, xcmd);

		snprintf(reg_dump, 100, "\nCONFIG BAR #%d\n",
			xcmd->resp.dev_info.config_bar);
		if (xcmd->log_msg_dump)
			xcmd->log_msg_dump(reg_dump);
		reg_dump_mmap(&xcmd->resp.dev_info, xcmd->resp.dev_info.config_bar, NULL,
				QDMA_CFG_BAR_SIZE, xcmd);
		break;
	case XNL_CMD_REG_INFO_READ:
		xnl_common_msg_send(xcmd, attrs);
		break;
	default:
		break;
	}

	return 0;
}
