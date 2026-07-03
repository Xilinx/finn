/*
 * This file is part of the QDMA userspace application
 * to enable the user to execute the QDMA functionality
 *
 * Copyright (c) 2019-2022,  Xilinx, Inc. All rights reserved.
 * Copyright (c) 2022-2024,  Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under BSD-style license (found in the
 * LICENSE file in the root directory of this source tree)
 */
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <unistd.h>
#include <err.h>
#include <errno.h>
#include <sys/socket.h>
#include <linux/genetlink.h>
#include "qdma_nl.h"
#include "dmaxfer.h"
#include "dmactl_internal.h"

#define MAX_KMALLOC_SIZE	(4*1024*1024)

//#define DEBUG 1
/*
 * netlink message
 */
struct xnl_hdr {
	struct nlmsghdr n;
	struct genlmsghdr g;
};

struct xnl_cb {
	int fd;
	unsigned short family;
	unsigned int snd_seq;
	unsigned int rcv_seq;
};

struct xnl_gen_msg {
	struct xnl_hdr hdr;
	char data[0];
};

void xnl_close(struct xnl_cb *cb)
{
	close(cb->fd);
}

static int xnl_send(struct xnl_cb *cb, struct xnl_hdr *hdr,
		    void (*log_err)(const char *))
{
	int rv;
	struct sockaddr_nl addr = {
		.nl_family = AF_NETLINK,
	};

	hdr->n.nlmsg_seq = cb->snd_seq;
	cb->snd_seq++;

	rv = sendto(cb->fd, (char *)hdr, hdr->n.nlmsg_len, 0,
			(struct sockaddr *)&addr, sizeof(addr));
	if (rv != hdr->n.nlmsg_len) {
		if (log_err)
			log_err("nl send err");
		return -1;
	}

	return 0;
}

static int xnl_recv(struct xnl_cb *cb, struct xnl_hdr *hdr, int dlen,
		    void (*log_err)(const char *))
{
	int rv;

	memset(hdr, 0, sizeof(struct xnl_gen_msg) + dlen);

	rv = recv(cb->fd, hdr, dlen, 0);
	if (rv < 0) {
		if (log_err)
			log_err("nl recv err");
		return -1;
	}
	/* as long as there is attribute, even if it is shorter than expected */
	if (!NLMSG_OK((&hdr->n), rv) && (rv <= sizeof(struct xnl_hdr))) {
		char err_msg[100] = {'\0'};

		snprintf(err_msg, 100,
			 "nl recv:, invalid message, cmd 0x%x, %d,%d.\n",
			 hdr->g.cmd, dlen, rv);
		if (log_err)
			log_err(err_msg);
		return -1;
	}

	if (hdr->n.nlmsg_type == NLMSG_ERROR) {
		char err_msg[100] = {'\0'};

		snprintf(err_msg, 100, "nl recv, msg error, cmd 0x%x\n",
				hdr->g.cmd);
		if (log_err)
			log_err(err_msg);
		return -1;
	}

	return 0;
}

static inline struct xnl_gen_msg *xnl_msg_alloc(unsigned int dlen,
						void (*log_err)(const char *))
{
	struct xnl_gen_msg *msg;
	unsigned int extra_mem = (XNL_ATTR_MAX * (sizeof(struct nlattr) +
			sizeof(uint32_t)));

	if (dlen)
		extra_mem = dlen;

	msg = malloc(sizeof(struct xnl_gen_msg) + extra_mem);
	if (!msg) {
		char err_msg[100] = {'\0'};

		snprintf(err_msg, 100, "%s: OOM, %u.\n",
			 __FUNCTION__, extra_mem);
		if (log_err)
			log_err(err_msg);
		return NULL;
	}

	memset(msg, 0, sizeof(struct xnl_gen_msg) + extra_mem);
	return msg;
}

static int xnl_connect(struct xnl_cb *cb, int vf, void (*log_err)(const char *))
{
	int fd;
	struct sockaddr_nl addr;
	struct xnl_gen_msg *msg = xnl_msg_alloc(0, log_err);
	struct xnl_hdr *hdr = &msg->hdr;
	struct nlattr *attr;
	int rv = -1;

	fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_GENERIC);
	if (fd < 0) {
		if (log_err)
			log_err("nl socket err");
		rv = fd;
		goto out;
        }
	cb->fd = fd;

	memset(&addr, 0, sizeof(struct sockaddr_nl));
	addr.nl_family = AF_NETLINK;
	rv = bind(fd, (struct sockaddr *)&addr, sizeof(struct sockaddr_nl));
	if (rv < 0) {
		if (log_err)
			log_err("nl bind err");
		goto out;
	}

	hdr->n.nlmsg_type = GENL_ID_CTRL;
	hdr->n.nlmsg_flags = NLM_F_REQUEST;
	hdr->n.nlmsg_pid = getpid();
	hdr->n.nlmsg_len = NLMSG_LENGTH(GENL_HDRLEN);

        hdr->g.cmd = CTRL_CMD_GETFAMILY;
        hdr->g.version = XNL_VERSION;

	attr = (struct nlattr *)(hdr + 1);
	attr->nla_type = CTRL_ATTR_FAMILY_NAME;
	cb->family = CTRL_ATTR_FAMILY_NAME;

	if (vf) {
		attr->nla_len = strlen(XNL_NAME_VF) + 1 + NLA_HDRLEN;
		memcpy((char *)(attr + 1), XNL_NAME_VF, sizeof(XNL_NAME_VF) - 1);
	} else {
		attr->nla_len = strlen(XNL_NAME_PF) + 1 + NLA_HDRLEN;
		memcpy((char *)(attr + 1), XNL_NAME_PF, sizeof(XNL_NAME_PF) - 1);
	}
        hdr->n.nlmsg_len += NLMSG_ALIGN(attr->nla_len);

	rv = xnl_send(cb, hdr, log_err);
	if (rv < 0)
		goto out;

	rv = xnl_recv(cb, hdr, XNL_RESP_BUFLEN_MIN, NULL);
	if (rv < 0)
		goto out;

	attr = (struct nlattr *)((char *)attr + NLA_ALIGN(attr->nla_len));
	/* family ID */
        if (attr->nla_type == CTRL_ATTR_FAMILY_ID)
		cb->family = *(__u16 *)(attr + 1);

	rv = 0;

out:
	free(msg);
	return rv;
}


static void xnl_msg_set_hdr(struct xnl_hdr *hdr, int family, int op)
{
	hdr->n.nlmsg_len = NLMSG_LENGTH(GENL_HDRLEN);
	hdr->n.nlmsg_type = family;
	hdr->n.nlmsg_flags = NLM_F_REQUEST;
	hdr->n.nlmsg_pid = getpid();

	hdr->g.cmd = op;
}

static int xnl_msg_add_int_attr(struct xnl_hdr *hdr, enum xnl_attr_t type,
				unsigned int v)
{
	struct nlattr *attr = (struct nlattr *)((char *)hdr + hdr->n.nlmsg_len);

        attr->nla_type = (__u16)type;
        attr->nla_len = sizeof(__u32) + NLA_HDRLEN;
	*(__u32 *)(attr+ 1) = v;

        hdr->n.nlmsg_len += NLMSG_ALIGN(attr->nla_len);
	return 0;
}

static int xnl_msg_add_str_attr(struct xnl_hdr *hdr, enum xnl_attr_t type,
				char *s)
{
	struct nlattr *attr = (struct nlattr *)((char *)hdr + hdr->n.nlmsg_len);
	int len = strlen(s);

        attr->nla_type = (__u16)type;
        attr->nla_len = len + 1 + NLA_HDRLEN;

	strcpy((char *)(attr + 1), s);

        hdr->n.nlmsg_len += NLMSG_ALIGN(attr->nla_len);
	return 0;
}

static int recv_attrs(struct xnl_hdr *hdr, struct xcmd_info *xcmd,
		      uint32_t *attrs)
{
	unsigned char *p = (unsigned char *)(hdr + 1);
	int maxlen = hdr->n.nlmsg_len - NLMSG_LENGTH(GENL_HDRLEN);

	while (maxlen > 0) {
		struct nlattr *na = (struct nlattr *)p;
		int len = NLA_ALIGN(na->nla_len);

		if (na->nla_type >= XNL_ATTR_MAX) {
			void (*log_err)(const char *) = xcmd->log_msg_dump;
			char err_msg[100] = {'\0'};

			snprintf(err_msg, 100,
				 "unknown attr type %d, len %d.\n",
				 na->nla_type, na->nla_len);
			if (log_err)
				log_err(err_msg);
			return -EINVAL;
		}

		if (na->nla_type == XNL_ATTR_GENMSG) {
			if (xcmd->log_msg_dump)
				xcmd->log_msg_dump((const char *)(na + 1));
		} else {
			attrs[na->nla_type] = *(uint32_t *)(na + 1);
		}

		p += len;
		maxlen -= len;
	}

	return 0;
}

static int get_cmd_resp_buf_len(enum xnl_op_t op, struct xcmd_info *xcmd)
{
	int buf_len = XNL_RESP_BUFLEN_MAX;
	unsigned int row_len = 50;

	switch (op) {
		case XNL_CMD_Q_DESC:
        	row_len *= 2;
        case XNL_CMD_Q_CMPT:
        	buf_len += ((xcmd->req.qparm.range_end -
        			xcmd->req.qparm.range_start)*row_len);
        	break;
        case XNL_CMD_INTR_RING_DUMP:
        	buf_len += ((xcmd->req.intr.end_idx -
				     xcmd->req.intr.start_idx)*row_len);
        	break;
        case XNL_CMD_DEV_LIST:
        case XNL_CMD_DEV_INFO:
        case XNL_CMD_DEV_CAP:
        case XNL_CMD_Q_START:
        case XNL_CMD_Q_STOP:
        case XNL_CMD_Q_DEL:
        case XNL_CMD_GLOBAL_CSR:
            return buf_len;
        case XNL_CMD_Q_ADD:
        case XNL_CMD_Q_DUMP:
        case XNL_CMD_Q_LIST:
        case XNL_CMD_Q_CMPT_READ:
            break;
        case XNL_CMD_REG_DUMP:
		case XNL_CMD_REG_INFO_READ:
            buf_len = XNL_RESP_BUFLEN_MAX * 6;
        break;
		case XNL_CMD_DEV_STAT:
			buf_len = XNL_RESP_BUFLEN_MAX;
		break;
		default:
        	buf_len = XNL_RESP_BUFLEN_MIN;
        	return buf_len;
	}
	if ((xcmd->req.qparm.flags & XNL_F_QDIR_BOTH) == XNL_F_QDIR_BOTH)
		buf_len *= 2;
	if(xcmd->req.qparm.num_q > 1)
			buf_len *= xcmd->req.qparm.num_q;
	if(buf_len > MAX_KMALLOC_SIZE)
		buf_len = MAX_KMALLOC_SIZE;
	return buf_len;
}


static int recv_nl_msg(struct xnl_hdr *hdr, struct xcmd_info *xcmd,
		       uint32_t *attrs)
{
	if (!attrs)
		return 0;
	recv_attrs(hdr, xcmd, attrs);

	if (attrs[XNL_ATTR_ERROR] != 0)
		return (int)attrs[XNL_ATTR_ERROR];

	return 0;
}

static void xnl_msg_add_extra_config_attrs(struct xnl_hdr *hdr,
                                       struct xcmd_info *xcmd)
{
	if (xcmd->req.qparm.sflags & (1 << QPARM_RNGSZ_IDX))
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QRNGSZ_IDX,
		                     xcmd->req.qparm.qrngsz_idx);
	if (xcmd->req.qparm.sflags & (1 << QPARM_C2H_BUFSZ_IDX))
		xnl_msg_add_int_attr(hdr, XNL_ATTR_C2H_BUFSZ_IDX,
		                     xcmd->req.qparm.c2h_bufsz_idx);
	if (xcmd->req.qparm.sflags & (1 << QPARM_CMPTSZ))
		xnl_msg_add_int_attr(hdr,  XNL_ATTR_CMPT_DESC_SIZE,
		                     xcmd->req.qparm.cmpt_entry_size);
	if (xcmd->req.qparm.sflags & (1 << QPARM_SW_DESC_SZ))
		xnl_msg_add_int_attr(hdr,  XNL_ATTR_SW_DESC_SIZE,
		                     xcmd->req.qparm.sw_desc_sz);
	if (xcmd->req.qparm.sflags & (1 << QPARM_CMPT_TMR_IDX))
		xnl_msg_add_int_attr(hdr,  XNL_ATTR_CMPT_TIMER_IDX,
		                     xcmd->req.qparm.cmpt_tmr_idx);
	if (xcmd->req.qparm.sflags & (1 << QPARM_CMPT_CNTR_IDX))
		xnl_msg_add_int_attr(hdr,  XNL_ATTR_CMPT_CNTR_IDX,
		                     xcmd->req.qparm.cmpt_cntr_idx);
	if (xcmd->req.qparm.sflags & (1 << QPARM_MM_CHANNEL))
		xnl_msg_add_int_attr(hdr,  XNL_ATTR_MM_CHANNEL,
		                     xcmd->req.qparm.mm_channel);
	if (xcmd->req.qparm.sflags & (1 << QPARM_CMPT_TRIG_MODE))
		xnl_msg_add_int_attr(hdr,  XNL_ATTR_CMPT_TRIG_MODE,
		                     xcmd->req.qparm.cmpt_trig_mode);
	if (xcmd->req.qparm.sflags & (1 << QPARM_PING_PONG_EN)) {
		xnl_msg_add_int_attr(hdr,  XNL_ATTR_PING_PONG_EN,
							 xcmd->req.qparm.ping_pong_en);
	}
	if (xcmd->req.qparm.sflags & (1 << QPARM_KEYHOLE_EN)) {
		xnl_msg_add_int_attr(hdr,  XNL_ATTR_APERTURE_SZ,
							 xcmd->req.qparm.aperture_sz);
	}
}

static int xnl_parse_response(struct xnl_cb *cb, struct xnl_hdr *hdr,
			      struct xcmd_info *xcmd, unsigned int dlen,
			      uint32_t *attrs)
{
	struct nlattr *attr;
	int rv;

	rv = xnl_recv(cb, hdr, dlen, xcmd->log_msg_dump);
	if (rv < 0)
		goto out;

	rv = recv_nl_msg(hdr, xcmd, attrs);
out:
	return rv;
}

static int xnl_send_cmd(struct xnl_cb *cb, struct xnl_hdr *hdr,
			struct xcmd_info *xcmd, unsigned int dlen)
{
	struct nlattr *attr;
	int rv;

	attr = (struct nlattr *)(hdr + 1);

	xnl_msg_add_int_attr(hdr, XNL_ATTR_DEV_IDX, xcmd->if_bdf);

	switch(xcmd->op) {
        case XNL_CMD_DEV_LIST:
        case XNL_CMD_DEV_INFO:
        case XNL_CMD_DEV_STAT:
        case XNL_CMD_DEV_STAT_CLEAR:
		/* no parameter */
		break;
        case XNL_CMD_Q_LIST:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QIDX, xcmd->req.qparm.idx);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_NUM_Q, xcmd->req.qparm.num_q);
        case XNL_CMD_Q_ADD:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QIDX, xcmd->req.qparm.idx);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_NUM_Q, xcmd->req.qparm.num_q);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QFLAG, xcmd->req.qparm.flags);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_RSP_BUF_LEN, dlen );
		break;
        case XNL_CMD_Q_START:
        	xnl_msg_add_extra_config_attrs(hdr, xcmd);
        case XNL_CMD_Q_STOP:
        case XNL_CMD_Q_DEL:
        case XNL_CMD_Q_DUMP:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QIDX, xcmd->req.qparm.idx);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_NUM_Q, xcmd->req.qparm.num_q);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QFLAG, xcmd->req.qparm.flags);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_RSP_BUF_LEN, dlen);
		break;
        case XNL_CMD_Q_DESC:
        case XNL_CMD_Q_CMPT:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QIDX, xcmd->req.qparm.idx);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_NUM_Q, xcmd->req.qparm.num_q);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QFLAG, xcmd->req.qparm.flags);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_RANGE_START,
					xcmd->req.qparm.range_start);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_RANGE_END,
					xcmd->req.qparm.range_end);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_RSP_BUF_LEN, dlen);
		break;
        case XNL_CMD_Q_RX_PKT:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QIDX, xcmd->req.qparm.idx);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_NUM_Q, xcmd->req.qparm.num_q);
		/* hard coded to C2H */
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QFLAG, XNL_F_QDIR_C2H);
		break;
        case XNL_CMD_Q_CMPT_READ:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QIDX, xcmd->req.qparm.idx);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_QFLAG, xcmd->req.qparm.flags);
		/*xnl_msg_add_int_attr(hdr, XNL_ATTR_NUM_ENTRIES,
					xcmd->u.qparm.num_entries);*/
		break;
        case XNL_CMD_INTR_RING_DUMP:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_INTR_VECTOR_IDX,
		                     xcmd->req.intr.vector);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_INTR_VECTOR_START_IDX,
		                     xcmd->req.intr.start_idx);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_INTR_VECTOR_END_IDX,
		                     xcmd->req.intr.end_idx);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_RSP_BUF_LEN, dlen);
		break;
        case XNL_CMD_REG_RD:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_REG_BAR_NUM,
    							 xcmd->req.reg.bar);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_REG_ADDR,
							 xcmd->req.reg.reg);
		break;
        case XNL_CMD_REG_WRT:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_REG_BAR_NUM,
    							 xcmd->req.reg.bar);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_REG_ADDR,
							 xcmd->req.reg.reg);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_REG_VAL,
							 xcmd->req.reg.val);
		break;
		case XNL_CMD_REG_INFO_READ:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_REG_BAR_NUM,
								 xcmd->req.reg.bar);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_REG_ADDR,
							 xcmd->req.reg.reg);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_NUM_REGS,
							 xcmd->req.reg.range_end);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_RSP_BUF_LEN,
				dlen);
		break;
        case XNL_CMD_REG_DUMP:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_RSP_BUF_LEN,
				dlen);
		break;
		case XNL_CMD_GLOBAL_CSR:
		xnl_msg_add_int_attr(hdr, XNL_ATTR_CSR_INDEX,
				0);
		xnl_msg_add_int_attr(hdr, XNL_ATTR_CSR_COUNT,
				QDMA_GLOBAL_CSR_ARRAY_SZ);
		break;
	default:
		break;
	}

	rv = xnl_send(cb, hdr, xcmd->log_msg_dump);
	if (rv < 0)
		goto out;

out:
	return rv;
}

void xnl_parse_dev_info_attrs(uint32_t *attrs, struct xcmd_info *xcmd)
{
	unsigned int usr_bar;
	struct xnl_dev_info *dev_info = &xcmd->resp.dev_info;

	dev_info->config_bar = attrs[XNL_ATTR_DEV_CFG_BAR];
	usr_bar = (int)attrs[XNL_ATTR_DEV_USR_BAR];
	dev_info->qmax = attrs[XNL_ATTR_DEV_QSET_MAX];
	dev_info->qbase = attrs[XNL_ATTR_DEV_QSET_QBASE];
	dev_info->pci_bus = attrs[XNL_ATTR_PCI_BUS];
	dev_info->pci_dev = attrs[XNL_ATTR_PCI_DEV];
	dev_info->dev_func = attrs[XNL_ATTR_PCI_FUNC];

	if (usr_bar+1 == 0)
		dev_info->user_bar = 2;
	else
		dev_info->user_bar = usr_bar;
}

void xnl_parse_dev_stat_attrs(uint32_t *attrs, struct xcmd_info *xcmd)
{
	unsigned int pkts;
	struct xnl_dev_stat *dev_stat = &xcmd->resp.dev_stat;

	pkts = attrs[XNL_ATTR_DEV_STAT_MMH2C_PKTS1];
	dev_stat->mm_h2c_pkts = pkts;
	pkts = attrs[XNL_ATTR_DEV_STAT_MMH2C_PKTS2];
	dev_stat->mm_h2c_pkts |= (((unsigned long long)pkts) << 32);

	pkts = attrs[XNL_ATTR_DEV_STAT_MMC2H_PKTS1];
	dev_stat->mm_c2h_pkts = pkts;
	pkts = attrs[XNL_ATTR_DEV_STAT_MMC2H_PKTS2];
	dev_stat->mm_c2h_pkts |= (((unsigned long long)pkts) << 32);

	pkts = attrs[XNL_ATTR_DEV_STAT_STH2C_PKTS1];
	dev_stat->st_h2c_pkts = pkts;
	pkts = attrs[XNL_ATTR_DEV_STAT_STH2C_PKTS2];
	dev_stat->st_h2c_pkts |= (((unsigned long long)pkts) << 32);

	pkts = attrs[XNL_ATTR_DEV_STAT_STC2H_PKTS1];
	dev_stat->st_c2h_pkts = pkts;
	pkts = attrs[XNL_ATTR_DEV_STAT_STC2H_PKTS2];
	dev_stat->st_c2h_pkts |= (((unsigned long long)pkts) << 32);

	pkts = attrs[XNL_ATTR_DEV_STAT_PING_PONG_LATMAX1];
	dev_stat->ping_pong_lat_max = pkts;
	pkts = attrs[XNL_ATTR_DEV_STAT_PING_PONG_LATMAX2];
	dev_stat->ping_pong_lat_max |= (((unsigned long long)pkts) << 32);

	pkts = attrs[XNL_ATTR_DEV_STAT_PING_PONG_LATMIN1];
	dev_stat->ping_pong_lat_min = pkts;
	pkts = attrs[XNL_ATTR_DEV_STAT_PING_PONG_LATMIN2];
	dev_stat->ping_pong_lat_min |= (((unsigned long long)pkts) << 32);

	pkts = attrs[XNL_ATTR_DEV_STAT_PING_PONG_LATAVG1];
	dev_stat->ping_pong_lat_avg = pkts;
	pkts = attrs[XNL_ATTR_DEV_STAT_PING_PONG_LATAVG2];
	dev_stat->ping_pong_lat_avg |= (((unsigned long long)pkts) << 32);
}

void xnl_parse_dev_cap_attrs(struct xnl_hdr *hdr, uint32_t *attrs,
				    struct xcmd_info *xcmd)
{
	unsigned char *p = (unsigned char *)(hdr + 1);
	struct nlattr *na = (struct nlattr *)p;

	xcmd->resp.cap.mailbox_en = attrs[XNL_ATTR_DEV_MAILBOX_ENABLE];
	xcmd->resp.cap.mm_channel_max = attrs[XNL_ATTR_DEV_MM_CHANNEL_MAX];
	xcmd->resp.cap.num_pfs = attrs[XNL_ATTR_DEV_NUM_PFS];
	xcmd->resp.cap.num_qs = attrs[XNL_ATTR_DEV_NUMQS];
	xcmd->resp.cap.flr_present = attrs[XNL_ATTR_DEV_FLR_PRESENT];
	xcmd->resp.cap.mm_en = attrs[XNL_ATTR_DEV_MM_ENABLE];
	xcmd->resp.cap.debug_mode = attrs[XNL_ATTR_DEBUG_EN];
	xcmd->resp.cap.desc_eng_mode = attrs[XNL_ATTR_DESC_ENGINE_MODE];
	xcmd->resp.cap.mm_cmpt_en =
			attrs[XNL_ATTR_DEV_MM_CMPT_ENABLE];
	xcmd->resp.cap.st_en = attrs[XNL_ATTR_DEV_ST_ENABLE];
	if (na->nla_type == XNL_ATTR_VERSION_INFO) {
		strncpy(xcmd->resp.cap.version_str, (char *)(na + 1), QDMA_VERSION_INFO_STR_LENGTH);
	}
	if (na->nla_type == XNL_ATTR_DEVICE_TYPE) {
		strncpy(xcmd->resp.cap.version_str, (char *)(na + 1), QDMA_VERSION_INFO_STR_LENGTH);
	}
	if (na->nla_type == XNL_ATTR_IP_TYPE) {
		strncpy(xcmd->resp.cap.version_str, (char *)(na + 1), QDMA_VERSION_INFO_STR_LENGTH);
	}
}

void xnl_parse_reg_attrs(uint32_t *attrs, struct xcmd_info *xcmd)
{
	xcmd->req.reg.val = attrs[XNL_ATTR_REG_VAL];
}

static void xnl_parse_q_state_attrs(uint32_t *attrs, struct xcmd_info *xcmd)
{
	xcmd->resp.q_info.flags = attrs[XNL_ATTR_QFLAG];
	xcmd->resp.q_info.qidx = attrs[XNL_ATTR_QIDX];
	xcmd->resp.q_info.state = attrs[XNL_ATTR_Q_STATE];
}

static void xnl_parse_csr_attrs(struct xnl_hdr *hdr, uint32_t *attrs, struct xcmd_info *xcmd)
{
	unsigned char *p = (unsigned char *)(hdr + 1);
	struct nlattr *na = (struct nlattr *)p;

	if (na->nla_type == XNL_ATTR_GLOBAL_CSR) {
		memcpy(&xcmd->resp.csr,(void *) (na + 1),
			   sizeof(struct global_csr_conf));

	}

}

static void xnl_parse_cmd_attrs(struct xnl_hdr *hdr, struct xcmd_info *xcmd,
				uint32_t *attrs)
{
	switch(xcmd->op) {
        case XNL_CMD_DEV_INFO:
        	xnl_parse_dev_info_attrs(attrs, xcmd);
		break;
        case XNL_CMD_DEV_STAT:
        	xnl_parse_dev_stat_attrs(attrs, xcmd);
		break;
        case XNL_CMD_DEV_CAP:
        	xnl_parse_dev_cap_attrs(hdr, attrs, xcmd);
		break;
        case XNL_CMD_REG_RD:
        case XNL_CMD_REG_WRT:
        	xnl_parse_reg_attrs(attrs, xcmd);
		break;
        case XNL_CMD_GET_Q_STATE:
        	xnl_parse_q_state_attrs(attrs, xcmd);
		break;
        case XNL_CMD_GLOBAL_CSR:
		xnl_parse_csr_attrs(hdr, attrs, xcmd);
		break;
	default:
		break;
	}
}

int xnl_common_msg_send(struct xcmd_info *cmd, uint32_t *attrs)
{
	int rv;
	struct xnl_gen_msg *msg = NULL;
	struct xnl_hdr *hdr;
	struct xnl_cb cb;
	unsigned int dlen;

	rv = xnl_connect(&cb, cmd->vf, cmd->log_msg_dump);
	if (rv < 0)
		return rv;
	dlen = get_cmd_resp_buf_len(cmd->op, cmd);
	msg = xnl_msg_alloc(dlen, cmd->log_msg_dump);
	if (!msg)
		goto close;
	hdr = &msg->hdr;

	xnl_msg_set_hdr(hdr, cb.family, cmd->op);

	rv = xnl_send_cmd(&cb, hdr, cmd, dlen);
	if (rv < 0)
		goto free_mem;

	rv = xnl_parse_response(&cb, hdr, cmd, dlen, attrs);
	if (rv < 0)
		goto free_mem;
	xnl_parse_cmd_attrs(hdr, cmd, attrs);
free_mem:
	free(msg);
close:
	xnl_close(&cb);

	return rv;
}

int qdma_dev_list_dump(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	xnl_common_msg_send(cmd, attrs);
	cmd->vf = 1;
	return xnl_common_msg_send(cmd, attrs);
}

int qdma_dev_info(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};
	int rv;

	rv = xnl_common_msg_send(cmd, attrs);
	if (rv < 0)
		return rv;
}

int qdma_dev_cap(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};
	int rv;

	rv = xnl_common_msg_send(cmd, attrs);
	if (rv < 0)
		return rv;
}

int qdma_dev_stat(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};
	int rv;

	rv = xnl_common_msg_send(cmd, attrs);
	if (rv < 0)
		return rv;
}

int qdma_dev_stat_clear(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}

int qdma_dev_intr_ring_dump(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}

int qdma_dev_get_global_csr(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}

int qdma_dev_q_list_dump(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}

int qdma_q_add(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}

int qdma_q_del(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}

int qdma_q_start(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};
	int rv = 0;
	if ((cmd->req.qparm.flags & XNL_F_QDIR_BOTH) ==
		XNL_F_QDIR_BOTH) {

		if ((cmd->req.qparm.fetch_credit &
			Q_ENABLE_C2H_FETCH_CREDIT) == Q_ENABLE_C2H_FETCH_CREDIT)
			cmd->req.qparm.flags |= XNL_F_FETCH_CREDIT;
		else
			cmd->req.qparm.flags &= ~XNL_F_FETCH_CREDIT;

		cmd->req.qparm.flags = ((cmd->req.qparm.flags &
						(~XNL_F_QDIR_BOTH)) | XNL_F_QDIR_C2H);

		rv = xnl_common_msg_send(cmd, attrs);
		if (rv < 0)
			return rv;

		if ((cmd->req.qparm.fetch_credit &
			Q_ENABLE_H2C_FETCH_CREDIT) == Q_ENABLE_H2C_FETCH_CREDIT)
			cmd->req.qparm.flags |= XNL_F_FETCH_CREDIT;
		else
			cmd->req.qparm.flags &= ~XNL_F_FETCH_CREDIT;

		cmd->req.qparm.flags = ((cmd->req.qparm.flags &
						(~XNL_F_QDIR_BOTH)) | XNL_F_QDIR_H2C);

		return xnl_common_msg_send(cmd, attrs);
	} else {
		if ((cmd->req.qparm.flags & XNL_F_QDIR_H2C) ==
			XNL_F_QDIR_H2C) {
			if ((cmd->req.qparm.fetch_credit &
				Q_ENABLE_H2C_FETCH_CREDIT) == Q_ENABLE_H2C_FETCH_CREDIT)
				cmd->req.qparm.flags |= XNL_F_FETCH_CREDIT;
			else
				cmd->req.qparm.flags &= ~XNL_F_FETCH_CREDIT;
		} else if ((cmd->req.qparm.flags & XNL_F_QDIR_C2H) ==
			XNL_F_QDIR_C2H) {
			if ((cmd->req.qparm.fetch_credit &
				Q_ENABLE_C2H_FETCH_CREDIT) == Q_ENABLE_C2H_FETCH_CREDIT)
					cmd->req.qparm.flags |= XNL_F_FETCH_CREDIT;
			else
				cmd->req.qparm.flags &= ~XNL_F_FETCH_CREDIT;
		}
		return xnl_common_msg_send(cmd, attrs);
	}
}

int qdma_q_stop(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}

int qdma_q_dump(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}

int qdma_q_get_state(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}


int qdma_q_desc_dump(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}

int qdma_q_cmpt_read(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};

	return xnl_common_msg_send(cmd, attrs);
}

int qdma_reg_read(struct xcmd_info *cmd)
{
	return proc_reg_cmd(cmd);
}

int qdma_reg_write(struct xcmd_info *cmd)
{
	return proc_reg_cmd(cmd);
}

int qdma_reg_info_read(struct xcmd_info *cmd)
{
	return proc_reg_cmd(cmd);
}

int qdma_reg_dump(struct xcmd_info *cmd)
{
	return proc_reg_cmd(cmd);
}

#ifdef TANDEM_BOOT_SUPPORTED
int qdma_en_st(struct xcmd_info *cmd)
{
	uint32_t attrs[XNL_ATTR_MAX] = {0};
	int rv;

	rv = xnl_common_msg_send(cmd, attrs);
	if (rv < 0)
		return rv;
}
#endif
