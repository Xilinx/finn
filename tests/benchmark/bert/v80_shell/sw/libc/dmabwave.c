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

#include "dmabwave.h"

struct queue_info *q_info;
int q_count;

// Queue util
// -------------------------------------------------------

static unsigned int num_trailing_blanks(char *word)
{
	unsigned int i = 0;
	unsigned int slen = strlen(word);

	if (!slen) return 0;
	while (isspace(word[slen - i - 1])) {
		i++;
	}

	return i;
}

static char * strip_blanks(char *word, long unsigned int *banlks)
{
	char *p = word;
	unsigned int i = 0;

	while (isblank(p[0])) {
		p++;
		i++;
	}
	if (banlks)
		*banlks = i;

	return p;
}

static unsigned int copy_value(char *src, char *dst, unsigned int max_len)
{
	char *p = src;
	unsigned int i = 0;

	while (max_len && !isspace(p[0])) {
		dst[i] = p[0];
		p++;
		i++;
		max_len--;
	}

	return i;
}

static char * strip_comments(char *word)
{
	size_t numblanks;
	char *p = strip_blanks(word, &numblanks);

	if (p[0] == '#')
		return NULL;
	else
		p = strtok(word, "#");

	return p;
}

static int arg_read_int(char *s, uint32_t *v)
{
	char *p = NULL;


	*v = strtoul(s, &p, 0);
	if (*p && (*p != '\n') && !isblank(*p)) {
		printf("Error:something not right%s %s %s",
				s, p, isblank(*p)? "true": "false");
		return -EINVAL;
	}
	return 0;
}

static int parse_config_file(const char *cfg_fname)
{
	char *linebuf = NULL;
	char *realbuf;
	FILE *fp;
	size_t linelen = 0;
	size_t numread;
	size_t numblanks;
	unsigned int linenum = 0;
	char *config, *value;
	unsigned int dir_factor = 1;
	char rng_sz[100] = {'\0'};
	char rng_sz_path[200] = {'\0'};
	int rng_sz_fd, ret = 0;
	struct stat st;

	fp = fopen(cfg_fname, "r");
	if (fp == NULL) {
		printf("Failed to open Config File [%s]\n", cfg_fname);
		return -EINVAL;
	}

	while ((numread = getline(&linebuf, &linelen, fp)) != -1) {
		numread--;
		linenum++;
		linebuf = strip_comments(linebuf);
		if (linebuf == NULL)
			continue;
		realbuf = strip_blanks(linebuf, &numblanks);
		linelen -= numblanks;
		if (0 == linelen)
			continue;
		config = strtok(realbuf, "=");
		value = strtok(NULL, "=");
		if (!strncmp(config, "mode", 4)) {
			if (!strncmp(value, "mm", 2))
				mode = QDMA_Q_MODE_MM;
			else if(!strncmp(value, "st", 2))
				mode = QDMA_Q_MODE_ST;
			else {
				printf("Error: Unknown mode\n");
				goto prase_cleanup;
			}
		} else if (!strncmp(config, "dir", 3)) {
			if (!strncmp(value, "h2c", 3))
				dir = QDMA_Q_DIR_H2C;
			else if(!strncmp(value, "c2h", 3))
				dir = QDMA_Q_DIR_C2H;
			else if(!strncmp(value, "bi", 2))
				dir = QDMA_Q_DIR_BIDI;
			else if(!strncmp(value, "cmpt", 4)) {
				printf("Error: cmpt type queue validation is not supported\n");
				goto prase_cleanup;
			} else {
				printf("Error: Unknown direction\n");
				goto prase_cleanup;
			}
		} else if (!strncmp(config, "name", 3)) {
			copy_value(value, cfg_name, 64);
		} else if (!strncmp(config, "function", 8)) {
			if (arg_read_int(value, &fun_id)) {
				printf("Error: Invalid function:%s\n", value);
				goto prase_cleanup;
			}
		} else if (!strncmp(config, "is_vf", 5)) {
			if (arg_read_int(value, &is_vf)) {
				printf("Error: Invalid is_vf param:%s\n", value);
				goto prase_cleanup;
			}
			if (is_vf > 1) {
				printf("Error: is_vf value is %d, expected 0 or 1\n",
						is_vf);
				goto prase_cleanup;
			}
		} else if (!strncmp(config, "q_range", 7)) {
			char *q_range_start = strtok(value, ":");
			char *q_range_end = strtok(NULL, ":");
			unsigned int start;
			unsigned int end;
			if (arg_read_int(q_range_start, &start)) {
				printf("Error: Invalid q range start:%s\n", q_range_start);
				goto prase_cleanup;
			}
			if (arg_read_int(q_range_end, &end)) {
				printf("Error: Invalid q range end:%s\n", q_range_end);
				goto prase_cleanup;
			}

			q_start = start;
			num_q = end - start + 1;
		} else if (!strncmp(config, "rngidx", 6)) {
			if (arg_read_int(value, &idx_rngsz)) {
				printf("Error: Invalid idx_rngsz:%s\n", value);
				goto prase_cleanup;
			}
		} else if (!strncmp(config, "tmr_idx", 7)) {
			if (arg_read_int(value, &idx_tmr)) {
				printf("Error: Invalid idx_tmr:%s\n", value);
				goto prase_cleanup;
			}
		}
		if (!strncmp(config, "cntr_idx", 8)) {
			if (arg_read_int(value, &idx_cnt)) {
				printf("Error: Invalid idx_cnt:%s\n", value);
				goto prase_cleanup;
			}
		} else if (!strncmp(config, "pfetch_en", 9)) {
			if (arg_read_int(value, &pfetch_en)) {
				printf("Error: Invalid pfetch_en:%s\n", value);
				goto prase_cleanup;
			}
		} else if (!strncmp(config, "cmptsz", 5)) {
			if (arg_read_int(value, &cmptsz)) {
				printf("Error: Invalid cmptsz:%s\n", value);
				goto prase_cleanup;
			}
		}  else if (!strncmp(config, "trig_mode", 9)) {
			copy_value(value, trigmode_str, 10);
		}  else if (!strncmp(config, "pkt_sz", 6)) {
			if (arg_read_int(value, &pkt_sz)) {
				printf("Error: Invalid pkt_sz:%s\n", value);
				goto prase_cleanup;
			}
		} else if (!strncmp(config, "pci_bus", 7)) {
			char *p;

			pci_bus = strtoul(value, &p, 16);
			if (*p && (*p != '\n')) {
				printf("Error: bad parameter \"%s\", integer expected", value);
				goto prase_cleanup;
			}
		} else if (!strncmp(config, "pci_dev", 7)) {
			char *p;

			pci_dev = strtoul(value, &p, 16);
			if (*p && (*p != '\n')) {
				printf("Error: bad parameter \"%s\", integer expected", value);
				goto prase_cleanup;
			}
		} else if (!strncmp(config, "io_type", 6)) {
			if (!strncmp(value, "io_sync", 6))
				io_type = 0;
			else if (!strncmp(value, "io_async", 6))
				io_type = 1;
			else {
				printf("Error: Unknown io_type\n");
				goto prase_cleanup;
			}
		}
	}
	fclose(fp);

	if (!pci_bus && !pci_dev) {
		printf("Error: PCI bus information not provided\n");
		return -EINVAL;
	}

	if (fun_id < 0) {
		printf("Error: Valid function required\n");
		return -EINVAL;
	}

	if (fun_id <= 3 && is_vf) {
		printf("Error: invalid is_vf and fun_id values."
				"Fun_id for vf must be higer than 3\n");
		return -EINVAL;
	}

	if (mode == QDMA_Q_MODE_ST && pkt_sz > QDMA_ST_MAX_PKT_SIZE) {
		printf("Error: Pkt size [%u] larger than supported size [%d]\n",
				pkt_sz, QDMA_ST_MAX_PKT_SIZE);
		return -EINVAL;
	}

	if (!strcmp(trigmode_str, "every"))
		trig_mode = 1;
	else if (!strcmp(trigmode_str, "usr_cnt"))
		trig_mode = 2;
	else if (!strcmp(trigmode_str, "usr"))
		trig_mode = 3;
	else if (!strcmp(trigmode_str, "usr_tmr"))
		trig_mode=4;
	else if (!strcmp(trigmode_str, "cntr_tmr"))
		trig_mode=5;
	else if (!strcmp(trigmode_str, "dis"))
		trig_mode = 0;
	else {
		printf("Error: unknown q trigmode %s.\n", trigmode_str);
		return -EINVAL;
	}

	snprintf(pci_resource, sizeof(pci_resource), "/sys/bus/pci/devices/0000:%02x:00.0/resource0", pci_bus);

	return 0;

prase_cleanup:
	fclose(fp);
	return -EINVAL;
}

static inline void qdma_q_prep_name(struct queue_info *q_info, int qid, int pf)
{
	q_info->q_name = calloc(QDMA_Q_NAME_LEN, 1);
	snprintf(q_info->q_name, QDMA_Q_NAME_LEN, "/dev/qdma%s%05x-%s-%d",
			(is_vf) ? "vf" : "",
			(pci_bus << 12) | (pci_dev << 4) | pf,
			(mode == QDMA_Q_MODE_MM) ? "MM" : "ST",
			qid);
}

static int qdma_validate_qrange(void)
{
	struct xcmd_info xcmd;
	int ret;

	memset(&xcmd, 0, sizeof(struct xcmd_info));
	xcmd.op = XNL_CMD_DEV_INFO;
	xcmd.vf = is_vf;
	xcmd.if_bdf = (pci_bus << 12) | (pci_dev << 4) | fun_id;

	/* Get dev info from qdma driver */
	ret = qdma_dev_info(&xcmd);
	if (ret < 0) {
		printf("Failed to read qmax for PF: %d\n", fun_id);
		return ret;
	}

	if (!xcmd.resp.dev_info.qmax) {
		printf("Error: invalid qmax assigned to function :%d qmax :%u\n",
				fun_id, xcmd.resp.dev_info.qmax);
		return -EINVAL;
	}

	if (xcmd.resp.dev_info.qmax <  num_q) {
		printf("Error: Q Range is beyond QMAX %u "
				"Funtion: %x Q start :%u Q Range End :%u\n",
				xcmd.resp.dev_info.qmax, fun_id, q_start, q_start + num_q);
		return -EINVAL;
	}

	return 0;
}

static int qdma_prepare_q_stop(struct xcmd_info *xcmd,
		enum qdmautils_io_dir dir,
		int qid, int pf)
{
	struct xcmd_q_parm *qparm;

	if (!xcmd)
		return -EINVAL;

	qparm = &xcmd->req.qparm;

	xcmd->op = XNL_CMD_Q_STOP;
	xcmd->vf = is_vf;
	xcmd->if_bdf = (pci_bus << 12) | (pci_dev << 4) | pf;
	qparm->idx = qid;
	qparm->num_q = 1;

	if (mode == QDMA_Q_MODE_MM)
		qparm->flags |= XNL_F_QMODE_MM;
	else if (mode == QDMA_Q_MODE_ST)
		qparm->flags |= XNL_F_QMODE_ST;
	else
		return -EINVAL;

	if (dir == DMAXFER_IO_WRITE)
		qparm->flags |= XNL_F_QDIR_H2C;
	else if (dir == DMAXFER_IO_READ)
		qparm->flags |= XNL_F_QDIR_C2H;
	else
		return -EINVAL;


	return 0;
}

static int qdma_prepare_q_start(struct xcmd_info *xcmd,
		enum qdmautils_io_dir dir,
		int qid, int pf)
{
	struct xcmd_q_parm *qparm;


	if (!xcmd) {
		printf("Error: Invalid Input Param\n");
		return -EINVAL;
	}
	qparm = &xcmd->req.qparm;

	xcmd->op = XNL_CMD_Q_START;
	xcmd->vf = is_vf;
	xcmd->if_bdf = (pci_bus << 12) | (pci_dev << 4) | pf;
	qparm->idx = qid;
	qparm->num_q = 1;

	if (mode == QDMA_Q_MODE_MM)
		qparm->flags |= XNL_F_QMODE_MM;
	else if (mode == QDMA_Q_MODE_ST)
		qparm->flags |= XNL_F_QMODE_ST;
	else {
		printf("Error: Invalid mode\n");
		return -EINVAL;
	}

	if (dir == DMAXFER_IO_WRITE)
		qparm->flags |= XNL_F_QDIR_H2C;
	else if (dir == DMAXFER_IO_READ)
		qparm->flags |= XNL_F_QDIR_C2H;
	else {
		printf("Error: Invalid Direction\n");
		return -EINVAL;
	}

	qparm->qrngsz_idx = idx_rngsz;

	if ((dir == QDMA_Q_DIR_C2H) && (mode == QDMA_Q_MODE_ST)) {
		if (cmptsz)
			qparm->cmpt_entry_size = cmptsz;
		else
			qparm->cmpt_entry_size = XNL_ST_C2H_CMPT_DESC_SIZE_8B;
		qparm->cmpt_tmr_idx = idx_tmr;
		qparm->cmpt_cntr_idx = idx_cnt;
		qparm->cmpt_trig_mode = trig_mode;
		if (pfetch_en)
			qparm->flags |= XNL_F_PFETCH_EN;
	}

	qparm->flags |= (XNL_F_CMPL_STATUS_EN | XNL_F_CMPL_STATUS_ACC_EN |
			XNL_F_CMPL_STATUS_PEND_CHK | XNL_F_CMPL_STATUS_DESC_EN |
			XNL_F_FETCH_CREDIT);

	return 0;
}

static int qdma_prepare_q_del(struct xcmd_info *xcmd,
		enum qdmautils_io_dir dir,
		int qid, int pf)
{
	struct xcmd_q_parm *qparm;

	if (!xcmd) {
		printf("Error: Invalid Input Param\n");
		return -EINVAL;
	}

	qparm = &xcmd->req.qparm;

	xcmd->op = XNL_CMD_Q_DEL;
	xcmd->vf = is_vf;
	xcmd->if_bdf = (pci_bus << 12) | (pci_dev << 4) | pf;
	qparm->idx = qid;
	qparm->num_q = 1;

	if (mode == QDMA_Q_MODE_MM)
		qparm->flags |= XNL_F_QMODE_MM;
	else if (mode == QDMA_Q_MODE_ST)
		qparm->flags |= XNL_F_QMODE_ST;
	else {
		printf("Error: Invalid mode\n");
		return -EINVAL;
	}

	if (dir == DMAXFER_IO_WRITE)
		qparm->flags |= XNL_F_QDIR_H2C;
	else if (dir == DMAXFER_IO_READ)
		qparm->flags |= XNL_F_QDIR_C2H;
	else {
		printf("Error: Invalid Direction\n");
		return -EINVAL;
	}

	return 0;
}

static int qdma_prepare_q_add(struct xcmd_info *xcmd,
		enum qdmautils_io_dir dir,
		int qid, int pf)
{
	struct xcmd_q_parm *qparm;

	if (!xcmd) {
		printf("Error: Invalid Input Param\n");
		return -EINVAL;
	}

	qparm = &xcmd->req.qparm;

	xcmd->op = XNL_CMD_Q_ADD;
	xcmd->vf = is_vf;
	xcmd->if_bdf = (pci_bus << 12) | (pci_dev << 4) | pf;
	qparm->idx = qid;
	qparm->num_q = 1;

	if (mode == QDMA_Q_MODE_MM)
		qparm->flags |= XNL_F_QMODE_MM;
	else if (mode == QDMA_Q_MODE_ST)
		qparm->flags |= XNL_F_QMODE_ST;
	else {
		printf("Error: Invalid mode\n");
		return -EINVAL;
	}
	if (dir == DMAXFER_IO_WRITE)
		qparm->flags |= XNL_F_QDIR_H2C;
	else if (dir == DMAXFER_IO_READ)
		qparm->flags |= XNL_F_QDIR_C2H;
	else {
		printf("Error: Invalid Direction\n");
		return -EINVAL;
	}
	qparm->sflags = qparm->flags;

	return 0;
}

static int qdma_destroy_queue(enum qdmautils_io_dir dir,
		int qid, int pf)
{
	struct xcmd_info xcmd;
	int ret;

	memset(&xcmd, 0, sizeof(struct xcmd_info));
	ret = qdma_prepare_q_stop(&xcmd, dir, qid, pf);
	if (ret < 0)
		return ret;

	ret = qdma_q_stop(&xcmd);
	if (ret < 0)
		printf("Q_STOP failed, ret :%d\n", ret);

	memset(&xcmd, 0, sizeof(struct xcmd_info));
	qdma_prepare_q_del(&xcmd, dir, qid, pf);
	ret = qdma_q_del(&xcmd);
	if (ret < 0)
		printf("Q_DEL failed, ret :%d\n", ret);

	return ret;
}

static int qdma_create_queue(enum qdmautils_io_dir dir,
		int qid, int pf)
{
	struct xcmd_info xcmd;
	int ret;

	memset(&xcmd, 0, sizeof(struct xcmd_info));
	ret = qdma_prepare_q_add(&xcmd, dir, qid, pf);
	if (ret < 0)
		return ret;

	ret = qdma_q_add(&xcmd);
	if (ret < 0) {
		printf("Q_ADD failed, ret :%d\n", ret);
		return ret;
	}

	memset(&xcmd, 0, sizeof(struct xcmd_info));
	ret = qdma_prepare_q_start(&xcmd, dir, qid, pf);
	if (ret < 0)
		return ret;

	ret = qdma_q_start(&xcmd);
	if (ret < 0) {
		printf("Q_START failed, ret :%d\n", ret);
		qdma_prepare_q_del(&xcmd, dir, qid, pf);
		qdma_q_del(&xcmd);
	}

	return ret;
}

static int qdma_prepare_queue(struct queue_info *q_info,
		enum qdmautils_io_dir dir, int qid, int pf)
{
	int ret;

	if (!q_info) {
		printf("Error: Invalid queue info\n");
		return -EINVAL;
	}

	qdma_q_prep_name(q_info, qid, pf);
	q_info->dir = dir;
	ret = qdma_create_queue(q_info->dir, qid, pf);
	if (ret < 0) {
		printf("Q creation Failed PF:%d QID:%d\n",
				pf, qid);
		return ret;
	}
	q_info->qid = qid;
	q_info->pf = pf;

	return ret;
}

static void qdma_queues_cleanup(struct queue_info *q_info, int q_count)
{
	unsigned int q_index;

	if (!q_info || q_count < 0)
		return;

	for (q_index = 0; q_index < q_count; q_index++) {
		qdma_destroy_queue(q_info[q_index].dir,
				q_info[q_index].qid,
				q_info[q_index].pf);
		free(q_info[q_index].q_name);
	}
}

static int qdma_setup_queues(struct queue_info **pq_info)
{
	struct queue_info *q_info;
	unsigned int qid;
	unsigned int q_count;
	unsigned int q_index;
	int ret;

	if (!pq_info) {
		printf("Error: Invalid queue info\n");
		return -EINVAL;
	}

	if (dir == QDMA_Q_DIR_BIDI)
		q_count = num_q * 2;
	else
		q_count = num_q;

	*pq_info = q_info = (struct queue_info *)calloc(q_count, sizeof(struct queue_info));
	if (!q_info) {
		printf("Error: OOM\n");
		return -ENOMEM;
	}

	q_index = 0;
	for (qid = 0; qid < num_q; qid++) {
		if ((dir == QDMA_Q_DIR_BIDI) ||
				(dir == QDMA_Q_DIR_H2C)) {
			ret = qdma_prepare_queue(q_info + q_index,
					DMAXFER_IO_WRITE,
					qid + q_start,
					fun_id);
			if (ret < 0)
				break;
			q_index++;
		}
		if ((dir == QDMA_Q_DIR_BIDI) ||
				(dir == QDMA_Q_DIR_C2H)) {
			ret = qdma_prepare_queue(q_info + q_index,
					DMAXFER_IO_READ,
					qid + q_start,
					fun_id);
			if (ret < 0)
				break;
			q_index++;
		}
	}
	if (ret < 0) {
		qdma_queues_cleanup(q_info, q_index);
		return ret;
	}

	for(int i = 0; i < q_count; i++) {
		printf("Queue %s: qid %d, pf %d\n", q_info[i].q_name, q_info[i].qid, q_info[i].pf);
	}

	return q_count;
}


static void qdma_env_cleanup()
{
	qdma_queues_cleanup(q_info, q_count);

	if (q_info)
		free(q_info);
	q_info = NULL;
	q_count = 0;
}

static void csr_env_cleanup()
{
	//printf("DBG:  csr cleanup\n");
	close(pci_fd);
}

int setup_qdma(const char *cfg_fname)
{
	int ret = 0;

	ret = parse_config_file(cfg_fname);
	if(ret < 0) {
		printf("ERR:  config File has invalid parameters.\n");
		return ret;
	}

	ret = qdma_validate_qrange();
	if(ret < 0) {
		printf("ERR:  queue range invalid.\n");
		return ret;
	}

	q_count = 0;
	q_count = qdma_setup_queues(&q_info);
	if (q_count < 0) {
		printf("ERR:  qdma_setup_queues failed, ret: %d\n", q_count);
		return q_count;
	}

	//atexit(qdma_env_cleanup);

	return ret;
}

void clean_qdma () 
{
	qdma_env_cleanup();
	csr_env_cleanup();
}

static ssize_t write_from_buffer(int fd,  char *buffer, uint64_t size,
			uint64_t base)
{
	ssize_t rc;
	uint64_t count = 0;
	char *buf = buffer;
	off_t offset = base;

	do { /* Support zero byte transfer */
		uint64_t bytes = size - count;

		if (bytes > RW_MAX_SIZE)
			bytes = RW_MAX_SIZE;

		if (offset) {
			rc = lseek(fd, offset, SEEK_SET);
			if (rc < 0) {
				fprintf(stderr,
					"seek off 0x%lx failed %zd.\n",
					offset, rc);
				perror("seek file");
				return -EIO;
			}
			if (rc != offset) {
				fprintf(stderr,
					"seek off 0x%lx != 0x%lx.\n",
					rc, offset);
				return -EIO;
			}
		}

		/* write data to file from memory buffer */
		rc = write(fd, buf, bytes);
		if (rc < 0) {
			fprintf(stderr, "W off 0x%lx, 0x%lx failed %zd.\n",
				offset, bytes, rc);
			perror("write file");
			return -EIO;
		}
		if (rc != bytes) {
			fprintf(stderr, "W off 0x%lx, 0x%lx != 0x%lx.\n",
				offset, rc, bytes);
			return -EIO;
		}

		count += bytes;
		buf += bytes;
		offset += bytes;
	} while (count < size);

	if (count != size) {
		fprintf(stderr, "R failed 0x%lx != 0x%lx.\n",
				count, size);
		return -EIO;
	}

	return count;
}

static ssize_t read_to_buffer(int fd, char *buffer, uint64_t size,
			uint64_t base)
{
	ssize_t rc;
	uint64_t count = 0;
	char *buf = buffer;
	off_t offset = base;

	do { /* Support zero byte transfer */
		uint64_t bytes = size - count;

		if (bytes > RW_MAX_SIZE)
			bytes = RW_MAX_SIZE;

		if (offset) {
			rc = lseek(fd, offset, SEEK_SET);
			if (rc < 0) {
				fprintf(stderr,
					"seek off 0x%lx failed %zd.\n",
					offset, rc);
				perror("seek file");
				return -EIO;
			}
			if (rc != offset) {
				fprintf(stderr,
					"seek off 0x%lx != 0x%lx.\n",
					rc, offset);
				return -EIO;
			}
		}

		/* read data from file into memory buffer */
		rc = read(fd, buf, bytes);
		if (rc < 0) {
			fprintf(stderr,
				"read off 0x%lx + 0x%lx failed %zd.\n",
				offset, bytes, rc);
			perror("read file");
			printf("Error: %s (errno: %d)\n", strerror(errno), errno);
			return -EIO;
		}
		if (rc != bytes) {
			fprintf(stderr,
				"R off 0x%lx, 0x%lx != 0x%lx.\n",
				count, rc, bytes);
			return -EIO;
		}

		count += bytes;
		buf += bytes;
		offset += bytes;
	} while (count < size);

	if (count != size) {
		fprintf(stderr, "R failed 0x%lx != 0x%lx.\n",
				count, size);
		return -EIO;
	}

	return count;
}

volatile uint64_t* map_csr() 
{

    pci_fd = open(pci_resource, O_RDWR);
    if (pci_fd < 0) {
        perror("ERR: open device.");
		return NULL;
    }

	volatile uint64_t *bar = mmap(NULL, CSRBW_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, pci_fd, CSRBW_OFFS);
	if(bar == NULL) {
		perror("ERR: mmap bar.");
		return NULL;
	}

	//atexit(csr_env_cleanup);

    return bar;
}

int dma_xfer(char *qname, char *buffer, size_t size, uint64_t base, int8_t dir) 
{
    
    unsigned int flags;
    ssize_t count;
    int fd;
    
    if(!qname || !buffer || size == 0) {
        printf("ERR:  invalid arguments.\n");
        return -EINVAL;
    }

    if(dir == H2C_TRANSFER) {
        flags = O_WRONLY | O_NONBLOCK;
    } else {
        flags = O_RDONLY | O_NONBLOCK;
    }

    fd = open(qname, flags);
    if(fd < 0) {
		printf("ERR:  device cannot be opened.\n");
        return fd;
    }

	if(dir == H2C_TRANSFER) { 
		count = write_from_buffer(fd, buffer, size, base);
	} else { // read
		count = read_to_buffer(fd, buffer, size, base);
	}

    close(fd);

    return 0;
}

int read_io_ref(char *filename, int32_t *buffer, size_t buffer_size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open file");
        return -1;
    }

    size_t count = 0;
    int32_t value;
    char line[256];

	// Size of the ref
    while (fgets(line, sizeof(line), file)) {
        count++;
    }

    // Check if the provided buffer size is sufficient
    if (buffer_size < count) {
        fprintf(stderr, "Buffer size is too small. Need %zu, but given %zu.\n", count, buffer_size);
        fclose(file);
        return -1;
    }

    // Read
    rewind(file);
    size_t index = 0;
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = '\0';

        if (sscanf(line, "%x", &value) != 1) {
            fprintf(stderr, "Error reading hex value from line: %s\n", line);
            fclose(file);
            return -1;
        }

        buffer[index++] = value;
    }

    fclose(file);
    return 0;
}

float bits_to_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(float));
    return value;
}

int compare_floats(float a, float b, float atol) {
    return fabsf(a - b) <= atol;
}

void fill_random(void *buffer, size_t size) {
    unsigned char *byte_buffer = (unsigned char *)buffer;

    srand((unsigned int)time(NULL));

    for (size_t i = 0; i < size; i++) {
        byte_buffer[i] = (unsigned char)(rand() % 256); // Random value between 0 and 255
    }
}

static int timespec_check(struct timespec *t)
{
	if ((t->tv_nsec < 0) || (t->tv_nsec >= 1000000000))
		return -1;
	return 0;

}

void timespec_sub(struct timespec *t1, struct timespec *t2)
{
	if (timespec_check(t1) < 0) {
		fprintf(stderr, "invalid time #1: %lld.%.9ld.\n",
			(long long)t1->tv_sec, t1->tv_nsec);
		return;
	}
	if (timespec_check(t2) < 0) {
		fprintf(stderr, "invalid time #2: %lld.%.9ld.\n",
			(long long)t2->tv_sec, t2->tv_nsec);
		return;
	}
	t1->tv_sec -= t2->tv_sec;
	t1->tv_nsec -= t2->tv_nsec;
	if (t1->tv_nsec >= 1000000000) {
		t1->tv_sec++;
		t1->tv_nsec -= 1000000000;
	} else if (t1->tv_nsec < 0) {
		t1->tv_sec--;
		t1->tv_nsec += 1000000000;
	}
}