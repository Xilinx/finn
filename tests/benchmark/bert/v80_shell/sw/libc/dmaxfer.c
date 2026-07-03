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

#include <semaphore.h>
#include <time.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <stdbool.h>
#include <linux/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <signal.h>
#include <stddef.h>
#include <errno.h>
#include <error.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include </usr/include/pthread.h>
#include <libaio.h>
#include <sys/sysinfo.h>
#include <sys/uio.h>
#include "dmaxfer.h"

#define SEC2NSEC           1000000000
#define SEC2USEC           1000000

#define DATA_VALIDATION 0
#define MSEC2NSEC 1000000

struct dmaxfer_perf_handle {
	int shmid;
	unsigned int active_threads;
	int *child_pid_lst;
};

struct dma_meminfo {
	void *memptr;
	unsigned int num_blks;
};

enum dmaio_direction {
	DMAIO_READ,
	DMAIO_WRITE,
	DMAIO_RDWR,
};

#define USE_MEMPOOL

struct mempool_handle {
	void *mempool;
	unsigned int mempool_blkidx;
	unsigned int mempool_blksz;
	unsigned int total_memblks;
	struct dma_meminfo *mempool_info;
#ifdef DEBUG
	unsigned int id;
	unsigned int loop;
#endif
};

struct io_info {
	unsigned int num_req_submitted;
	unsigned int num_req_completed;
	struct list_head *head;
	struct list_head *tail;
	sem_t llock;
	int pid;
	int fd;
	pthread_t evt_id;
	enum dmaio_direction dir;
#ifdef DEBUG
	unsigned long long total_nodes;
	unsigned long long freed_nodes;
#endif
	unsigned int thread_id;
	unsigned int max_reqs;
	struct dmaxfer_io_info *dinfo;
	bool io_exit;
	bool force_exit;
	struct timespec g_ts_start;
	struct timespec ts_cur;
	struct mempool_handle ctxhandle;
	struct mempool_handle iocbhandle;
	struct mempool_handle datahandle;
};

struct list_head {
	struct list_head *next;
	unsigned int max_events;
	unsigned int completed_events;
	io_context_t ctxt;
};

#if DATA_VALIDATION
unsigned short valid_data[2*1024];
#endif

static void clear_events(struct io_info *_info, struct list_head *node);
static void dump_thrd_info(struct io_info *_info) {
	printf("dir = %d\n", _info->dir);
}

static int mempool_create(struct mempool_handle *mpool, unsigned int entry_size,
		unsigned int max_entries)
{
#ifdef USE_MEMPOOL
	if (posix_memalign((void **)&mpool->mempool, DMAPERF_PAGE_SIZE,
			   max_entries * (entry_size + sizeof(struct dma_meminfo)))) {
		printf("OOM Mempool\n");
		return -ENOMEM;
	}
	mpool->mempool_info = (struct dma_meminfo *)(((char *)mpool->mempool) + (max_entries * entry_size));
#endif
	mpool->mempool_blksz = entry_size;
	mpool->total_memblks = max_entries;
	mpool->mempool_blkidx = 0;

	return 0;
}

static void mempool_free(struct mempool_handle *mpool)
{
#ifdef USE_MEMPOOL
	free(mpool->mempool);
	mpool->mempool = NULL;
#endif
}

static void *dma_memalloc(struct mempool_handle *mpool, unsigned int num_blks)
{
	unsigned int _mempool_blkidx = mpool->mempool_blkidx;
	unsigned int tmp_blkidx = _mempool_blkidx;
	unsigned int max_blkcnt = tmp_blkidx + num_blks;
	unsigned int i, avail = 0;
	void *memptr = NULL;
	char *mempool = mpool->mempool;
	struct dma_meminfo *_mempool_info = mpool->mempool_info;
	unsigned int _total_memblks = mpool->total_memblks;

#ifdef USE_MEMPOOL
	if (max_blkcnt > _total_memblks) {
		tmp_blkidx = 0;
		max_blkcnt = num_blks;
	}
	for (i = tmp_blkidx; (i < _total_memblks) && (i < max_blkcnt); i++) {
		if (_mempool_info[i].memptr) { /* occupied blks ahead */
			i += _mempool_info[i].num_blks;
			max_blkcnt = i + num_blks;
			avail = 0;
			tmp_blkidx = i;
		} else
			avail++;
		if (max_blkcnt > _total_memblks) { /* reached the end of mempool. circle through*/
			if (num_blks > _mempool_blkidx) return NULL; /* Continuous num_blks not available */
			i = 0;
			avail = 0;
			max_blkcnt = num_blks;
			tmp_blkidx = 0;
		}
	}
	if (avail < num_blks) { /* no required available blocks */
		return NULL;
	}

	memptr = &(mempool[tmp_blkidx * mpool->mempool_blksz]);
	_mempool_info[tmp_blkidx].memptr = memptr;
	_mempool_info[tmp_blkidx].num_blks = num_blks;
	mpool->mempool_blkidx = tmp_blkidx + num_blks;
#else
	memptr = calloc(num_blks, mpool->mempool_blksz);
#endif

	return memptr;
}

static void dma_free(struct mempool_handle *mpool, void *memptr)
{
#ifdef USE_MEMPOOL
	struct dma_meminfo *_meminfo = mpool->mempool_info;
	unsigned int idx;

	if (!memptr) return;

	idx = (memptr - mpool->mempool)/mpool->mempool_blksz;
#ifdef DEBUG
	if (idx >= mpool->total_memblks) {
		printf("Asserting: %u:Invalid memory index %u acquired\n", mpool->id, idx);
		while(1);
	}
#endif

	_meminfo[idx].num_blks = 0;
	_meminfo[idx].memptr = NULL;
#else
	free(memptr);
#endif
}

static int dmasetio_info(struct io_info *info, struct dmaxfer_io_info *ptr,
		unsigned int base, enum dmaio_direction dir)
{

	int fd;
	unsigned int flags;

	info[base].dir = dir;
	info[base].dinfo = ptr;
	info[base].thread_id = base;
	info[base].max_reqs = ptr->max_req_outstanding;
	sem_init(&info[base].llock, 0, 1);

	if (dir == DMAIO_READ)
	       flags = O_RDONLY | O_NONBLOCK;
	else
	       flags = O_WRONLY | O_NONBLOCK;

	fd = open(ptr->file_name, flags);
	if (fd < 0)
		return fd;
	info[base].fd = fd;

	return 0;
}

static int create_thread_info(struct dmaxfer_io_info *list,
		unsigned int num_entries,
		unsigned int num_jobs)
{
	unsigned int base = 0;
	unsigned int i, j;
	struct io_info *_info;
	int shmid;

	if ((shmid = shmget(IPC_PRIVATE,
			num_jobs * sizeof(struct io_info),
			IPC_CREAT | 0666)) < 0)
	{
		perror("smget returned -1\n");
		error(-1, errno, " ");
		return -EINVAL;
	}

	if ((_info = (struct io_info *) shmat(shmid, NULL, 0)) == (struct io_info *) -1) {
		perror("Process shmat returned NULL\n");
		error(-1, errno, " ");
	}

	for (i = 0; i < num_entries; i++) {
		for (j = 0; j < list[i].num_jobs; j++) {
			if ((list[i].write == DMAIO_RDWR) ||
					(list[i].write == DMAIO_READ)) {
				dmasetio_info(_info, list + i, base,
					      DMAIO_READ);
				base++;
			}
			if ((list[i].write == DMAIO_RDWR) ||
					(list[i].write == DMAIO_WRITE)) {
				dmasetio_info(_info, list + i, base,
					      DMAIO_WRITE);
				base++;
			}
			//dump_thrd_info(&_info[i]);
		}
	}

	if (shmdt(_info) == -1) {
		perror("shmdt returned -1\n");
	        error(-1, errno, " ");
	}
	return shmid;
}

#define MAX_AIO_EVENTS 65536

static void list_add_tail(struct io_info *_info, struct list_head *node)
{
    sem_wait(&_info->llock);
    if (_info->head == NULL) {
        _info->head = node;
        _info->tail = node;
    } else {
        _info->tail->next = node;
        _info->tail = node;
    }
#ifdef DEBUG
    _info->total_nodes++;
#endif
    sem_post( &_info->llock);
}

static void list_add_head(struct io_info *_info, struct list_head *node)
{
    sem_wait(&_info->llock);
    node->next = _info->head;
    if (_info->head == NULL) {
        _info->tail = node;
    }
    _info->head = node;
    sem_post( &_info->llock);
}

static struct list_head *list_pop(struct io_info *_info)
{
	struct list_head *node = NULL;

	sem_wait(&_info->llock);
	node = _info->head;
	if (_info->head == _info->tail)
		_info->tail = NULL;

	if (node)
		_info->head = node->next;

	sem_post(&_info->llock);

	return node;
}

static void list_free(struct io_info *_info)
{
	struct list_head *node = NULL;
	struct list_head *prev_node = NULL;

	sem_wait(&_info->llock);
#ifdef DEBUG
	printf("Need to free %llu nodes in thrd%u\n", _info->total_nodes - _info->freed_nodes, _info->thread_id);
#endif
	node = _info->head;

	while (node != NULL) {
		clear_events(_info, node);
		io_destroy(node->ctxt);
		prev_node = node;
		node = node->next;
		dma_free(&_info->ctxhandle, prev_node);
#ifdef DEBUG
		_info->freed_nodes++;
#endif
	}
	sem_post(&_info->llock);
}

/* Subtract timespec t2 from t1
 *
 * Both t1 and t2 must already be normalized
 * i.e. 0 <= nsec < 1000000000
 */
static int timespec_check(struct timespec *t)
{
	if ((t->tv_nsec < 0) || (t->tv_nsec >= 1000000000))
		return -1;
	return 0;

}

void xfer_timespec_sub(struct timespec *t1, struct timespec *t2)
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

static void clear_events(struct io_info *_info, struct list_head *node) {
	struct io_event *events = NULL;
	int num_events = 0;
	unsigned int j, bufcnt;
	struct timespec ts_cur = {1, 0};
	struct mempool_handle *iocbhandle;
	struct mempool_handle *datahandle;

	iocbhandle = &_info->iocbhandle;
	datahandle = &_info->datahandle;

	if (node->max_events <= node->completed_events)
		return;
#ifdef DEBUG
	printf("Thrd%u: Need to clear %u/%u events in node %p\n",
	       _info->thread_id, node->max_events - node->completed_events,
	       node->max_events, node);
#endif
	events = calloc(node->max_events - node->completed_events, sizeof(struct io_event));
	if (events == NULL) {
		printf("[%s] OOM\n", __func__);
		return;
	}
	do {
		num_events = io_getevents(node->ctxt, 1,
					  node->max_events - node->completed_events, events,
					  &ts_cur);
		for (j = 0; (num_events > 0) && (j < num_events); j++) {
			struct iocb *iocb = (struct iocb *)events[j].obj;
			struct iovec *iov;

			node->completed_events++;
			if (!iocb) {
				printf("Error: Invalid IOCB from events\n");
				continue;
			}

			iov = (struct iovec *)iocb->u.c.buf;

			for (bufcnt = 0; bufcnt < iocb->u.c.nbytes; bufcnt++)
				dma_free(datahandle, iov[bufcnt].iov_base);
			dma_free(iocbhandle, iocb);
		}
	} while ((num_events > 0) && (node->max_events > node->completed_events));

	free(events);
}

static void *event_mon(void *handle)
{
	unsigned int j, bufcnt;
	struct io_info *_info;
	struct io_event *events = NULL;
	int num_events = 0;
	struct timespec ts_cur = {0, 0};
#if DATA_VALIDATION
	unsigned short *rcv_data;
	unsigned int k;
#endif
	struct mempool_handle *ctxhandle;
	struct mempool_handle *iocbhandle;
	struct mempool_handle *datahandle;

	_info = (struct io_info *) handle;


	ctxhandle = &_info->ctxhandle;
	iocbhandle = &_info->iocbhandle;
	datahandle = &_info->datahandle;

	events = calloc(MAX_AIO_EVENTS, sizeof(struct io_event));
	if (events == NULL) {
		printf("OOM AIO_EVENTS\n");
		return NULL;
	}
	while (_info->io_exit == 0 && _info->force_exit == 0) {
		struct list_head *node = list_pop(_info);

		if (!node)
			continue;

		memset(events, 0, MAX_AIO_EVENTS * sizeof(struct io_event));
		do {
			num_events = io_getevents(node->ctxt, 1,
						  node->max_events - node->completed_events,
						  events, &ts_cur);
			for (j = 0; (num_events > 0) && (j < num_events); j++) {
				struct iocb *iocb = (struct iocb *)events[j].obj;
				struct iovec *iov = NULL;

				if (!iocb) {
					printf("Error: Invalid IOCB from events\n");
					continue;
				}
				_info->num_req_completed += events[j].res;

				iov = (struct iovec *)(iocb->u.c.buf);
				if (!iov) {
					printf("invalid buffer\n");
					continue;
				}
#if DATA_VALIDATION
				rcv_data = iov[0].iov_base;
				for (k = 0; k < (iov[0].iov_len/2) && events[j].res && !(events[j].res2); k += 8) {
					printf("%04x: %04x %04x %04x %04x %04x %04x %04x %04x\n", k,
							rcv_data[k], rcv_data[k+1], rcv_data[k+2],
							rcv_data[k+3], rcv_data[k+4], rcv_data[k+5],
							rcv_data[k+6], rcv_data[k+7]);
				}
#endif
				for (bufcnt = 0; (bufcnt < iocb->u.c.nbytes) && iov; bufcnt++)
					dma_free(datahandle, iov[bufcnt].iov_base);
				dma_free(iocbhandle, iocb);
			}
			if (num_events > 0)
			    node->completed_events += num_events;
			if (node->completed_events >= node->max_events) {
				io_destroy(node->ctxt);
				dma_free(ctxhandle, node);
				break;
			}
		} while ((_info->io_exit == 0) && (_info->force_exit == 0));

		if (node->completed_events < node->max_events)
		    list_add_head(_info, node);
	}
	free(events);
#ifdef DEBUG
	printf("Exiting evt_thrd: %u\n", _info->thread_id);
#endif

	return NULL;
}

static void io_proc_cleanup(struct io_info *_info)
{
	_info->io_exit = 1;
	pthread_join(_info->evt_id, NULL);
	list_free(_info);
	mempool_free(&_info->iocbhandle);
	mempool_free(&_info->ctxhandle);
	mempool_free(&_info->datahandle);
	close(_info->fd);
}

static void *io_thread(struct io_info *_info)
{
	struct dmaxfer_io_info *dinfo;
	int ret;
	int s;
	unsigned int tsecs;
	unsigned int max_io = MAX_AIO_EVENTS;
	unsigned int cnt = 0;
	pthread_attr_t attr;
	unsigned int io_sz;
	unsigned int burst_cnt;
	unsigned int num_desc;
	unsigned int max_reqs;
	struct iocb *io_list[1];
	struct mempool_handle *ctxhandle;
	struct mempool_handle *iocbhandle;
	struct mempool_handle *datahandle;

	ctxhandle = &_info->ctxhandle;
	iocbhandle = &_info->iocbhandle;
	datahandle = &_info->datahandle;

	dinfo = _info->dinfo;
	io_sz = dinfo->pkt_sz;
	burst_cnt = dinfo->pkt_burst;
	tsecs = dinfo->runtime;
	num_desc = (io_sz + DMAPERF_PAGE_SIZE - 1) >> DMAPERF_PAGE_SHIFT;
	max_reqs = _info->max_reqs;

	mempool_create(datahandle, num_desc*DMAPERF_PAGE_SIZE,
			max_reqs + (burst_cnt * num_desc));
	mempool_create(ctxhandle, sizeof(struct list_head), max_reqs);
	mempool_create(iocbhandle,
			sizeof(struct iocb) + (burst_cnt * sizeof(struct iovec)),
			max_reqs + (burst_cnt * num_desc));
#ifdef DEBUG
	ctxhandle->id = 1;
	datahandle->id = 0;
	iocbhandle->id = 2;
#endif
	s = pthread_attr_init(&attr);
	if (s != 0)
		printf("pthread_attr_init failed\n");
	if (pthread_create(&_info->evt_id, &attr, event_mon, _info))
		return NULL;

	clock_gettime(CLOCK_MONOTONIC, &_info->g_ts_start);

	do {
		struct list_head *node = NULL;
		struct timespec ts_cur;

		if (tsecs) {
			ret = clock_gettime(CLOCK_MONOTONIC, &ts_cur);
			xfer_timespec_sub(&ts_cur, &_info->g_ts_start);
			if (ts_cur.tv_sec >= tsecs)
				break;
		}
		node = dma_memalloc(ctxhandle, 1);
		if (!node) {
			continue;
		}
		ret = io_queue_init(max_io, &node->ctxt);
		if (ret != 0) {
			printf("Error: io_setup error %d on %u\n", ret, _info->thread_id);
			dma_free(ctxhandle, node);
			sched_yield();
			continue;
		}
		cnt = 0;
		node->max_events = max_io;
		list_add_tail(_info, node);
		do {
			struct iovec *iov = NULL;
			unsigned int iovcnt;
			if (tsecs) {
				ret = clock_gettime(CLOCK_MONOTONIC, &ts_cur);
				xfer_timespec_sub(&ts_cur, &_info->g_ts_start);
				if (ts_cur.tv_sec >= tsecs) {
					node->max_events = cnt;
					break;
				}
			}

			if (((_info->num_req_submitted - _info->num_req_completed) *
					num_desc) > max_reqs) {
				sched_yield();
				continue;
			}

			io_list[0] = dma_memalloc(iocbhandle, 1);
			if (io_list[0] == NULL) {
				if (cnt) {
					node->max_events = cnt;
					break;
				}
				else {
					sched_yield();
					continue;
				}
			}
			iov = (struct iovec *)(io_list[0] + 1);
			for (iovcnt = 0; iovcnt < burst_cnt; iovcnt++) {
				iov[iovcnt].iov_base = dma_memalloc(datahandle, 1);
				if (iov[iovcnt].iov_base == NULL)
					break;
				iov[iovcnt].iov_len = io_sz;
			}
			if (iovcnt == 0) {
				dma_free(iocbhandle, io_list[0]);
				continue;
			}
			if (_info->dir == DMAIO_WRITE) {
#if 0
				printf("DEBUG WRITE: _info->fd :%d iov:%d iovcnt:%d\n",

						_info->fd, iov, iovcnt);
#endif
				io_prep_pwritev(io_list[0],
					       _info->fd,
					       iov,
					       iovcnt,
					       0);
			} else {
				io_prep_preadv(io_list[0],
					       _info->fd,
					       iov,
					       iovcnt,
					       0);
			}

			ret = io_submit(node->ctxt, 1, io_list);
			if(ret != 1) {
				for (; iovcnt > 0; iovcnt--)
					dma_free(datahandle, iov[iovcnt].iov_base);
				dma_free(iocbhandle, io_list[0]);
				node->max_events = cnt;
				break;
			} else {
				cnt++;
				_info->num_req_submitted += iovcnt;
				sched_yield();
			}
		} while (!_info->force_exit && (cnt < max_io));
	} while (!_info->force_exit);
	io_proc_cleanup(_info);

	return _info;
}

#ifdef DEBUG
static void dump_result(unsigned long long total_io_sz)
{
	unsigned long long gig_div = ((unsigned long long)tsecs * 1000000000);
	unsigned long long meg_div = ((unsigned long long)tsecs * 1000000);
	unsigned long long kil_div = ((unsigned long long)tsecs * 1000);
	unsigned long long byt_div = ((unsigned long long)tsecs);

	if ((total_io_sz/gig_div)) {
		printf("BW = %f GB/sec\n", ((double)total_io_sz/gig_div));
	} else if ((total_io_sz/meg_div)) {
		printf("BW = %f MB/sec\n", ((double)total_io_sz/meg_div));
	} else if ((total_io_sz/kil_div)) {
		printf("BW = %f KB/sec\n", ((double)total_io_sz/kil_div));
	} else
		printf("BW = %f Bytes/sec\n", ((double)total_io_sz/byt_div));
}
#endif

int is_valid_fd(int fd)
{
    return fcntl(fd, F_GETFL) != -1 || errno != EBADF;
}

static void cleanup(struct dmaxfer_io_info *list,
		    unsigned int num_entries,
		    unsigned int num_threads)
{
    struct io_info *info;
    struct dmaxfer_io_info *dinfo;
	struct dmaxfer_perf_handle *handle;
	int shmid;
    int i;

	if (!list || !num_entries)
		return;

	handle = (struct dmaxfer_perf_handle *)list->handle;
	if (!handle)
		return;

	if (handle->shmid > 0)
		shmid = handle->shmid;
	else
		return;

    if ((info = (struct io_info *)shmat(shmid, NULL, 0)) ==
            (struct io_info *) -1) {
        perror("Process shmat returned NULL\n");
        error(-1, errno, " ");
        return;
    }

    /* accumulate the statistics */
	for (i = 0; i < num_threads; i++) {
		dinfo = info[i].dinfo;
		dinfo->pps += info[i].num_req_completed;
	}
    for (i = 0; i < num_entries; i++) {
	    dinfo = &list[i];
	    dinfo->pps /= dinfo->runtime;
    }

    if (shmdt(info) == -1) {
        perror("shmdt returned -1\n");
        error(-1, errno, " ");
    }

    if (shmctl(shmid, IPC_RMID, NULL) == -1) {
        perror("shmctl returned -1\n");
            error(-1, errno, " ");
    }
}

int dmaxfer_perf_run(struct dmaxfer_io_info *list,
		unsigned int num_entries)
{
	unsigned int i;
	unsigned int aio_max_nr = 0xFFFFFFFF;
	char aio_max_nr_cmd[100] = {'\0'};
	unsigned int num_thrds = 0;
	unsigned int tot_num_jobs;
	struct dmaxfer_perf_handle *handle;
	struct io_info *info;
	int shmid;
	int base_pid;
	int *child_pid_lst;
	int sys_ret;

#if DATA_VALIDATION
	for (i = 0; i < 2*1024; i++)
		valid_data[i] = i;
#endif

	if (!list) {
		perror("Invalid configuration\n");
		return -EINVAL;
	}


	snprintf(aio_max_nr_cmd, 100, "echo %u > /proc/sys/fs/aio-max-nr",
		 aio_max_nr);
	sys_ret = system(aio_max_nr_cmd);

	handle = (struct dmaxfer_perf_handle *)calloc(1, sizeof(struct dmaxfer_perf_handle));
	if (!handle)
		return -ENOMEM;

	/** Validate input config and calculate SHM segments required */
	tot_num_jobs = 0;
	for (i = 0; i < num_entries; i++) {
		if (!list[i].file_name) {
			perror("Invalid device File Name\n");
			error(-1, errno, " ");
			return -EINVAL;
		}

		if (list[i].num_jobs <= 0) {
			perror("Invalid number of jobs at entry\n");
			return -EINVAL;
		}
		if (list[i].write == DMAIO_RDWR)
			tot_num_jobs += list[i].num_jobs * 2;
		else
			tot_num_jobs += list[i].num_jobs;
		list[i].handle = (unsigned long)handle;
	}

	shmid = create_thread_info(list, num_entries, tot_num_jobs);
	if (shmid < 0) {
		printf("ERROR: Invalid SHMID\n");
		return shmid;
	}

	handle->shmid = shmid;
	handle->active_threads = num_thrds = tot_num_jobs;
	printf("dmautils(%u) threads\n", num_thrds);
	base_pid = getpid();
	if (num_thrds > 1) {
		handle->child_pid_lst = child_pid_lst = calloc(num_thrds, sizeof(int));
		child_pid_lst[0] = base_pid;
		for (i = 1; i < num_thrds; i++) {
			if (getpid() == base_pid)
				child_pid_lst[i] = fork();
			else
				break;
		}
	}

	if ((info = (struct io_info *) shmat(shmid, NULL, 0)) == (struct io_info *) -1) {
		perror("Process shmat returned NULL\n");
		error(-1, errno, " ");
		return -EINVAL;
	}

	if (getpid() == base_pid) {
		info->pid = base_pid;
		io_thread(info);
		if (num_thrds > 1) {
			for(i = 1; i < num_thrds; i++) {
				waitpid(child_pid_lst[i], NULL, 0);
				info[i].pid = 0;
				handle->active_threads--;
			}
			free(handle->child_pid_lst);
			handle->child_pid_lst = NULL;
		}
		if ((shmdt(info) == -1)) {
			perror("shmdt returned -1\n");
			error(-1, errno, " ");
		}
		cleanup(list, num_entries, num_thrds);
		//active_threads = 0;
	} else {
		info[i - 1].pid = getpid();
		io_thread(&info[i - 1]);
		if ((shmdt(info) == -1)) {
			perror("shmdt returned -1\n");
			error(-1, errno, " ");
		}
		return (i - 1);
	}

	return 0;
}

#define RW_MAX_SIZE	0x7ffff000
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

ssize_t dmaperf_asynciosubmit(int fd, char *buffer, ssize_t size,
		uint64_t offset, bool write)
{
	io_context_t aioctx;
	struct iocb *iocb;
	struct iocb *iocbp;
	struct iovec *iov;
	struct io_event event;
	struct timespec ts_cur = {1, 0};
	unsigned int num_events;
	int ret;

	memset(&aioctx, 0, sizeof(aioctx));
	ret = io_queue_init(1, &aioctx);
	if (ret != 0) {
		printf("Error: io_setup error %d\n", ret);
		return ret;
	}

	iocb = (struct iocb *) calloc(sizeof(struct iocb), 1);
	if (!iocb) {
		printf("Error: OOM iocb\n");
		return -ENOMEM;
	}

	if (write)
		io_prep_pwrite(iocb, fd, buffer, 1, offset);
	else
		io_prep_pread(iocb, fd, buffer, 1, offset);

	ret = io_submit(aioctx, 1, &iocb);
	if(ret != 1) {
		printf("Error: io_submit failed\n");
		io_destroy(aioctx);
		return ret;
	}

	num_events = io_getevents(aioctx, 1, 1,
			&event, &ts_cur);

	if (num_events != 1) {
		printf("Error: io_getevents timed out\n");
		return -EIO;
	}

	iocbp = (struct iocb *)event.obj;
	if (!iocbp) {
		printf("Error: Invalid IOCB from events\n");
		return -EIO;
	}

	iov = (struct iovec *)(iocbp->u.c.buf);
	if (!iov) {
		printf("invalid buffer\n");
		return -EIO;
	}

	return size;
}

ssize_t dmaxfer_iosubmit(char *fname, unsigned char write,
		enum dmaxfer_io_type io_type, char *buffer,
		uint64_t size)
{
	unsigned int flags;
	ssize_t count;
	int fd;
	unsigned int base = 0;

	if (!fname || !buffer || size == 0) {
		printf("Invalid arguments\n");
		return -EINVAL;
	}

	if (write)
		flags = O_WRONLY | O_NONBLOCK;
	else
		flags = O_RDONLY | O_NONBLOCK;

	fd = open(fname, flags);
	if (fd < 0)
		return fd;

	if (io_type == DMAXFER_IO_SYNC)
		if (write)
			count = write_from_buffer(fd, buffer, size, base);
		else
			count = read_to_buffer(fd, buffer, size, base);
	else
		count = dmaperf_asynciosubmit(fd, buffer, size, base, write);

	close(fd);

	return count;
}

void dmaxfer_perf_stop(struct dmaxfer_io_info *list,
			unsigned int num_entries)
{
	struct dmaxfer_perf_handle *handle;
	struct io_info *info;
	int shmid;
	int i;

	handle = (struct dmaxfer_perf_handle *)list->handle;
	if (!handle)
		return;

	if (handle->shmid <= 0)
		return;
	shmid = handle->shmid;

	if ((info = (struct io_info *)shmat(shmid, NULL, 0)) == (struct io_info *) -1) {
		perror("Process shmat returned NULL\n");
		error(-1, errno, " ");
		return;
	}

	for (i = 0; i < handle->active_threads; i++) {
		sem_wait(&info[i].llock);
		info[i].io_exit = 1;
		sem_post(&info[i].llock);
	}
}
