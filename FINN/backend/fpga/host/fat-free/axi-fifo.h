#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>

#define MIN(x,y) ((x < y) ? x : y)
#define MAX(x,y) ((x > y) ? x : y)

/* ltoh: little to host */
/* htol: little to host */
#if __BYTE_ORDER == __LITTLE_ENDIAN
#  define ltohl(x)       (x)
#  define ltohs(x)       (x)
#  define htoll(x)       (x)
#  define htols(x)       (x)
#elif __BYTE_ORDER == __BIG_ENDIAN
#  define ltohl(x)     __bswap_32(x)
#  define ltohs(x)     __bswap_16(x)
#  define htoll(x)     __bswap_32(x)
#  define htols(x)     __bswap_16(x)
#endif
  
#define FATAL do { fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno)); exit(1); } while(0)
 
#define MAP_SIZE_USER (4*1024*1024UL)
#define MAP_MASK (MAP_SIZE_USER - 1)

#define DMA_BLOCK_SIZE 	512

#define AXI_FIFO_WORD_WIDTH 32

/* AXI FIFO Registers Addresses */

// Interrupt Status Register
#define AXI_FIFO_ISR 	0xc0000
// Interrupt Enable Register
#define AXI_FIFO_IER	0xc0004
// Transmit Data FIFO Reset
#define AXI_FIFO_TDFR 	0xc0008
// Transmit Data FIFO Vacancy
#define AXI_FIFO_TDFV	0xc000c
// Transmit Lenght Register
#define AXI_FIFO_TLR	0xc0014
// Receive Data FIFO Reset
#define AXI_FIFO_RDFR 	0xc0018
// Receive Data FIFO Occupancy
#define AXI_FIFO_RDFO	0xc001c
// Receive Lenght Register
#define AXI_FIFO_RLR	0xc0024
// AXI4-Stream Reset
#define AXI_FIFO_SRR 	0xc0028

typedef struct axi_fifo {
	char dev_user[256];
	int fd_user;

	char dev_h2c[256];
	int fd_h2c;	

	char dev_c2h[256];
	int fd_c2h;

	void *map_base;

	unsigned int tf_vacancy;
	unsigned int rf_occupancy;
} AXI_Fifo;

inline uint32_t read_reg(void *address) {
	return ltohl(*((uint32_t *) address));
}

inline void write_reg(void *address, uint32_t value) {
	*((uint32_t *) address) = htoll(value);
}

inline uint32_t max_packet_size(uint32_t bytes){
	uint32_t tmp = bytes;
	tmp--;
	tmp |= tmp >> 1;
	tmp |= tmp >> 2;
	tmp |= tmp >> 4;
	tmp |= tmp >> 8;
	tmp |= tmp >> 16;
	tmp++;

	return MIN(DMA_BLOCK_SIZE, ((tmp == bytes) ? bytes : tmp >> 1));
}

void AXI_Fifo_init(AXI_Fifo &fifo, const char *dev_user, const char *dev_h2c, const char *dev_c2h);

void AXI_Fifo_reset(AXI_Fifo &fifo);

void AXI_Fifo_clear(AXI_Fifo &fifo);

void AXI_Fifo_dd_write(AXI_Fifo &fifo, const char *ifile, uint32_t bytes);

void AXI_Fifo_dd_read(AXI_Fifo &fifo, const char *ofile, uint32_t bytes);