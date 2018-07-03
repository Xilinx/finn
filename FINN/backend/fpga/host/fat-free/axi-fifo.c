#include "axi-fifo.h"


void AXI_Fifo_init(AXI_Fifo &fifo, const char *dev_user, const char *dev_h2c, const char *dev_c2h){
	uint8_t *virt_addr;
	
	mode_t perms = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

	/* Open FIFO registers character device */
	printf("Initializing AXI-FIFO Core...\n\n");
	strcpy(fifo.dev_user, dev_user);
	if ((fifo.fd_user = open(dev_user, O_RDWR | O_SYNC, perms)) == -1) FATAL;
	printf("Device %s opened.\n", dev_user); 
	fflush(stdout);	
	
	/* Open DMA Host to Core character device */
	strcpy(fifo.dev_h2c, dev_h2c);
	if ((fifo.fd_h2c = open(dev_h2c, O_CREAT | O_TRUNC | O_WRONLY, perms)) == -1) FATAL;
	printf("Device %s opened.\n", dev_h2c); 
	fflush(stdout);

	/* Open DMA Core to Host character device */
	strcpy(fifo.dev_c2h, dev_c2h);
	if ((fifo.fd_c2h = open(dev_c2h, O_RDONLY, perms)) == -1) FATAL;
	printf("Device %s opened.\n", dev_c2h); 
	fflush(stdout);

	/* Map one page */
	fifo.map_base = mmap(NULL, MAP_SIZE_USER, PROT_READ | PROT_WRITE, MAP_SHARED, fifo.fd_user, 0);
	if (fifo.map_base == MAP_FAILED) FATAL;
	printf("Memory mapped at address %p.\n", fifo.map_base); 
	fflush(stdout);

	/* Reset the AXI4-Stream interface (SRR) */
	virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_SRR;
	write_reg(virt_addr, 0xa5);
	printf("Reset AXI4-Stream interface (SRR)\n");
	fflush(stdout);

	/* Reset the TX FIFO interface (TDFR) */
	virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_TDFR;
	write_reg(virt_addr, 0xa5);
	printf("Reset TX FIFO interface (TDFR)\n");
	fflush(stdout);

	/* Reset the RX FIFO interface (RDFR) */
	virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_RDFR;
	write_reg(virt_addr, 0xa5);
	printf("Reset RX FIFO interface (RDFR)\n");
	fflush(stdout);

	/* Read ISR to confirm resets complete - expect value of 0x01d00000 (ISR) */
	printf("Check ISR to confirm resets complete...");
	fflush(stdout);
	virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_ISR;
	if (read_reg(virt_addr) != 0x01d00000) FATAL;
	printf("Done.\n");
	fflush(stdout);

	/* Write to ISR to clear reset done interrupt bits */
	printf("Clearing ISR bits...");
	fflush(stdout);
	write_reg(virt_addr, 0xffffffff);
	
	/* Read ISR to confirm cleared bits - expect value of 0x0 (ISR) */
	printf("0x%08x\n", read_reg(virt_addr));
	if (read_reg(virt_addr) != 0x0) FATAL;
	printf("Done.\n");
	fflush(stdout);

	/* Read IER to confirm interrupt sources are clear - expect value of 0x0 (IER) */
	printf("Reading IER to confirm interrupt sources are clear...");
	fflush(stdout);
	virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_IER;
	if (read_reg(virt_addr) != 0x0) FATAL;
	printf("Done.\n");
	fflush(stdout);

	/* Read transmit FIFO vacancy */
	virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_TDFV;
	fifo.tf_vacancy = (unsigned int) read_reg(virt_addr);
	printf("Transmit FIFO vacancy:\t\t0x%08x\n", fifo.tf_vacancy);
	fflush(stdout);

	/* Read receive FIFO occupancy - expect value of 0x0 locations (RDFO) */
	virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_RDFO;
	fifo.rf_occupancy = (unsigned int) read_reg(virt_addr);
	printf("Receive FIFO occupancy:\t\t0x%08x\n", fifo.rf_occupancy);
	fflush(stdout);
	if (read_reg(virt_addr) != 0x0) FATAL;
}

void AXI_Fifo_reset(AXI_Fifo &fifo){
	uint8_t *virt_addr;

	printf("Resetting AXI-FIFO Core...");
	fflush(stdout);

	if (fcntl(fifo.fd_user, F_GETFD) != -1)
	{
		/* Reset the AXI4-Stream interface (SRR) */
		virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_SRR;
		write_reg(virt_addr, 0xa5);
		/* Reset the TX FIFO interface (TDFR) */
		virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_TDFR;
		write_reg(virt_addr, 0xa5);
		/* Reset the RX FIFO interface (RDFR) */
		virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_RDFR;
		write_reg(virt_addr, 0xa5);
		/* Read ISR to confirm resets complete - expect value of 0x01d00000 (ISR) */
		virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_ISR;
		if (read_reg(virt_addr) != 0x01d00000) FATAL;
		/* Write to ISR to clear reset done interrupt bits */
		write_reg(virt_addr, 0xffffffff);	
		/* Read ISR to confirm cleared bits - expect value of 0x0 (ISR) */
		if (read_reg(virt_addr) != 0x0) FATAL;
		/* Read transmit FIFO vacancy - expect value of full vacancy (TDFV) */
		virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_TDFV;
		if (read_reg(virt_addr) != fifo.tf_vacancy) FATAL;
		/* Read IER to confirm interrupt sources are clear - expect value of 0x0 (IER) */
		virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_IER;
		if (read_reg(virt_addr) != 0x0) FATAL;
		/* Read receive FIFO occupancy - expect value of 0x0 locations (RDFO) */
		virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_RDFO;
		if (read_reg(virt_addr) != 0x0) FATAL;
	}
	else
	{
		printf("FIFO core is not initialized\n");
		fflush(stdout);
	}
	
	printf("Done.\n");
	fflush(stdout);
}

void AXI_Fifo_clear(AXI_Fifo &fifo){
	uint8_t *virt_addr;

	printf("\nClean-up\n");
	fflush(stdout);

	if (fcntl(fifo.fd_user, F_GETFD) != -1)
	{
		// Read transmit FIFO vacancy - expect value of 0x7FC locations (TDFV) 
		virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_TDFV;
		if (read_reg(virt_addr) != fifo.tf_vacancy) printf("\nTransmit FIFO contains left-over data.\n");
		// Read receive FIFO occupancy - expect value of 0x0 locations (RDFO) 
		virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_RDFO;
		if (read_reg(virt_addr) != 0x0) printf("\nReceive FIFO contains left-over data.\n");
		// Reset AXI FIFO Core 
		AXI_Fifo_reset(fifo);
		// Unmap and close FIFO registers file descriptor 
		if (munmap((uint8_t *) fifo.map_base, MAP_SIZE_USER) == -1) FATAL;
		// Close FIFO registers character device
		close(fifo.fd_user);
		// Close DMA Host to Core character device, if opened
		if (fcntl(fifo.fd_h2c, F_GETFD) != -1) close(fifo.fd_h2c);
		// Close DMA Core to Host character device, if opened
		if (fcntl(fifo.fd_c2h, F_GETFD) != -1) close(fifo.fd_c2h);
	}
	else
	{
		printf("FIFO core is not initialized\n");
		fflush(stdout);
	}
	
	printf("Done.\n");
	fflush(stdout);
}

void AXI_Fifo_dd_write(AXI_Fifo &fifo, const char *ifile, uint32_t bytes){
	char cmd[512];
	uint32_t sent = 0;

	if (fcntl(fifo.fd_user, F_GETFD) == -1 || fcntl(fifo.fd_h2c, F_GETFD) == -1) FATAL;

	/* Write IER to enable transmit complete and reset complete interrupts (IER) */
	uint8_t *virt_addr = (uint8_t *) fifo.map_base + AXI_FIFO_IER;
	write_reg(virt_addr, 0x0c000000);

	#if defined(DEBUG)
		printf("\nBytes to be sent:\t\t%u\n", bytes);
		fflush(stdout);
	#endif

	while (sent < bytes){
		const uint32_t packet = max_packet_size(bytes - sent);

		#if defined(DEBUG) && defined(VERBOSE)
			printf("\nTransaction %u\tPacket size:%u\n", sent / packet, packet);
			fflush(stdout);
		#endif

		sprintf(cmd, "dd if=%s of=%s bs=%u skip=%u count=1 status=none", 
				ifile, fifo.dev_h2c, packet, sent / packet);
		system(cmd);
		sent += packet;

		#if defined(DEBUG) && defined(VERBOSE)
			printf("TDFV after sending:\t0x%08x\n", read_reg((uint8_t *) fifo.map_base + AXI_FIFO_TDFV));
			printf("Bytes sent:\t\t%u\n", packet);
			fflush(stdout);
		#endif
	}

	write_reg((uint8_t *) fifo.map_base + AXI_FIFO_TLR, bytes);
	while (read_reg((uint8_t *) fifo.map_base + AXI_FIFO_TDFV) != fifo.tf_vacancy);
}

void AXI_Fifo_dd_read(AXI_Fifo &fifo, const char *ofile, uint32_t bytes){
	char cmd[512];
	uint32_t received = 0;

	if (fcntl(fifo.fd_user, F_GETFD) == -1 || fcntl(fifo.fd_c2h, F_GETFD) == -1) FATAL;

	#if defined(DEBUG)
		printf("\nBytes to be received:\t\t%u\n", bytes);
		fflush(stdout);
	#endif

	write_reg((uint8_t *) fifo.map_base + AXI_FIFO_RLR, bytes);
	
	while (received < bytes){
		// TODO: read packet larger than just one FIFO word
		const uint32_t packet = AXI_FIFO_WORD_WIDTH;

		#if defined(DEBUG) && defined(VERBOSE)
			printf("\nTransaction %u\tPacket size:%u\n", received / packet, packet);
			fflush(stdout);
		#endif

		sprintf(cmd, "dd if=%s of=%s bs=%u skip=128 count=1 oflag=append conv=notrunc status=none", 
				fifo.dev_c2h, ofile, packet);
		system(cmd);
		received += packet;

		#if defined(DEBUG) && defined(VERBOSE)
			printf("RDFO after receiving:\t0x%08x\n", read_reg((uint8_t *) fifo.map_base + AXI_FIFO_RDFO));
			printf("Bytes sent:\t\t%u\n", packet);
			fflush(stdout);
		#endif
	}

	while (read_reg((uint8_t *) fifo.map_base + AXI_FIFO_RDFO) != fifo.rf_occupancy);
}