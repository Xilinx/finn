#include <stdio.h>
#include <xil_printf.h>
#include <stdint.h>
#include <xparameters.h>
#include <sleep.h>
#define INSTR_BASE (XPAR_XINSTRUMENTATION_WRAPPER_0_S_AXI_CTRL_BASEADDR)
#define OFST_CFG (0x10)
#define OFST_STATUS (0x18)
#define OFST_LATENCY (0x28)
#define OFST_INTERVAL (0x38)
#define OFST_CHECKSUM (0x48)

#define CLK_FREQ_MHZ 200
int main() {

    // Start the instrumentation wrapper
    // CFG reg offset 0x10:
    // bit[0] = 1 -> enable lfsr.
    // bit[31:16] = seed
    uint32_t *instrumentation_base = (uint32_t *)(INSTR_BASE+OFST_CFG);
    *instrumentation_base = 0x00010003;

    float total_latency_cycles = 0.0;
    float total_interval_cycles = 0.0;
    int num_samples = 10;

    for(int j=0; j<num_samples; j++) {

        sleep(1);

        xil_printf("\033[2J\033[H");
        xil_printf("\r\nRunning Instrumentation Wrapper\r\n\r\n");

        xil_printf("**************************************\r\n");
        xil_printf("Sample %d\r\n", j+1);
        xil_printf("**************************************\r\n\r\n");
        xil_printf("Status                 :    %08X\r\n", *(uint32_t *)(INSTR_BASE+OFST_STATUS));
        xil_printf("Latency        (cycles): %11d\r\n", *(uint32_t *)(INSTR_BASE+OFST_LATENCY));
        total_latency_cycles += *(uint32_t *)(INSTR_BASE+OFST_LATENCY);
        xil_printf("Interval       (cycles): %11d\r\n", *(uint32_t *)(INSTR_BASE+OFST_INTERVAL));
        total_interval_cycles += *(uint32_t *)(INSTR_BASE+OFST_INTERVAL);
        xil_printf("Checksum          (hex):    %08X\r\n", *(uint32_t *)(INSTR_BASE+OFST_CHECKSUM));
        xil_printf("--------------------------------------\r\n\r\n");

    }

    sleep(1);

    xil_printf("\033[2J\033[H");
    xil_printf("\r\nInstrumentation Wrapper has finished running\r\n\r\n");

    xil_printf("*****************************************\r\n");
    xil_printf("Average metrics after 10 seconds:\r\n");
    xil_printf("*****************************************\r\n\r\n");
    xil_printf("Status                 :        %08X\r\n\r\n", *(uint32_t *)(INSTR_BASE+OFST_STATUS));
    xil_printf("Latency        (cycles): %15d\r\n", int(total_latency_cycles/num_samples));
    xil_printf("Interval       (cycles): %15d\r\n", int(total_interval_cycles/num_samples));
    xil_printf("Latency  (microseconds): %15d\r\n", int((total_latency_cycles/num_samples)*(1.0/CLK_FREQ_MHZ)));
    xil_printf("Interval (microseconds): %15d\r\n\r\n", int((total_interval_cycles/num_samples)*(1.0/CLK_FREQ_MHZ)));
    xil_printf("Checksum          (hex):        %08X\r\n", *(uint32_t *)(INSTR_BASE+OFST_CHECKSUM));
    xil_printf("-----------------------------------------\r\n\r\n");

    return 0;
}
