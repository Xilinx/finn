// (c) Copyright 2022, Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a 
// copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation 
// the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the 
// Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
// DEALINGS IN THE SOFTWARE.
////////////////////////////////////////////////////////////

module cmd_queue_v2_0_0_regs
(
  // Register Interface
  cmd_queue_v2_0_0_reg_if.sub         sq_reg_if,
  cmd_queue_v2_0_0_reg_if.sub         cq_reg_if,

  // Clock/Reset
  input  logic                      aclk,
  input  logic                      aresetn,

  // Interrupts
  output logic                      irq_sq,
  output logic                      irq_cq
);

// --------------------------------------------------------
// Time Units/Precision
// --------------------------------------------------------
// synthesis translate_off
timeunit 1ns/1ps;
// synthesis translate_on

// --------------------------------------------------------
// Parameters
// --------------------------------------------------------
localparam int  C_SQ_ADDR_WIDTH          = $bits(sq_reg_if.reg_wr_addr);
localparam int  C_CQ_ADDR_WIDTH          = $bits(cq_reg_if.reg_wr_addr);
localparam int  IRQ_SR_WIDTH             = 2;

// --------------------------------------------------------
// Types
// --------------------------------------------------------

// --------------------------------------------------------
// Functions
// --------------------------------------------------------

// --------------------------------------------------------
// Variables/Nets
// --------------------------------------------------------

logic [31:0]              sq_tail_pntr;
logic                     sq_irq_tail_pntr;
logic                     sq_irq_tail_pntr_isr;
logic                     sq_irq_tail_pntr_clr;
logic                     sq_irq_reg;
logic                     sq_irq_reg_isr;
logic                     sq_irq_reg_clr;
logic                     sq_irq_en;
logic [IRQ_SR_WIDTH-1:0]  sq_irq_en_sr;
logic                     sq_irq_type;
logic                     sq_rst;
logic [31:0]              sq_mem_addr_lo;
logic [31:0]              sq_mem_addr_hi;

logic [31:0]              cq_tail_pntr;
logic                     cq_irq_tail_pntr;
logic                     cq_irq_tail_pntr_isr;
logic                     cq_irq_tail_pntr_clr;
logic                     cq_irq_reg;
logic                     cq_irq_reg_isr;
logic                     cq_irq_reg_clr;
logic                     cq_irq_en;
logic [IRQ_SR_WIDTH-1:0]  cq_irq_en_sr;
logic                     cq_irq_type;
logic                     cq_rst;
logic [31:0]              cq_mem_addr_lo;
logic [31:0]              cq_mem_addr_hi;

logic                     soft_rst;

// ========================================================

// Assert the soft reset when either SQ/CQ resets are set
assign soft_rst = sq_rst | cq_rst;

// ========================================================

// Interrupt Generation
always_ff @(posedge aclk) begin
  if (!aresetn) begin
    irq_sq                <= '0;
    sq_irq_en_sr          <= '0;
    sq_irq_reg_isr        <= '0;
    sq_irq_tail_pntr_isr  <= '0;
    irq_cq                <= '0;
    cq_irq_en_sr          <= '0;
    cq_irq_reg_isr        <= '0;
    cq_irq_tail_pntr_isr  <= '0;
  end else begin
    // Defaults
    irq_sq                <= '0;
    sq_irq_en_sr          <= {sq_irq_en_sr[IRQ_SR_WIDTH-2:0],1'b1};
    irq_cq                <= '0;
    cq_irq_en_sr          <= {cq_irq_en_sr[IRQ_SR_WIDTH-2:0],1'b1};

    // SQ Interrupt Generation
    
    // SQ Register Interrupt - Asserted when the Interrupt Register is written

    if ((sq_irq_reg_clr && sq_irq_type) || soft_rst) begin
      // Clear the register interrupt when the interrupt status register is read or on a soft reset
      sq_irq_reg_isr  <= '0;
      sq_irq_en_sr    <= '0;
    end
    
    if (sq_irq_type && sq_irq_reg) begin
      // Assert the SQ register interrupt when the interrupt register is written 
      sq_irq_reg_isr  <= '1;
    end else if (!sq_irq_type) begin
      // If the interrupt type is configured for the tail pointer, then clear any pending register
      // interrupts that get asserted
      sq_irq_reg_isr  <= '0;
    end

    // SQ Tail Pointer Interrupt - Asserted when the Tail Pointer register is written

    if ((sq_irq_tail_pntr_clr && !sq_irq_type) || soft_rst) begin
      // Clear the register interrupt when the interrupt status register is read or on a soft reset
      sq_irq_tail_pntr_isr  <= '0;
      sq_irq_en_sr          <= '0;
    end

    if (!sq_irq_type && sq_irq_tail_pntr) begin
      // Assert the interrupt when the tail pointer register is written
      sq_irq_tail_pntr_isr  <= '1;
    end else if (sq_irq_type) begin
      // If the interrupt type is configured for register interrupt, then clear any pending tail
      // pointer interrupts that get asserted
      sq_irq_tail_pntr_isr  <= '0;
    end

    if (sq_irq_en) begin
      // Assert the SQ interrupt output when enabled and either the register or tail pointer
      // interrupts are set. The MSB of the shift register ensures that the interrupt de-asserts
      // in the event of a coincident interrupt set/clear that would prevent a new rising-edge 
      irq_sq  <= (sq_irq_reg_isr | sq_irq_tail_pntr_isr) & sq_irq_en_sr[IRQ_SR_WIDTH-1];
    end

    // ========================================================

    // CQ Interrupt Generation
    
    // CQ Register Interrupt - Asserted when the Interrupt Register is written

    if ((cq_irq_reg_clr && cq_irq_type) || soft_rst) begin
      // Clear the register interrupt when the interrupt status register is read or on a soft reset
      cq_irq_reg_isr  <= '0;
      cq_irq_en_sr    <= '0;
    end
    
    if (cq_irq_type && cq_irq_reg) begin
      // Assert the CQ register interrupt when the interrupt register is written 
      cq_irq_reg_isr  <= '1;
    end else if (!cq_irq_type) begin
      // If the interrupt type is configured for the tail pointer, then clear any pending register
      // interrupts that get asserted
      cq_irq_reg_isr  <= '0;
    end

    // CQ Tail Pointer Interrupt - Asserted when the Tail Pointer register is written

    if ((cq_irq_tail_pntr_clr && !cq_irq_type) || soft_rst) begin
      // Clear the register interrupt when the interrupt status register is read or on a soft reset
      cq_irq_tail_pntr_isr  <= '0;
      cq_irq_en_sr          <= '0;
    end

    if (!cq_irq_type && cq_irq_tail_pntr) begin
      // Assert the interrupt when the tail pointer register is written
      cq_irq_tail_pntr_isr  <= '1;
    end else if (cq_irq_type) begin
      // If the interrupt type is configured for register interrupt, then clear any pending tail
      // pointer interrupts that get asserted
      cq_irq_tail_pntr_isr  <= '0;
    end

    if (cq_irq_en) begin
      // Assert the CQ interrupt output when enabled and either the register or tail pointer
      // interrupts are set. The MSB of the shift register ensures that the interrupt de-asserts
      // in the event of a coincident interrupt set/clear that would prevent a new rising-edge 
      irq_cq  <= (cq_irq_reg_isr | cq_irq_tail_pntr_isr) & cq_irq_en_sr[IRQ_SR_WIDTH-1];
    end
  end
end

// ========================================================

// Producer Register Interface - Write
always_ff @(posedge aclk) begin
  if (!aresetn) begin
    sq_tail_pntr            <= '0;
    sq_irq_tail_pntr        <= '0;
    sq_irq_reg              <= '0;
    sq_irq_en               <= '0;
    sq_irq_type             <= '0;
    sq_mem_addr_hi          <= '0;
    sq_mem_addr_lo          <= '0;
    sq_rst                  <= '0;
    sq_reg_if.reg_wr_done   <= '0;
  end else begin
    // Defaults
    sq_reg_if.reg_wr_done   <= '0;
    sq_rst                  <= '0;
    sq_irq_tail_pntr        <= '0;
    sq_irq_reg              <= '0;

    if (sq_reg_if.reg_wr_valid) begin
      // Exclude unused address space to prevent aliasing
      if (!(|sq_reg_if.reg_wr_addr[C_SQ_ADDR_WIDTH-1:9])) begin
        case (sq_reg_if.reg_wr_addr[8:0]) inside
          9'b0000000??: // SQ Tail Pointer - 0x000
            begin
              sq_tail_pntr      <= sq_reg_if.reg_wr_data;
              sq_irq_tail_pntr  <= '1;                   
            end
          9'b0000001??: // SQ IRQ Control - 0x004
            sq_irq_reg        <= sq_reg_if.reg_wr_data[0];
          9'b0000010??: // SQ Queue Memory Address Low - 0x008
            sq_mem_addr_lo    <= sq_reg_if.reg_wr_data;
          9'b0000011??: // SQ Reset IRQ Control - 0x00C
            begin
              sq_irq_en       <= sq_reg_if.reg_wr_data[0];
              sq_irq_type     <= sq_reg_if.reg_wr_data[1];
              sq_rst          <= sq_reg_if.reg_wr_data[31];
            end
          9'b0000100??: // SQ Queue Memory Address High - 0x010
            sq_mem_addr_hi    <= sq_reg_if.reg_wr_data;
        endcase
      end
      // Signal write done
      sq_reg_if.reg_wr_done <= 1'b1;
    end

    // Clear the registers on a soft reset
    if (soft_rst) begin
      sq_tail_pntr            <= '0;
      sq_irq_en               <= '0;
      sq_irq_type             <= '0;
      sq_mem_addr_hi          <= '0;
      sq_mem_addr_lo          <= '0;
    end
  end
end

// Always respond with OKAY to writes
assign sq_reg_if.reg_wr_resp = '0;

// ========================================================

// Producer Register Interface - Read
always_ff @(posedge aclk) begin
  //Defaults
  sq_reg_if.reg_rd_data <= '0;
  sq_reg_if.reg_rd_done <= '0;
  cq_irq_reg_clr        <= '0;
  cq_irq_tail_pntr_clr  <= '0;

  if (sq_reg_if.reg_rd_valid) begin
    // Exclude unused address space to prevent aliasing
    if (!(|sq_reg_if.reg_rd_addr[C_SQ_ADDR_WIDTH-1:9])) begin
      case (sq_reg_if.reg_rd_addr[8:0]) inside
        9'b0000000??: // SQ Tail Pointer - 0x000
          sq_reg_if.reg_rd_data           <= sq_tail_pntr;
        9'b0000001??: // SQ IRQ Control - 0x004
          sq_reg_if.reg_rd_data[1]        <= sq_irq_reg_isr | sq_irq_tail_pntr_isr;
        9'b0000010??: // SQ Queue Memory Address Low - 0x008
          sq_reg_if.reg_rd_data           <= sq_mem_addr_lo;
        9'b0000011??: // SQ Reset IRQ Control - 0x00C
          begin
            sq_reg_if.reg_rd_data[1]      <= sq_irq_type;
            sq_reg_if.reg_rd_data[0]      <= sq_irq_en;
          end
        9'b0000100??: // SQ Queue Memory Address High - 0x010
          sq_reg_if.reg_rd_data           <= sq_mem_addr_hi;
        9'b1000000??: // CQ Tail Pointer - 0x100
          begin
            sq_reg_if.reg_rd_data           <= cq_tail_pntr;
            cq_irq_tail_pntr_clr            <= '1;
          end
        9'b1000001??: // CQ IRQ Status - 0x104
          begin
            sq_reg_if.reg_rd_data[0]        <= cq_irq_reg_isr;
            cq_irq_reg_clr                  <= '1;
          end
        9'b1000010??: // CQ Queue Memory Address Low - 0x108
          sq_reg_if.reg_rd_data           <= cq_mem_addr_lo;
        9'b1000011??: // CQ Reset IRQ Control - 0x10C
          begin  
            sq_reg_if.reg_rd_data[1]      <= cq_irq_type;
            sq_reg_if.reg_rd_data[0]      <= cq_irq_en;
          end
        9'b1000100??: // CQ Queue Memory Address High - 0x110
          sq_reg_if.reg_rd_data           <= cq_mem_addr_hi;
        default:
          sq_reg_if.reg_rd_data           <= '0;
      endcase
    end
    // Signal read done
    sq_reg_if.reg_rd_done <= 1'b1;
  end
end

// Always respond with OKAY to reads
assign sq_reg_if.reg_rd_resp = '0;

// ========================================================

// Consumer Register Interface - Write
always_ff @(posedge aclk) begin
  if (!aresetn) begin
    cq_tail_pntr            <= '0;
    cq_irq_tail_pntr        <= '0;
    cq_irq_reg              <= '0;
    cq_irq_en               <= '0;
    cq_irq_type             <= '0;
    cq_mem_addr_hi          <= '0;
    cq_mem_addr_lo          <= '0;
    cq_rst                  <= '0;
    cq_reg_if.reg_wr_done   <= '0;
  end else begin
    // Defaults
    cq_reg_if.reg_wr_done   <= '0;
    cq_rst                  <= '0;
    cq_irq_tail_pntr        <= '0;
    cq_irq_reg              <= '0;

    if (cq_reg_if.reg_wr_valid) begin
      // Exclude unused address space to prevent aliasing
      if (!(|cq_reg_if.reg_wr_addr[C_CQ_ADDR_WIDTH-1:9])) begin
        case (cq_reg_if.reg_wr_addr[8:0]) inside
          9'b0000000??: // CQ Tail Pointer - 0x000
            begin
              cq_tail_pntr      <= cq_reg_if.reg_wr_data;   
              cq_irq_tail_pntr  <= '1;                
            end
          9'b0000001??: // CQ IRQ Control - 0x004
            cq_irq_reg        <= cq_reg_if.reg_wr_data[0];
          9'b0000010??: // CQ Queue Memory Address Low - 0x008
            cq_mem_addr_lo    <= cq_reg_if.reg_wr_data;
          9'b0000011??: // CQ Reset IRQ Control - 0x00C
            begin
              cq_irq_en       <= cq_reg_if.reg_wr_data[0];
              cq_irq_type     <= cq_reg_if.reg_wr_data[1];
              cq_rst          <= cq_reg_if.reg_wr_data[31];
            end
          9'b0000100??: // CQ Queue Memory Address High - 0x010
            cq_mem_addr_hi    <= cq_reg_if.reg_wr_data;
        endcase
      end
      // Signal write done
      cq_reg_if.reg_wr_done <= 1'b1;
    end

    // Clear the registers on a soft reset
    if (soft_rst) begin
      cq_tail_pntr            <= '0;
      cq_irq_en               <= '0;
      cq_irq_type             <= '0;
      cq_mem_addr_hi          <= '0;
      cq_mem_addr_lo          <= '0;
    end
  end
end

// Always respond with OKAY to writes
assign cq_reg_if.reg_wr_resp = '0;

// ========================================================

// Consumer Register Interface - Read
always_ff @(posedge aclk) begin
  //Defaults
  cq_reg_if.reg_rd_data <= '0;
  cq_reg_if.reg_rd_done <= '0;
  sq_irq_reg_clr        <= '0;
  sq_irq_tail_pntr_clr  <= '0;

  if (cq_reg_if.reg_rd_valid) begin
    // Exclude unused address space to prevent aliasing
    if (!(|cq_reg_if.reg_rd_addr[C_CQ_ADDR_WIDTH-1:9])) begin
      case (cq_reg_if.reg_rd_addr[8:0]) inside
        9'b0000000??: // CQ Tail Pointer - 0x000
          cq_reg_if.reg_rd_data           <= cq_tail_pntr;
        9'b0000001??: // CQ IRQ Control - 0x004
          cq_reg_if.reg_rd_data[1]        <= cq_irq_reg_isr | cq_irq_tail_pntr_isr;
        9'b0000010??: // CQ Queue Memory Address Low - 0x008
          cq_reg_if.reg_rd_data           <= cq_mem_addr_lo;
        9'b0000011??: // CQ Reset IRQ Control - 0x00C
          begin  
            cq_reg_if.reg_rd_data[1]      <= cq_irq_type;
            cq_reg_if.reg_rd_data[0]      <= cq_irq_en;
          end
        9'b0000100??: // CQ Queue Memory Address High - 0x010
          cq_reg_if.reg_rd_data           <= cq_mem_addr_hi;
        9'b1000000??: // SQ Tail Pointer - 0x100
          begin
            cq_reg_if.reg_rd_data         <= sq_tail_pntr;
            sq_irq_tail_pntr_clr          <= '1;
          end
        9'b1000001??: // SQ IRQ Status - 0x104
          begin
            cq_reg_if.reg_rd_data[0]      <= sq_irq_reg_isr;
            sq_irq_reg_clr                <= '1;
          end
        9'b1000010??: // SQ Queue Memory Address Low - 0x108
          cq_reg_if.reg_rd_data           <= sq_mem_addr_lo;
        9'b1000011??: // SQ Reset IRQ Control - 0x10C
          begin
            cq_reg_if.reg_rd_data[1]      <= sq_irq_type;
            cq_reg_if.reg_rd_data[0]      <= sq_irq_en;
          end
        9'b1000100??: // SQ Queue Memory Address High - 0x110
          cq_reg_if.reg_rd_data           <= sq_mem_addr_hi;
        default:
          cq_reg_if.reg_rd_data           <= '0;
      endcase
    end
    // Signal read done
    cq_reg_if.reg_rd_done <= 1'b1;
  end
end

// Always respond with OKAY to reads
assign cq_reg_if.reg_rd_resp = '0;

// ========================================================

endmodule : cmd_queue_v2_0_0_regs
