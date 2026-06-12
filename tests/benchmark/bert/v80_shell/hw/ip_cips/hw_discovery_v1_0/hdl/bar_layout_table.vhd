-- (c) Copyright 2022, Advanced Micro Devices, Inc.
-- 
-- Permission is hereby granted, free of charge, to any person obtaining a 
-- copy of this software and associated documentation files (the "Software"), 
-- to deal in the Software without restriction, including without limitation 
-- the rights to use, copy, modify, merge, publish, distribute, sublicense, 
-- and/or sell copies of the Software, and to permit persons to whom the 
-- Software is furnished to do so, subject to the following conditions:
-- 
-- The above copyright notice and this permission notice shall be included in 
-- all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
-- THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
-- FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
-- DEALINGS IN THE SOFTWARE.
------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

library axi_lite_ipif_v3_0_4;
    use axi_lite_ipif_v3_0_4.ipif_pkg.all;
    
library hw_discovery_v1_0_0;    

entity hw_discovery_v1_0_0_bar_layout_table is
    generic (
        C_NUM_SLOTS_BAR_LAYOUT_TABLE  : integer range 1 to 16           := 1;
        C_ENTRY_TYPE_0                : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_0                 : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_0                : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_0       : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_0       : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_0        : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_0               : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_1                : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_1                 : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_1                : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_1       : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_1       : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_1        : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_1               : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_2                : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_2                 : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_2                : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_2       : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_2       : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_2        : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_2               : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_3                : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_3                 : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_3                : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_3       : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_3       : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_3        : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_3               : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_4                : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_4                 : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_4                : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_4       : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_4       : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_4        : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_4               : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_5                : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_5                 : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_5                : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_5       : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_5       : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_5        : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_5               : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_6                : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_6                 : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_6                : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_6       : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_6       : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_6        : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_6               : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_7                : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_7                 : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_7                : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_7       : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_7       : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_7        : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_7               : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_8                : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_8                 : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_8                : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_8       : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_8       : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_8        : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_8               : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_9                : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_9                 : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_9                : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_9       : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_9       : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_9        : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_9               : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_10               : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_10                : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_10               : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_10      : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_10      : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_10       : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_10              : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_11               : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_11                : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_11               : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_11      : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_11      : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_11       : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_11              : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_12               : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_12                : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_12               : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_12      : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_12      : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_12       : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_12              : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_13               : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_13                : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_13               : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_13      : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_13      : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_13       : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_13              : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_14               : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_14                : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_14               : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_14      : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_14      : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_14       : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_14              : std_logic_vector(3 downto 0)    := (others => '0');
        C_ENTRY_TYPE_15               : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_BAR_15                : integer range 0 to 6            := 0;
        C_ENTRY_ADDR_15               : std_logic_vector(47 downto 0)   := (others => '0');
        C_ENTRY_MAJOR_VERSION_15      : integer range 0 to 255          := 0;
        C_ENTRY_MINOR_VERSION_15      : integer range 0 to 255          := 0;
        C_ENTRY_VERSION_TYPE_15       : std_logic_vector(7 downto 0)    := (others => '0');
        C_ENTRY_RSVD0_15              : std_logic_vector(3 downto 0)    := (others => '0');
        C_S_AXI_DATA_WIDTH            : integer range 32 to 32          := 32;
        C_S_AXI_ADDR_WIDTH            : integer range 1 to 64           := 32;
        C_XDEVICEFAMILY               : string                          := "no_family"
        );
    port (

        -----------------------------------------------------------------------
        -- Processor AXI Interface (S_AXI_ACLK)
        -----------------------------------------------------------------------

        s_axi_aclk                : in  std_logic;
        s_axi_aresetn             : in  std_logic;
        s_axi_awaddr              : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
        s_axi_awvalid             : in  std_logic;
        s_axi_awready             : out std_logic;
        s_axi_wdata               : in  std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
        s_axi_wstrb               : in  std_logic_vector((C_S_AXI_DATA_WIDTH/8)-1 downto 0);
        s_axi_wvalid              : in  std_logic;
        s_axi_wready              : out std_logic;
        s_axi_bresp               : out std_logic_vector(1 downto 0);
        s_axi_bvalid              : out std_logic;
        s_axi_bready              : in  std_logic;
        s_axi_araddr              : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
        s_axi_arvalid             : in  std_logic;
        s_axi_arready             : out std_logic;
        s_axi_rdata               : out std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
        s_axi_rresp               : out std_logic_vector(1 downto 0);
        s_axi_rvalid              : out std_logic;
        s_axi_rready              : in  std_logic
    );

end hw_discovery_v1_0_0_bar_layout_table;


architecture top of hw_discovery_v1_0_0_bar_layout_table is


    -------------------------------------------------------------------------------
    -- Constant Declarations
    -------------------------------------------------------------------------------

    constant ZEROES : std_logic_vector(0 to 31) := X"00000000";

    constant C_FAMILY : string := C_XDEVICEFAMILY;

    constant REG_BASEADDR : std_logic_vector := X"00000000";

    impure function makemask (Width: INTEGER) return std_logic_vector is
      variable retv: std_logic_vector (31 downto 0) := (others => '0');
      begin
        for i in (Width - 1) downto 0 loop
            retv(i) := '1';
        end loop;
        return retv;
      end function;

    constant REG_HIGHADDR : std_logic_vector(0 to 31) := makemask(C_S_AXI_ADDR_WIDTH);

    constant C_ARD_ADDR_RANGE_ARRAY : SLV64_ARRAY_TYPE := (
        ZEROES & REG_BASEADDR,
        ZEROES & REG_HIGHADDR
        );

    constant C_ARD_IDX_REGS : integer := 0;

    constant C_ARD_NUM_CE_ARRAY : INTEGER_ARRAY_TYPE := (
        C_ARD_IDX_REGS => 1
        );

    constant C_S_AXI_MIN_SIZE : std_logic_vector(31 downto 0) := makemask(C_S_AXI_ADDR_WIDTH);

    constant C_USE_WSTRB : integer := 0;

    constant C_DPHASE_TIMEOUT : integer := 12;

    subtype IIC_CE_RNG is integer range calc_start_ce_index(C_ARD_NUM_CE_ARRAY, 0) to calc_start_ce_index(C_ARD_NUM_CE_ARRAY, 0) + C_ARD_NUM_CE_ARRAY(0) - 1;

    attribute ram_style : string;

    -- BAR Layout Table ROM type
    type bar_layout_rom_type is array (0 to 63) of std_logic_vector(31 downto 0);
    type rom_header_type is array (0 to 3) of std_logic_vector(31 downto 0);
    type rom_entry_type is array (0 to 63) of std_logic_vector(31 downto 0);

    -- Field Constants
    constant HEADER_FORMAT      : std_logic_vector(19 downto 0)   := x"00001";
    constant HEADER_REV         : std_logic_vector(7 downto 0)    := x"00";
    constant HEADER_LAST_CAP    : std_logic                       := '1';
    constant HEADER_RESERVED    : std_logic_vector(2 downto 0)    := "000";
    constant HEADER_LENGTH      : std_logic_vector(31 downto 0)   := std_logic_vector(to_unsigned((C_NUM_SLOTS_BAR_LAYOUT_TABLE * 16) + 32, 32));
    constant FORMAT_ENTRY_SIZE  : std_logic_vector(7 downto 0)    := x"10";
    constant ENTRY_REVISION     : std_logic_vector(4 downto 0)    := (others => '0');
    constant ENTRY_END_OF_TABLE : std_logic_vector(7 downto 0)    := (others => '1');

    constant ROM_HEADER         : rom_header_type                 := (0 => (HEADER_RESERVED & HEADER_LAST_CAP & HEADER_REV & HEADER_FORMAT),
                                                                      1 => HEADER_LENGTH,
                                                                      2 => (x"000000" & FORMAT_ENTRY_SIZE),
                                                                      3 => (others => '0'));

    constant ROM_ENTRIES        : rom_entry_type                  := (0   => (C_ENTRY_ADDR_0(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_0, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_0),
                                                                      1   => C_ENTRY_ADDR_0(47 downto 16),
                                                                      2   => x"0" & C_ENTRY_RSVD0_0 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_0, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_0, 8)) & C_ENTRY_VERSION_TYPE_0,
                                                                      3   => x"00000000",
                                                                      4   => (C_ENTRY_ADDR_1(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_1, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_1),
                                                                      5   => C_ENTRY_ADDR_1(47 downto 16),
                                                                      6   => x"0" & C_ENTRY_RSVD0_1 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_1, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_1, 8)) & C_ENTRY_VERSION_TYPE_1,
                                                                      7   => x"00000000",
                                                                      8   => (C_ENTRY_ADDR_2(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_2, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_2),
                                                                      9   => C_ENTRY_ADDR_2(47 downto 16),
                                                                      10  => x"0" & C_ENTRY_RSVD0_2 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_2, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_2, 8)) & C_ENTRY_VERSION_TYPE_2,
                                                                      11  => x"00000000",
                                                                      12  => (C_ENTRY_ADDR_3(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_3, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_3),
                                                                      13  => C_ENTRY_ADDR_3(47 downto 16),
                                                                      14  => x"0" & C_ENTRY_RSVD0_3 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_3, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_3, 8)) & C_ENTRY_VERSION_TYPE_3,
                                                                      15  => x"00000000",
                                                                      16  => (C_ENTRY_ADDR_4(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_4, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_4),
                                                                      17  => C_ENTRY_ADDR_4(47 downto 16),
                                                                      18  => x"0" & C_ENTRY_RSVD0_4 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_4, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_4, 8)) & C_ENTRY_VERSION_TYPE_4,
                                                                      19  => x"00000000",
                                                                      20  => (C_ENTRY_ADDR_5(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_5, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_5),
                                                                      21  => C_ENTRY_ADDR_5(47 downto 16),
                                                                      22  => x"0" & C_ENTRY_RSVD0_5 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_5, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_5, 8)) & C_ENTRY_VERSION_TYPE_5,
                                                                      23  => x"00000000",
                                                                      24  => (C_ENTRY_ADDR_6(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_6, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_6),
                                                                      25  => C_ENTRY_ADDR_6(47 downto 16),
                                                                      26  => x"0" & C_ENTRY_RSVD0_6 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_6, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_6, 8)) & C_ENTRY_VERSION_TYPE_6,
                                                                      27  => x"00000000",
                                                                      28  => (C_ENTRY_ADDR_7(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_7, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_7),
                                                                      29  => C_ENTRY_ADDR_7(47 downto 16),
                                                                      30  => x"0" & C_ENTRY_RSVD0_7 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_7, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_7, 8)) & C_ENTRY_VERSION_TYPE_7,
                                                                      31  => x"00000000",
                                                                      32  => (C_ENTRY_ADDR_8(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_8, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_8),
                                                                      33  => C_ENTRY_ADDR_8(47 downto 16),
                                                                      34  => x"0" & C_ENTRY_RSVD0_8 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_8, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_8, 8)) & C_ENTRY_VERSION_TYPE_8,
                                                                      35  => x"00000000",
                                                                      36  => (C_ENTRY_ADDR_9(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_9, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_9),
                                                                      37  => C_ENTRY_ADDR_9(47 downto 16),
                                                                      38  => x"0" & C_ENTRY_RSVD0_9 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_9, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_9, 8)) & C_ENTRY_VERSION_TYPE_9,
                                                                      39  => x"00000000",
                                                                      40  => (C_ENTRY_ADDR_10(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_10, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_10),
                                                                      41  => C_ENTRY_ADDR_10(47 downto 16),
                                                                      42  => x"0" & C_ENTRY_RSVD0_10 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_10, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_10, 8)) & C_ENTRY_VERSION_TYPE_10,
                                                                      43  => x"00000000",
                                                                      44  => (C_ENTRY_ADDR_11(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_11, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_11),
                                                                      45  => C_ENTRY_ADDR_11(47 downto 16),
                                                                      46  => x"0" & C_ENTRY_RSVD0_11 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_11, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_11, 8)) & C_ENTRY_VERSION_TYPE_11,
                                                                      47  => x"00000000",
                                                                      48  => (C_ENTRY_ADDR_12(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_12, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_12),
                                                                      49  => C_ENTRY_ADDR_12(47 downto 16),
                                                                      50  => x"0" & C_ENTRY_RSVD0_12 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_12, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_12, 8)) & C_ENTRY_VERSION_TYPE_12,
                                                                      51  => x"00000000",
                                                                      52  => (C_ENTRY_ADDR_13(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_13, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_13),
                                                                      53  => C_ENTRY_ADDR_13(47 downto 16),
                                                                      54  => x"0" & C_ENTRY_RSVD0_13 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_13, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_13, 8)) & C_ENTRY_VERSION_TYPE_13,
                                                                      55  => x"00000000",
                                                                      56  => (C_ENTRY_ADDR_14(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_14, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_14),
                                                                      57  => C_ENTRY_ADDR_14(47 downto 16),
                                                                      58  => x"0" & C_ENTRY_RSVD0_14 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_14, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_14, 8)) & C_ENTRY_VERSION_TYPE_14,
                                                                      59  => x"00000000",
                                                                      60  => (C_ENTRY_ADDR_15(15 downto 0) & std_logic_vector(to_unsigned(C_ENTRY_BAR_15, 3)) & ENTRY_REVISION & C_ENTRY_TYPE_15),
                                                                      61  => C_ENTRY_ADDR_15(47 downto 16),
                                                                      62  => x"0" & C_ENTRY_RSVD0_15 & std_logic_vector(to_unsigned(C_ENTRY_MAJOR_VERSION_15, 8)) & std_logic_vector(to_unsigned(C_ENTRY_MINOR_VERSION_15, 8)) & C_ENTRY_VERSION_TYPE_15,
                                                                      63  => x"00000000");                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

    -------------------------------------------------------------------------------
    -- Function Declarations
    -------------------------------------------------------------------------------

    function fn_rom_init return bar_layout_rom_type is

        variable rom  : bar_layout_rom_type := (others => (others => '0'));
        variable j    : integer             := 0;

    begin

        -- Insert the ROM Header & Format Fields
        for i in rom_header_type'RANGE loop

            rom(i)  := ROM_HEADER(i);

        end loop;

        -- Insert the configured table entries
        j := 0;

        for i in 4 to (C_NUM_SLOTS_BAR_LAYOUT_TABLE * 4 + 3) loop

            rom(i)  := ROM_ENTRIES(j);
            j       := j + 1;

        end loop;

        -- Insert the end of table entry
        rom((C_NUM_SLOTS_BAR_LAYOUT_TABLE * 4 + 4)) := x"000000" & ENTRY_END_OF_TABLE;

        return rom;

    end function;

    -------------------------------------------------------------------------------
    -- Signal Declarations
    -------------------------------------------------------------------------------

    signal Bus2IP_Clk           : std_logic                                                                := '0';
    signal Bus2IP_Resetn        : std_logic                                                                := '0';
    signal Bus2IP_Addr          : std_logic_vector((C_S_AXI_ADDR_WIDTH-1) downto 0)                        := (others => '0');
    signal Bus2IP_RNW           : std_logic                                                                := '0';
    signal Bus2IP_BE            : std_logic_vector(((C_S_AXI_DATA_WIDTH/8)-1) downto 0)                    := (others => '0');
    signal Bus2IP_CS            : std_logic_vector(((C_ARD_ADDR_RANGE_ARRAY'length)/2-1) downto 0)         := (others => '0');
    signal Bus2IP_RdCE          : std_logic_vector((calc_num_ce(C_ARD_NUM_CE_ARRAY)-1) downto 0)           := (others => '0');
    signal Bus2IP_WrCE          : std_logic_vector((calc_num_ce(C_ARD_NUM_CE_ARRAY)-1) downto 0)           := (others => '0');
    signal Bus2IP_Data          : std_logic_vector((C_S_AXI_DATA_WIDTH-1) downto 0)                        := (others => '0');
    signal IP2Bus_Data          : std_logic_vector((C_S_AXI_DATA_WIDTH-1) downto 0)                        := (others => '0');
    signal IP2Bus_WrAck         : std_logic                                                                := '0';
    signal IP2Bus_RdAck         : std_logic                                                                := '0';
    signal IP2Bus_Error         : std_logic                                                                := '0';
    signal IP2Bus_Ack           : std_logic_vector(1 to 4)                                                 := (others => '0');
    signal BAR_Layout_ROM       : bar_layout_rom_type                                                      := fn_rom_init;

    attribute ram_style of BAR_Layout_ROM       : signal is "distributed";

begin

    axi_lite_ipif_1 : entity axi_lite_ipif_v3_0_4.axi_lite_ipif
        generic map
        (
            C_S_AXI_DATA_WIDTH     => C_S_AXI_DATA_WIDTH,
            C_S_AXI_ADDR_WIDTH     => C_S_AXI_ADDR_WIDTH,
            C_S_AXI_MIN_SIZE       => C_S_AXI_MIN_SIZE,
            C_USE_WSTRB            => C_USE_WSTRB,
            C_DPHASE_TIMEOUT       => C_DPHASE_TIMEOUT,
            C_ARD_ADDR_RANGE_ARRAY => C_ARD_ADDR_RANGE_ARRAY,
            C_ARD_NUM_CE_ARRAY     => C_ARD_NUM_CE_ARRAY,
            C_FAMILY               => C_FAMILY
        )
        port map (
            s_axi_aclk    => s_axi_aclk,
            s_axi_aresetn => s_axi_aresetn,
            s_axi_awaddr  => s_axi_awaddr,
            s_axi_awvalid => s_axi_awvalid,
            s_axi_awready => s_axi_awready,
            s_axi_wdata   => s_axi_wdata,
            s_axi_wstrb   => s_axi_wstrb,
            s_axi_wvalid  => s_axi_wvalid,
            s_axi_wready  => s_axi_wready,
            s_axi_bresp   => s_axi_bresp,
            s_axi_bvalid  => s_axi_bvalid,
            s_axi_bready  => s_axi_bready,
            s_axi_araddr  => s_axi_araddr,
            s_axi_arvalid => s_axi_arvalid,
            s_axi_arready => s_axi_arready,
            s_axi_rdata   => s_axi_rdata,
            s_axi_rresp   => s_axi_rresp,
            s_axi_rvalid  => s_axi_rvalid,
            s_axi_rready  => s_axi_rready,
            Bus2IP_Clk    => Bus2IP_Clk,
            Bus2IP_Resetn => Bus2IP_Resetn,
            Bus2IP_Addr   => Bus2IP_Addr,
            Bus2IP_RNW    => Bus2IP_RNW,
            Bus2IP_BE     => Bus2IP_BE,
            Bus2IP_CS     => Bus2IP_CS,
            Bus2IP_RdCE   => Bus2IP_RdCE,
            Bus2IP_WrCE   => Bus2IP_WrCE,
            Bus2IP_Data   => Bus2IP_Data,
            IP2Bus_Data   => IP2Bus_Data,
            IP2Bus_WrAck  => IP2Bus_WrAck,
            IP2Bus_RdAck  => IP2Bus_RdAck,
            IP2Bus_Error  => IP2Bus_Error
        );

    axi_dec : process(Bus2IP_Clk)

        variable Addr_Slice1    : std_logic_vector(7 downto 2)   := (others => '0');

    begin

        if rising_edge(Bus2IP_Clk) then

            -- Default assignments
            IP2Bus_Data   <= (others => '0');
            IP2Bus_Ack    <= (others => '0');
            IP2Bus_WrAck  <= '0';
            IP2Bus_RdAck  <= '0';

            if (Bus2IP_CS(0) = '1') then

                Addr_Slice1  := Bus2IP_Addr(Addr_Slice1'RANGE);

                -- Read the BAR Layout Table ROM
                IP2Bus_Data  <= BAR_Layout_ROM(to_integer(unsigned(Addr_Slice1)));

                -- Generate the Ack shift reg
                IP2Bus_Ack   <= '1' & IP2Bus_Ack(1 to IP2Bus_Ack'HIGH-1);

            end if;

            -- Single cycle Rd/Wr Ack to IPIF
            if ((IP2Bus_Ack(3) = '1') and (IP2Bus_Ack(4) = '0')) then

                IP2Bus_WrAck  <= '1';
                IP2Bus_RdAck  <= '1';

            end if;

        end if;

    end process axi_dec;

    IP2Bus_Error <= '0';


end top;
