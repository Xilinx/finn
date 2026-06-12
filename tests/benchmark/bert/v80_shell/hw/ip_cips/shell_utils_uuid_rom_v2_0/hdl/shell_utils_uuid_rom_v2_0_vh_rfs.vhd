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

library axi_lite_ipif_v3_0_4;
    use axi_lite_ipif_v3_0_4.ipif_pkg.all;

library xpm;
    use xpm.vcomponents.all;

entity shell_utils_uuid_rom is
    generic (
        ------------------------------------------------------------------------
        C_S_AXI_DATA_WIDTH          : integer range 32 to 32    := 32;
        C_S_AXI_ADDR_WIDTH          : integer range  3 to  9    := 4;
        C_MEMORY_INIT               : string                    := "0";
        C_XDEVICEFAMILY             : string                    := "no_family"
        ------------------------------------------------------------------------
        );
    port (
        ------------------------------------------------------------------------
        -- Processor AXI Interface
        ------------------------------------------------------------------------
        S_AXI_ACLK                  : in  std_logic;
        S_AXI_ARESETN               : in  std_logic;
        S_AXI_ARADDR                : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
        S_AXI_ARVALID               : in  std_logic;
        S_AXI_ARREADY               : out std_logic;
        S_AXI_RDATA                 : out std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
        S_AXI_RRESP                 : out std_logic_vector(1 downto 0);
        S_AXI_RVALID                : out std_logic;
        S_AXI_RREADY                : in  std_logic
    );

end shell_utils_uuid_rom;

architecture rtl of shell_utils_uuid_rom is

    -------------------------------------------------------------------------------
    -- Constant Declarations
    -------------------------------------------------------------------------------

    -- Constants for AXI4-Lite.
    constant ZEROES : std_logic_vector(0 to 31) := (others => '0');
    constant ONES   : std_logic_vector(0 to 31) := (others => '1');

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
        ZEROES(0 to 31) & REG_BASEADDR,
        ZEROES(0 to 31) & REG_HIGHADDR
    );

    constant C_ARD_IDX_REGS : integer := 0;

    constant C_ARD_NUM_CE_ARRAY : INTEGER_ARRAY_TYPE := (
        C_ARD_IDX_REGS => 1
    );

    constant C_S_AXI_MIN_SIZE : std_logic_vector(31 downto 0) := makemask(C_S_AXI_ADDR_WIDTH);

    constant C_USE_WSTRB : integer := 0;

    constant C_DPHASE_TIMEOUT : integer := 3;

    constant XPM_ADDR_WIDTH : integer := C_S_AXI_ADDR_WIDTH - 2;
    constant XPM_MEMORY_SIZE : integer := (2 ** XPM_ADDR_WIDTH) * C_S_AXI_DATA_WIDTH;

    attribute DONT_TOUCH : string;
    attribute DONT_TOUCH of xpm_memory_spram_inst: label is "TRUE";

    -------------------------------------------------------------------------------
    --      SIGNALS
    -------------------------------------------------------------------------------
    signal Bus2IP_Clk    : std_logic                     := '0';
    signal Bus2IP_Resetn : std_logic;
    signal Bus2IP_Addr   : std_logic_vector((C_S_AXI_ADDR_WIDTH - 1) downto 0);
    signal Bus2IP_RNW    : std_logic;
    signal Bus2IP_BE     : std_logic_vector(((C_S_AXI_DATA_WIDTH / 8) - 1) downto 0);
    signal Bus2IP_CS     : std_logic_vector(((C_ARD_ADDR_RANGE_ARRAY'length) / 2 - 1) downto 0);
    signal Bus2IP_RdCE   : std_logic_vector((calc_num_ce(C_ARD_NUM_CE_ARRAY) - 1) downto 0);
    signal Bus2IP_WrCE   : std_logic_vector((calc_num_ce(C_ARD_NUM_CE_ARRAY) - 1) downto 0);
    signal Bus2IP_Data   : std_logic_vector((C_S_AXI_DATA_WIDTH - 1) downto 0);
    signal IP2Bus_Data   : std_logic_vector((C_S_AXI_DATA_WIDTH - 1) downto 0);
    signal IP2Bus_WrAck  : std_logic                     := '0';
    signal IP2Bus_RdAck  : std_logic                     := '0';
    signal IP2Bus_Error  : std_logic                     := '0';

begin

axi_lite_ipif_1 : entity axi_lite_ipif_v3_0_4.axi_lite_ipif
    generic map(
        C_S_AXI_DATA_WIDTH     => C_S_AXI_DATA_WIDTH,
        C_S_AXI_ADDR_WIDTH     => C_S_AXI_ADDR_WIDTH,
        C_S_AXI_MIN_SIZE       => C_S_AXI_MIN_SIZE,
        C_USE_WSTRB            => C_USE_WSTRB,
        C_DPHASE_TIMEOUT       => C_DPHASE_TIMEOUT,
        C_ARD_ADDR_RANGE_ARRAY => C_ARD_ADDR_RANGE_ARRAY,
        C_ARD_NUM_CE_ARRAY     => C_ARD_NUM_CE_ARRAY,
        C_FAMILY               => C_FAMILY)
    port map(
        S_AXI_ACLK    => S_AXI_ACLK,
        S_AXI_ARESETN => S_AXI_ARESETN,
        S_AXI_AWADDR  => (others => '0'),
        S_AXI_AWVALID => '0',
        S_AXI_AWREADY => open,
        S_AXI_WDATA   => (others => '0'),
        S_AXI_WSTRB   => (others => '0'),
        S_AXI_WVALID  => '0',
        S_AXI_WREADY  => open,
        S_AXI_BRESP   => open,
        S_AXI_BVALID  => open,
        S_AXI_BREADY  => '0',
        S_AXI_ARADDR  => S_AXI_ARADDR,
        S_AXI_ARVALID => S_AXI_ARVALID,
        S_AXI_ARREADY => S_AXI_ARREADY,
        S_AXI_RDATA   => S_AXI_RDATA,
        S_AXI_RRESP   => S_AXI_RRESP,
        S_AXI_RVALID  => S_AXI_RVALID,
        S_AXI_RREADY  => S_AXI_RREADY,
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
        IP2Bus_Error  => IP2Bus_Error);

xpm_memory_spram_inst : xpm_memory_spram
    generic map (
        ADDR_WIDTH_A => XPM_ADDR_WIDTH,
        AUTO_SLEEP_TIME => 0,
        BYTE_WRITE_WIDTH_A => C_S_AXI_DATA_WIDTH,
        CASCADE_HEIGHT => 0,
        ECC_MODE => "no_ecc",
        MEMORY_INIT_FILE => "none",
        MEMORY_INIT_PARAM => C_MEMORY_INIT,
        MEMORY_OPTIMIZATION => "true",
        MEMORY_PRIMITIVE => "distributed",
        MEMORY_SIZE => XPM_MEMORY_SIZE,
        MESSAGE_CONTROL => 0,
        READ_DATA_WIDTH_A => C_S_AXI_DATA_WIDTH,
        READ_LATENCY_A => 1,
        READ_RESET_VALUE_A => "0",
        RST_MODE_A => "SYNC",
        SIM_ASSERT_CHK => 0,
        USE_MEM_INIT => 1,
        WAKEUP_TIME => "disable_sleep",
        WRITE_DATA_WIDTH_A => C_S_AXI_DATA_WIDTH,
        WRITE_MODE_A => "read_first"
    )
    port map (
        dbiterra => open,
        douta => IP2Bus_Data,
        sbiterra => open,
        addra => Bus2IP_Addr(C_S_AXI_ADDR_WIDTH-1 downto 2),
        clka => Bus2IP_Clk,
        ena => Bus2IP_CS(0),
        injectdbiterra => '0',
        injectsbiterra => '0',
        regcea => '0',
        rsta => '0',
        sleep => '0',
        wea => Bus2IP_WrCE(0 downto 0),
        dina => Bus2IP_Data
    );

end architecture rtl;


