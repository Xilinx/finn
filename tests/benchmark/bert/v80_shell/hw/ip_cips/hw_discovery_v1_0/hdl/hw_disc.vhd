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
    use ieee.std_logic_misc.all;
    use ieee.numeric_std.all;

library axi_lite_ipif_v3_0_4;
    use axi_lite_ipif_v3_0_4.ipif_pkg.all;

library xpm;
    use xpm.vcomponents.all;
    
library hw_discovery_v1_0_0;

entity hw_discovery_v1_0_0_hw_disc is
    generic (
    	  C_NUM_PFS                     		: integer range 1 to 4           := 1;
        C_CAP_BASE_ADDR               		: std_logic_vector(11 downto 0)  := x"480"; -- 0x480 default for PCIE4
        C_NEXT_CAP_ADDR           		    : std_logic_vector(11 downto 0)  := (others => '0');
        C_PF0_NUM_SLOTS_BAR_LAYOUT_TABLE  : integer range 1 to 16          := 1;
        C_PF0_BAR_INDEX               		: integer range 0 to 6           := 0;
        C_PF0_LOW_OFFSET              		: std_logic_vector(27 downto 0)  := (others => '0');
        C_PF0_HIGH_OFFSET             		: std_logic_vector(31 downto 0)  := (others => '0');
        C_PF0_ENTRY_TYPE_0                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_0                 : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_0                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_0       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_0       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_0        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_0               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_1                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_1                 : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_1                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_1       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_1       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_1        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_1               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_2                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_2                 : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_2                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_2       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_2       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_2        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_2               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_3                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_3                 : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_3                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_3       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_3       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_3        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_3               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_4                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_4                 : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_4                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_4       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_4       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_4        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_4               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_5                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_5                 : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_5                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_5       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_5       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_5        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_5               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_6                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_6                 : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_6                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_6       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_6       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_6        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_6               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_7                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_7                 : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_7                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_7       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_7       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_7        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_7               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_8                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_8                 : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_8                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_8       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_8       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_8        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_8               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_9                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_9                 : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_9                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_9       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_9       : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_9        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_9               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_10               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_10                : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_10               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_10      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_10      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_10       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_10              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_11               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_11                : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_11               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_11      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_11      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_11       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_11              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_12               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_12                : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_12               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_12      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_12      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_12       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_12              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_13               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_13                : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_13               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_13      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_13      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_13       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_13              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_14               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_14                : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_14               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_14      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_14      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_14       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_14              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_ENTRY_TYPE_15               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_BAR_15                : integer range 0 to 6           := 0;
        C_PF0_ENTRY_ADDR_15               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF0_ENTRY_MAJOR_VERSION_15      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_MINOR_VERSION_15      : integer range 0 to 255         := 0;
        C_PF0_ENTRY_VERSION_TYPE_15       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF0_ENTRY_RSVD0_15              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF0_S_AXI_DATA_WIDTH            : integer range 32 to 32         := 32;
        C_PF0_S_AXI_ADDR_WIDTH            : integer range 1 to 64          := 32;
        C_PF1_NUM_SLOTS_BAR_LAYOUT_TABLE  : integer range 1 to 16          := 1; 
        C_PF1_BAR_INDEX               		: integer range 0 to 6           := 0;
        C_PF1_LOW_OFFSET              		: std_logic_vector(27 downto 0)  := (others => '0');
        C_PF1_HIGH_OFFSET             		: std_logic_vector(31 downto 0)  := (others => '0');
        C_PF1_ENTRY_TYPE_0                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_0                 : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_0                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_0       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_0       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_0        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_0               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_1                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_1                 : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_1                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_1       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_1       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_1        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_1               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_2                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_2                 : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_2                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_2       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_2       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_2        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_2               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_3                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_3                 : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_3                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_3       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_3       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_3        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_3               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_4                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_4                 : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_4                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_4       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_4       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_4        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_4               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_5                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_5                 : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_5                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_5       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_5       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_5        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_5               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_6                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_6                 : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_6                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_6       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_6       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_6        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_6               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_7                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_7                 : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_7                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_7       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_7       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_7        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_7               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_8                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_8                 : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_8                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_8       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_8       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_8        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_8               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_9                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_9                 : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_9                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_9       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_9       : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_9        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_9               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_10               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_10                : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_10               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_10      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_10      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_10       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_10              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_11               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_11                : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_11               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_11      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_11      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_11       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_11              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_12               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_12                : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_12               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_12      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_12      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_12       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_12              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_13               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_13                : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_13               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_13      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_13      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_13       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_13              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_14               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_14                : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_14               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_14      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_14      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_14       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_14              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_ENTRY_TYPE_15               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_BAR_15                : integer range 0 to 6           := 0;
        C_PF1_ENTRY_ADDR_15               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF1_ENTRY_MAJOR_VERSION_15      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_MINOR_VERSION_15      : integer range 0 to 255         := 0;
        C_PF1_ENTRY_VERSION_TYPE_15       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF1_ENTRY_RSVD0_15              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF1_S_AXI_DATA_WIDTH            : integer range 32 to 32         := 32;
        C_PF1_S_AXI_ADDR_WIDTH            : integer range 1 to 64          := 32;
        C_PF2_NUM_SLOTS_BAR_LAYOUT_TABLE  : integer range 1 to 16          := 1;
        C_PF2_BAR_INDEX               		: integer range 0 to 6           := 0;
        C_PF2_LOW_OFFSET              		: std_logic_vector(27 downto 0)  := (others => '0');
        C_PF2_HIGH_OFFSET             		: std_logic_vector(31 downto 0)  := (others => '0');
        C_PF2_ENTRY_TYPE_0                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_0                 : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_0                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_0       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_0       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_0        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_0               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_1                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_1                 : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_1                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_1       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_1       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_1        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_1               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_2                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_2                 : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_2                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_2       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_2       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_2        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_2               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_3                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_3                 : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_3                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_3       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_3       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_3        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_3               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_4                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_4                 : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_4                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_4       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_4       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_4        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_4               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_5                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_5                 : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_5                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_5       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_5       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_5        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_5               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_6                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_6                 : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_6                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_6       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_6       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_6        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_6               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_7                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_7                 : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_7                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_7       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_7       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_7        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_7               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_8                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_8                 : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_8                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_8       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_8       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_8        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_8               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_9                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_9                 : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_9                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_9       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_9       : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_9        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_9               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_10               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_10                : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_10               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_10      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_10      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_10       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_10              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_11               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_11                : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_11               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_11      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_11      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_11       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_11              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_12               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_12                : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_12               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_12      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_12      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_12       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_12              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_13               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_13                : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_13               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_13      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_13      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_13       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_13              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_14               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_14                : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_14               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_14      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_14      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_14       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_14              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_ENTRY_TYPE_15               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_BAR_15                : integer range 0 to 6           := 0;
        C_PF2_ENTRY_ADDR_15               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF2_ENTRY_MAJOR_VERSION_15      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_MINOR_VERSION_15      : integer range 0 to 255         := 0;
        C_PF2_ENTRY_VERSION_TYPE_15       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF2_ENTRY_RSVD0_15              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF2_S_AXI_DATA_WIDTH            : integer range 32 to 32         := 32;
        C_PF2_S_AXI_ADDR_WIDTH            : integer range 1 to 64          := 32;
        C_PF3_NUM_SLOTS_BAR_LAYOUT_TABLE  : integer range 1 to 16          := 1;
        C_PF3_BAR_INDEX               		: integer range 0 to 6           := 0;
        C_PF3_LOW_OFFSET              		: std_logic_vector(27 downto 0)  := (others => '0');
        C_PF3_HIGH_OFFSET             		: std_logic_vector(31 downto 0)  := (others => '0');
        C_PF3_ENTRY_TYPE_0                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_0                 : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_0                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_0       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_0       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_0        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_0               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_1                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_1                 : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_1                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_1       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_1       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_1        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_1               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_2                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_2                 : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_2                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_2       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_2       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_2        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_2               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_3                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_3                 : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_3                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_3       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_3       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_3        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_3               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_4                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_4                 : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_4                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_4       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_4       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_4        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_4               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_5                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_5                 : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_5                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_5       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_5       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_5        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_5               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_6                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_6                 : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_6                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_6       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_6       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_6        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_6               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_7                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_7                 : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_7                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_7       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_7       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_7        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_7               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_8                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_8                 : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_8                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_8       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_8       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_8        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_8               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_9                : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_9                 : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_9                : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_9       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_9       : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_9        : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_9               : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_10               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_10                : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_10               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_10      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_10      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_10       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_10              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_11               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_11                : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_11               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_11      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_11      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_11       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_11              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_12               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_12                : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_12               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_12      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_12      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_12       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_12              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_13               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_13                : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_13               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_13      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_13      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_13       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_13              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_14               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_14                : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_14               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_14      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_14      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_14       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_14              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_ENTRY_TYPE_15               : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_BAR_15                : integer range 0 to 6           := 0;
        C_PF3_ENTRY_ADDR_15               : std_logic_vector(47 downto 0)  := (others => '0');
        C_PF3_ENTRY_MAJOR_VERSION_15      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_MINOR_VERSION_15      : integer range 0 to 255         := 0;
        C_PF3_ENTRY_VERSION_TYPE_15       : std_logic_vector(7 downto 0)   := (others => '0');
        C_PF3_ENTRY_RSVD0_15              : std_logic_vector(3 downto 0)   := (others => '0');
        C_PF3_S_AXI_DATA_WIDTH            : integer range 32 to 32         := 32;
        C_PF3_S_AXI_ADDR_WIDTH            : integer range 1 to 64          := 32;
        C_XDEVICEFAMILY               		: string                         := "no_family"        	
        );
    port (
        -----------------------------------------------------------------------
        -- Clocks & Resets
        -----------------------------------------------------------------------

        aclk_pcie                  								: in  std_logic;
        aresetn_pcie               								: in  std_logic;
        
        aclk_ctrl                   							: in  std_logic;
        aresetn_ctrl             									: in  std_logic;

        -----------------------------------------------------------------------
        -- slave pcie4_cfg_ext Interface (aclk_pcie)
        -----------------------------------------------------------------------

        s_pcie4_cfg_ext_function_number           : in  std_logic_vector(15 downto 0);
        s_pcie4_cfg_ext_read_data                 : out std_logic_vector(31 downto 0);
        s_pcie4_cfg_ext_read_data_valid           : out std_logic;
        s_pcie4_cfg_ext_read_received             : in  std_logic;
        s_pcie4_cfg_ext_register_number           : in  std_logic_vector(9 downto 0);
        s_pcie4_cfg_ext_write_byte_enable         : in  std_logic_vector(3 downto 0);
        s_pcie4_cfg_ext_write_data                : in  std_logic_vector(31 downto 0);
        s_pcie4_cfg_ext_write_received            : in  std_logic;
        
        -----------------------------------------------------------------------
        -- master pcie4_cfg_ext Interface (aclk_pcie)
        -----------------------------------------------------------------------

        m_pcie4_cfg_ext_function_number           : out std_logic_vector(15 downto 0);
        m_pcie4_cfg_ext_read_data                 : in  std_logic_vector(31 downto 0);
        m_pcie4_cfg_ext_read_data_valid           : in  std_logic;
        m_pcie4_cfg_ext_read_received             : out std_logic;
        m_pcie4_cfg_ext_register_number           : out std_logic_vector(9 downto 0);
        m_pcie4_cfg_ext_write_byte_enable         : out std_logic_vector(3 downto 0);
        m_pcie4_cfg_ext_write_data                : out std_logic_vector(31 downto 0);
        m_pcie4_cfg_ext_write_received            : out std_logic;
        
        -----------------------------------------------------------------------
        -- AXI Interface (aclk_ctrl) for PF0
        -----------------------------------------------------------------------

        s_axi_ctrl_pf0_awaddr              				: in  std_logic_vector(C_PF0_S_AXI_ADDR_WIDTH-1 downto 0);
        s_axi_ctrl_pf0_awvalid             				: in  std_logic;
        s_axi_ctrl_pf0_awready             				: out std_logic;
        s_axi_ctrl_pf0_wdata               				: in  std_logic_vector(C_PF0_S_AXI_DATA_WIDTH-1 downto 0);
        s_axi_ctrl_pf0_wstrb               				: in  std_logic_vector((C_PF0_S_AXI_DATA_WIDTH/8)-1 downto 0);
        s_axi_ctrl_pf0_wvalid              				: in  std_logic;
        s_axi_ctrl_pf0_wready              				: out std_logic;
        s_axi_ctrl_pf0_bresp               				: out std_logic_vector(1 downto 0);
        s_axi_ctrl_pf0_bvalid              				: out std_logic;
        s_axi_ctrl_pf0_bready              				: in  std_logic;
        s_axi_ctrl_pf0_araddr              				: in  std_logic_vector(C_PF0_S_AXI_ADDR_WIDTH-1 downto 0);
        s_axi_ctrl_pf0_arvalid             				: in  std_logic;
        s_axi_ctrl_pf0_arready             				: out std_logic;
        s_axi_ctrl_pf0_rdata               				: out std_logic_vector(C_PF0_S_AXI_DATA_WIDTH-1 downto 0);
        s_axi_ctrl_pf0_rresp               				: out std_logic_vector(1 downto 0);
        s_axi_ctrl_pf0_rvalid              				: out std_logic;
        s_axi_ctrl_pf0_rready              				: in  std_logic;
        
        -----------------------------------------------------------------------
        -- AXI Interface (aclk_ctrl) for PF1
        -----------------------------------------------------------------------

        s_axi_ctrl_pf1_awaddr              				: in  std_logic_vector(C_PF1_S_AXI_ADDR_WIDTH-1 downto 0);
        s_axi_ctrl_pf1_awvalid             				: in  std_logic;
        s_axi_ctrl_pf1_awready             				: out std_logic;
        s_axi_ctrl_pf1_wdata               				: in  std_logic_vector(C_PF1_S_AXI_DATA_WIDTH-1 downto 0);
        s_axi_ctrl_pf1_wstrb               				: in  std_logic_vector((C_PF1_S_AXI_DATA_WIDTH/8)-1 downto 0);
        s_axi_ctrl_pf1_wvalid              				: in  std_logic;
        s_axi_ctrl_pf1_wready              				: out std_logic;
        s_axi_ctrl_pf1_bresp               				: out std_logic_vector(1 downto 0);
        s_axi_ctrl_pf1_bvalid              				: out std_logic;
        s_axi_ctrl_pf1_bready              				: in  std_logic;
        s_axi_ctrl_pf1_araddr              				: in  std_logic_vector(C_PF1_S_AXI_ADDR_WIDTH-1 downto 0);
        s_axi_ctrl_pf1_arvalid             				: in  std_logic;
        s_axi_ctrl_pf1_arready             				: out std_logic;
        s_axi_ctrl_pf1_rdata               				: out std_logic_vector(C_PF1_S_AXI_DATA_WIDTH-1 downto 0);
        s_axi_ctrl_pf1_rresp               				: out std_logic_vector(1 downto 0);
        s_axi_ctrl_pf1_rvalid              				: out std_logic;
        s_axi_ctrl_pf1_rready              				: in  std_logic;
        
        -----------------------------------------------------------------------
        -- AXI Interface (aclk_ctrl) for PF2
        -----------------------------------------------------------------------

        s_axi_ctrl_pf2_awaddr              				: in  std_logic_vector(C_PF2_S_AXI_ADDR_WIDTH-1 downto 0);
        s_axi_ctrl_pf2_awvalid             				: in  std_logic;
        s_axi_ctrl_pf2_awready             				: out std_logic;
        s_axi_ctrl_pf2_wdata               				: in  std_logic_vector(C_PF2_S_AXI_DATA_WIDTH-1 downto 0);
        s_axi_ctrl_pf2_wstrb               				: in  std_logic_vector((C_PF2_S_AXI_DATA_WIDTH/8)-1 downto 0);
        s_axi_ctrl_pf2_wvalid              				: in  std_logic;
        s_axi_ctrl_pf2_wready              				: out std_logic;
        s_axi_ctrl_pf2_bresp               				: out std_logic_vector(1 downto 0);
        s_axi_ctrl_pf2_bvalid              				: out std_logic;
        s_axi_ctrl_pf2_bready              				: in  std_logic;
        s_axi_ctrl_pf2_araddr              				: in  std_logic_vector(C_PF2_S_AXI_ADDR_WIDTH-1 downto 0);
        s_axi_ctrl_pf2_arvalid             				: in  std_logic;
        s_axi_ctrl_pf2_arready             				: out std_logic;
        s_axi_ctrl_pf2_rdata               				: out std_logic_vector(C_PF2_S_AXI_DATA_WIDTH-1 downto 0);
        s_axi_ctrl_pf2_rresp               				: out std_logic_vector(1 downto 0);
        s_axi_ctrl_pf2_rvalid              				: out std_logic;
        s_axi_ctrl_pf2_rready              				: in  std_logic;
        
        -----------------------------------------------------------------------
        -- AXI Interface (aclk_ctrl) for PF3
        -----------------------------------------------------------------------

        s_axi_ctrl_pf3_awaddr              				: in  std_logic_vector(C_PF3_S_AXI_ADDR_WIDTH-1 downto 0);
        s_axi_ctrl_pf3_awvalid             				: in  std_logic;
        s_axi_ctrl_pf3_awready             				: out std_logic;
        s_axi_ctrl_pf3_wdata               				: in  std_logic_vector(C_PF3_S_AXI_DATA_WIDTH-1 downto 0);
        s_axi_ctrl_pf3_wstrb               				: in  std_logic_vector((C_PF3_S_AXI_DATA_WIDTH/8)-1 downto 0);
        s_axi_ctrl_pf3_wvalid              				: in  std_logic;
        s_axi_ctrl_pf3_wready              				: out std_logic;
        s_axi_ctrl_pf3_bresp               				: out std_logic_vector(1 downto 0);
        s_axi_ctrl_pf3_bvalid              				: out std_logic;
        s_axi_ctrl_pf3_bready              				: in  std_logic;
        s_axi_ctrl_pf3_araddr              				: in  std_logic_vector(C_PF3_S_AXI_ADDR_WIDTH-1 downto 0);
        s_axi_ctrl_pf3_arvalid             				: in  std_logic;
        s_axi_ctrl_pf3_arready             				: out std_logic;
        s_axi_ctrl_pf3_rdata               				: out std_logic_vector(C_PF3_S_AXI_DATA_WIDTH-1 downto 0);
        s_axi_ctrl_pf3_rresp               				: out std_logic_vector(1 downto 0);
        s_axi_ctrl_pf3_rvalid              				: out std_logic;
        s_axi_ctrl_pf3_rready              				: in  std_logic
        
    );

end hw_discovery_v1_0_0_hw_disc;

architecture rtl of hw_discovery_v1_0_0_hw_disc is

    -------------------------------------------------------------------------------
    -- Constant Declarations
    -------------------------------------------------------------------------------

    -- Constants for AXI4-Lite.
    constant ZEROES : std_logic_vector(0 to 31) := (others => '0');
    constant ONES   : std_logic_vector(0 to 31) := (others => '1');

    constant C_FAMILY : string := C_XDEVICEFAMILY;

begin
		
	pcie_vsec_inst : entity hw_discovery_v1_0_0.hw_discovery_v1_0_0_pcie_vsec
		generic map (
			C_NUM_PFS                 					=> C_NUM_PFS,
			C_CAP_BASE_ADDR           					=> C_CAP_BASE_ADDR,
			C_NEXT_CAP_ADDR           					=> C_NEXT_CAP_ADDR,
			C_PF0_BAR_INDEX           					=> C_PF0_BAR_INDEX,
			C_PF0_LOW_OFFSET          					=> C_PF0_LOW_OFFSET,
			C_PF0_HIGH_OFFSET         					=> C_PF0_HIGH_OFFSET,
			C_PF1_BAR_INDEX           					=> C_PF1_BAR_INDEX,
			C_PF1_LOW_OFFSET          					=> C_PF1_LOW_OFFSET,
			C_PF1_HIGH_OFFSET         					=> C_PF1_HIGH_OFFSET,
			C_PF2_BAR_INDEX           					=> C_PF2_BAR_INDEX,
			C_PF2_LOW_OFFSET          					=> C_PF2_LOW_OFFSET,
			C_PF2_HIGH_OFFSET         					=> C_PF2_HIGH_OFFSET,
			C_PF3_BAR_INDEX           					=> C_PF3_BAR_INDEX,
			C_PF3_LOW_OFFSET          					=> C_PF3_LOW_OFFSET,
			C_PF3_HIGH_OFFSET         					=> C_PF3_HIGH_OFFSET,
			C_XDEVICEFAMILY           					=> C_XDEVICEFAMILY
		)
		port map (
			aclk_pcie                  				 	=> aclk_pcie,
			aresetn_pcie               				 	=> aresetn_pcie,
			s_pcie4_cfg_ext_function_number    	=> s_pcie4_cfg_ext_function_number,
			s_pcie4_cfg_ext_read_data          	=> s_pcie4_cfg_ext_read_data,
			s_pcie4_cfg_ext_read_data_valid    	=> s_pcie4_cfg_ext_read_data_valid,
			s_pcie4_cfg_ext_read_received      	=> s_pcie4_cfg_ext_read_received,
			s_pcie4_cfg_ext_register_number    	=> s_pcie4_cfg_ext_register_number,
			s_pcie4_cfg_ext_write_byte_enable  	=> s_pcie4_cfg_ext_write_byte_enable,
			s_pcie4_cfg_ext_write_data         	=> s_pcie4_cfg_ext_write_data,
			s_pcie4_cfg_ext_write_received     	=> s_pcie4_cfg_ext_write_received,
			m_pcie4_cfg_ext_function_number    	=> m_pcie4_cfg_ext_function_number,
			m_pcie4_cfg_ext_read_data          	=> m_pcie4_cfg_ext_read_data,
			m_pcie4_cfg_ext_read_data_valid    	=> m_pcie4_cfg_ext_read_data_valid,
			m_pcie4_cfg_ext_read_received      	=> m_pcie4_cfg_ext_read_received,
			m_pcie4_cfg_ext_register_number    	=> m_pcie4_cfg_ext_register_number,
			m_pcie4_cfg_ext_write_byte_enable  	=> m_pcie4_cfg_ext_write_byte_enable,
			m_pcie4_cfg_ext_write_data         	=> m_pcie4_cfg_ext_write_data,
			m_pcie4_cfg_ext_write_received     	=> m_pcie4_cfg_ext_write_received
		);
	
	G_GENERATE: for i in 0 to C_NUM_PFS-1 generate
	
		G_GENERATE_PF0 : if (i = 0) generate

	  	-- Instantiate BAR Layout table
			BAR_LAYOUT_TABLE_inst_0 : entity hw_discovery_v1_0_0.hw_discovery_v1_0_0_bar_layout_table
	    	generic map (
	    		C_NUM_SLOTS_BAR_LAYOUT_TABLE  => C_PF0_NUM_SLOTS_BAR_LAYOUT_TABLE,
	    		C_ENTRY_TYPE_0                => C_PF0_ENTRY_TYPE_0,
	    		C_ENTRY_BAR_0                 => C_PF0_ENTRY_BAR_0,
	    		C_ENTRY_ADDR_0                => C_PF0_ENTRY_ADDR_0,
	    		C_ENTRY_MAJOR_VERSION_0       => C_PF0_ENTRY_MAJOR_VERSION_0,
	    		C_ENTRY_MINOR_VERSION_0       => C_PF0_ENTRY_MINOR_VERSION_0,
	    		C_ENTRY_VERSION_TYPE_0        => C_PF0_ENTRY_VERSION_TYPE_0,
	    		C_ENTRY_RSVD0_0               => C_PF0_ENTRY_RSVD0_0,
	    		C_ENTRY_TYPE_1                => C_PF0_ENTRY_TYPE_1,
	    		C_ENTRY_BAR_1                 => C_PF0_ENTRY_BAR_1,
	    		C_ENTRY_ADDR_1                => C_PF0_ENTRY_ADDR_1,
	    		C_ENTRY_MAJOR_VERSION_1       => C_PF0_ENTRY_MAJOR_VERSION_1,
	    		C_ENTRY_MINOR_VERSION_1       => C_PF0_ENTRY_MINOR_VERSION_1,
	    		C_ENTRY_VERSION_TYPE_1        => C_PF0_ENTRY_VERSION_TYPE_1,
	    		C_ENTRY_RSVD0_1               => C_PF0_ENTRY_RSVD0_1,
	    		C_ENTRY_TYPE_2                => C_PF0_ENTRY_TYPE_2,
	    		C_ENTRY_BAR_2                 => C_PF0_ENTRY_BAR_2,
	    		C_ENTRY_ADDR_2                => C_PF0_ENTRY_ADDR_2,
	    		C_ENTRY_MAJOR_VERSION_2       => C_PF0_ENTRY_MAJOR_VERSION_2,
	    		C_ENTRY_MINOR_VERSION_2       => C_PF0_ENTRY_MINOR_VERSION_2,
	    		C_ENTRY_VERSION_TYPE_2        => C_PF0_ENTRY_VERSION_TYPE_2,
	    		C_ENTRY_RSVD0_2               => C_PF0_ENTRY_RSVD0_2,
	    		C_ENTRY_TYPE_3                => C_PF0_ENTRY_TYPE_3,
	    		C_ENTRY_BAR_3                 => C_PF0_ENTRY_BAR_3,
	    		C_ENTRY_ADDR_3                => C_PF0_ENTRY_ADDR_3,
	    		C_ENTRY_MAJOR_VERSION_3       => C_PF0_ENTRY_MAJOR_VERSION_3,
	    		C_ENTRY_MINOR_VERSION_3       => C_PF0_ENTRY_MINOR_VERSION_3,
	    		C_ENTRY_VERSION_TYPE_3        => C_PF0_ENTRY_VERSION_TYPE_3,
	    		C_ENTRY_RSVD0_3               => C_PF0_ENTRY_RSVD0_3,
	    		C_ENTRY_TYPE_4                => C_PF0_ENTRY_TYPE_4,
	    		C_ENTRY_BAR_4                 => C_PF0_ENTRY_BAR_4,
	    		C_ENTRY_ADDR_4                => C_PF0_ENTRY_ADDR_4,
	    		C_ENTRY_MAJOR_VERSION_4       => C_PF0_ENTRY_MAJOR_VERSION_4,
	    		C_ENTRY_MINOR_VERSION_4       => C_PF0_ENTRY_MINOR_VERSION_4,
	    		C_ENTRY_VERSION_TYPE_4        => C_PF0_ENTRY_VERSION_TYPE_4,
	    		C_ENTRY_RSVD0_4               => C_PF0_ENTRY_RSVD0_4,
	    		C_ENTRY_TYPE_5                => C_PF0_ENTRY_TYPE_5,
	    		C_ENTRY_BAR_5                 => C_PF0_ENTRY_BAR_5,
	    		C_ENTRY_ADDR_5                => C_PF0_ENTRY_ADDR_5,
	    		C_ENTRY_MAJOR_VERSION_5       => C_PF0_ENTRY_MAJOR_VERSION_5,
	    		C_ENTRY_MINOR_VERSION_5       => C_PF0_ENTRY_MINOR_VERSION_5,
	    		C_ENTRY_VERSION_TYPE_5        => C_PF0_ENTRY_VERSION_TYPE_5,
	    		C_ENTRY_RSVD0_5               => C_PF0_ENTRY_RSVD0_5,
	    		C_ENTRY_TYPE_6                => C_PF0_ENTRY_TYPE_6,
	    		C_ENTRY_BAR_6                 => C_PF0_ENTRY_BAR_6,
	    		C_ENTRY_ADDR_6                => C_PF0_ENTRY_ADDR_6,
	    		C_ENTRY_MAJOR_VERSION_6       => C_PF0_ENTRY_MAJOR_VERSION_6,
	    		C_ENTRY_MINOR_VERSION_6       => C_PF0_ENTRY_MINOR_VERSION_6,
	    		C_ENTRY_VERSION_TYPE_6        => C_PF0_ENTRY_VERSION_TYPE_6,
	    		C_ENTRY_RSVD0_6               => C_PF0_ENTRY_RSVD0_6,
	    		C_ENTRY_TYPE_7                => C_PF0_ENTRY_TYPE_7,
	    		C_ENTRY_BAR_7                 => C_PF0_ENTRY_BAR_7,
	    		C_ENTRY_ADDR_7                => C_PF0_ENTRY_ADDR_7,
	    		C_ENTRY_MAJOR_VERSION_7       => C_PF0_ENTRY_MAJOR_VERSION_7,
	    		C_ENTRY_MINOR_VERSION_7       => C_PF0_ENTRY_MINOR_VERSION_7,
	    		C_ENTRY_VERSION_TYPE_7        => C_PF0_ENTRY_VERSION_TYPE_7,
	    		C_ENTRY_RSVD0_7               => C_PF0_ENTRY_RSVD0_7,
	    		C_ENTRY_TYPE_8                => C_PF0_ENTRY_TYPE_8,
	    		C_ENTRY_BAR_8                 => C_PF0_ENTRY_BAR_8,
	    		C_ENTRY_ADDR_8                => C_PF0_ENTRY_ADDR_8,
	    		C_ENTRY_MAJOR_VERSION_8       => C_PF0_ENTRY_MAJOR_VERSION_8,
	    		C_ENTRY_MINOR_VERSION_8       => C_PF0_ENTRY_MINOR_VERSION_8,
	    		C_ENTRY_VERSION_TYPE_8        => C_PF0_ENTRY_VERSION_TYPE_8,
	    		C_ENTRY_RSVD0_8               => C_PF0_ENTRY_RSVD0_8,
	    		C_ENTRY_TYPE_9                => C_PF0_ENTRY_TYPE_9,
	    		C_ENTRY_BAR_9                 => C_PF0_ENTRY_BAR_9,
	    		C_ENTRY_ADDR_9                => C_PF0_ENTRY_ADDR_9,
	    		C_ENTRY_MAJOR_VERSION_9       => C_PF0_ENTRY_MAJOR_VERSION_9,
	    		C_ENTRY_MINOR_VERSION_9       => C_PF0_ENTRY_MINOR_VERSION_9,
	    		C_ENTRY_VERSION_TYPE_9        => C_PF0_ENTRY_VERSION_TYPE_9,
	    		C_ENTRY_RSVD0_9               => C_PF0_ENTRY_RSVD0_9,
	    		C_ENTRY_TYPE_10               => C_PF0_ENTRY_TYPE_10,
	    		C_ENTRY_BAR_10                => C_PF0_ENTRY_BAR_10,
	    		C_ENTRY_ADDR_10               => C_PF0_ENTRY_ADDR_10,
	    		C_ENTRY_MAJOR_VERSION_10      => C_PF0_ENTRY_MAJOR_VERSION_10,
	    		C_ENTRY_MINOR_VERSION_10      => C_PF0_ENTRY_MINOR_VERSION_10,
	    		C_ENTRY_VERSION_TYPE_10       => C_PF0_ENTRY_VERSION_TYPE_10,
	    		C_ENTRY_RSVD0_10              => C_PF0_ENTRY_RSVD0_10,
	    		C_ENTRY_TYPE_11               => C_PF0_ENTRY_TYPE_11,
	    		C_ENTRY_BAR_11                => C_PF0_ENTRY_BAR_11,
	    		C_ENTRY_ADDR_11               => C_PF0_ENTRY_ADDR_11,
	    		C_ENTRY_MAJOR_VERSION_11      => C_PF0_ENTRY_MAJOR_VERSION_11,
	    		C_ENTRY_MINOR_VERSION_11      => C_PF0_ENTRY_MINOR_VERSION_11,
	    		C_ENTRY_VERSION_TYPE_11       => C_PF0_ENTRY_VERSION_TYPE_11,
	    		C_ENTRY_RSVD0_11              => C_PF0_ENTRY_RSVD0_11,
	    		C_ENTRY_TYPE_12               => C_PF0_ENTRY_TYPE_12,
	    		C_ENTRY_BAR_12                => C_PF0_ENTRY_BAR_12,
	    		C_ENTRY_ADDR_12               => C_PF0_ENTRY_ADDR_12,
	    		C_ENTRY_MAJOR_VERSION_12      => C_PF0_ENTRY_MAJOR_VERSION_12,
	    		C_ENTRY_MINOR_VERSION_12      => C_PF0_ENTRY_MINOR_VERSION_12,
	    		C_ENTRY_VERSION_TYPE_12       => C_PF0_ENTRY_VERSION_TYPE_12,
	    		C_ENTRY_RSVD0_12              => C_PF0_ENTRY_RSVD0_12,
	    		C_ENTRY_TYPE_13               => C_PF0_ENTRY_TYPE_13,
	    		C_ENTRY_BAR_13                => C_PF0_ENTRY_BAR_13,
	    		C_ENTRY_ADDR_13               => C_PF0_ENTRY_ADDR_13,
	    		C_ENTRY_MAJOR_VERSION_13      => C_PF0_ENTRY_MAJOR_VERSION_13,
	    		C_ENTRY_MINOR_VERSION_13      => C_PF0_ENTRY_MINOR_VERSION_13,
	    		C_ENTRY_VERSION_TYPE_13       => C_PF0_ENTRY_VERSION_TYPE_13,
	    		C_ENTRY_RSVD0_13              => C_PF0_ENTRY_RSVD0_13,
	    		C_ENTRY_TYPE_14               => C_PF0_ENTRY_TYPE_14,
	    		C_ENTRY_BAR_14                => C_PF0_ENTRY_BAR_14,
	    		C_ENTRY_ADDR_14               => C_PF0_ENTRY_ADDR_14,
	    		C_ENTRY_MAJOR_VERSION_14      => C_PF0_ENTRY_MAJOR_VERSION_14,
	    		C_ENTRY_MINOR_VERSION_14      => C_PF0_ENTRY_MINOR_VERSION_14,
	    		C_ENTRY_VERSION_TYPE_14       => C_PF0_ENTRY_VERSION_TYPE_14,
	    		C_ENTRY_RSVD0_14              => C_PF0_ENTRY_RSVD0_14,
	    		C_ENTRY_TYPE_15               => C_PF0_ENTRY_TYPE_15,
	    		C_ENTRY_BAR_15                => C_PF0_ENTRY_BAR_15,
	    		C_ENTRY_ADDR_15               => C_PF0_ENTRY_ADDR_15,
	    		C_ENTRY_MAJOR_VERSION_15      => C_PF0_ENTRY_MAJOR_VERSION_15,
	    		C_ENTRY_MINOR_VERSION_15      => C_PF0_ENTRY_MINOR_VERSION_15,
	    		C_ENTRY_VERSION_TYPE_15       => C_PF0_ENTRY_VERSION_TYPE_15,
	    		C_ENTRY_RSVD0_15              => C_PF0_ENTRY_RSVD0_15,     
	    		C_S_AXI_DATA_WIDTH            => C_PF0_S_AXI_DATA_WIDTH,
	    		C_S_AXI_ADDR_WIDTH            => C_PF0_S_AXI_ADDR_WIDTH,
	    		C_XDEVICEFAMILY               => C_XDEVICEFAMILY
	    	)
	    	port map (
			    s_axi_aclk            		    => aclk_ctrl,
			    s_axi_aresetn         		    => aresetn_ctrl,
	    		s_axi_awaddr              		=> s_axi_ctrl_pf0_awaddr,
	    		s_axi_awvalid             		=> s_axi_ctrl_pf0_awvalid,
	    		s_axi_awready             		=> s_axi_ctrl_pf0_awready,
	    		s_axi_wdata               		=> s_axi_ctrl_pf0_wdata,
	    		s_axi_wstrb               		=> s_axi_ctrl_pf0_wstrb,
	    		s_axi_wvalid              		=> s_axi_ctrl_pf0_wvalid,
	    		s_axi_wready              		=> s_axi_ctrl_pf0_wready,
	    		s_axi_bresp               		=> s_axi_ctrl_pf0_bresp,
	    		s_axi_bvalid              		=> s_axi_ctrl_pf0_bvalid,
	    		s_axi_bready              		=> s_axi_ctrl_pf0_bready,
	    		s_axi_araddr              		=> s_axi_ctrl_pf0_araddr,
	    		s_axi_arvalid             		=> s_axi_ctrl_pf0_arvalid,
	    		s_axi_arready             		=> s_axi_ctrl_pf0_arready,
	    		s_axi_rdata               		=> s_axi_ctrl_pf0_rdata,
	    		s_axi_rresp               		=> s_axi_ctrl_pf0_rresp,
	    		s_axi_rvalid              		=> s_axi_ctrl_pf0_rvalid,
	    		s_axi_rready              		=> s_axi_ctrl_pf0_rready
	    	);
	
			end generate G_GENERATE_PF0;
			
		G_GENERATE_PF1 : if (i = 1) generate

	  	-- Instantiate BAR Layout table
			BAR_LAYOUT_TABLE_inst_1 : entity hw_discovery_v1_0_0.hw_discovery_v1_0_0_bar_layout_table
	    	generic map (
	    		C_NUM_SLOTS_BAR_LAYOUT_TABLE  => C_PF1_NUM_SLOTS_BAR_LAYOUT_TABLE,
	    		C_ENTRY_TYPE_0                => C_PF1_ENTRY_TYPE_0,
	    		C_ENTRY_BAR_0                 => C_PF1_ENTRY_BAR_0,
	    		C_ENTRY_ADDR_0                => C_PF1_ENTRY_ADDR_0,
	    		C_ENTRY_MAJOR_VERSION_0       => C_PF1_ENTRY_MAJOR_VERSION_0,
	    		C_ENTRY_MINOR_VERSION_0       => C_PF1_ENTRY_MINOR_VERSION_0,
	    		C_ENTRY_VERSION_TYPE_0        => C_PF1_ENTRY_VERSION_TYPE_0,
	    		C_ENTRY_RSVD0_0               => C_PF1_ENTRY_RSVD0_0,
	    		C_ENTRY_TYPE_1                => C_PF1_ENTRY_TYPE_1,
	    		C_ENTRY_BAR_1                 => C_PF1_ENTRY_BAR_1,
	    		C_ENTRY_ADDR_1                => C_PF1_ENTRY_ADDR_1,
	    		C_ENTRY_MAJOR_VERSION_1       => C_PF1_ENTRY_MAJOR_VERSION_1,
	    		C_ENTRY_MINOR_VERSION_1       => C_PF1_ENTRY_MINOR_VERSION_1,
	    		C_ENTRY_VERSION_TYPE_1        => C_PF1_ENTRY_VERSION_TYPE_1,
	    		C_ENTRY_RSVD0_1               => C_PF1_ENTRY_RSVD0_1,
	    		C_ENTRY_TYPE_2                => C_PF1_ENTRY_TYPE_2,
	    		C_ENTRY_BAR_2                 => C_PF1_ENTRY_BAR_2,
	    		C_ENTRY_ADDR_2                => C_PF1_ENTRY_ADDR_2,
	    		C_ENTRY_MAJOR_VERSION_2       => C_PF1_ENTRY_MAJOR_VERSION_2,
	    		C_ENTRY_MINOR_VERSION_2       => C_PF1_ENTRY_MINOR_VERSION_2,
	    		C_ENTRY_VERSION_TYPE_2        => C_PF1_ENTRY_VERSION_TYPE_2,
	    		C_ENTRY_RSVD0_2               => C_PF1_ENTRY_RSVD0_2,
	    		C_ENTRY_TYPE_3                => C_PF1_ENTRY_TYPE_3,
	    		C_ENTRY_BAR_3                 => C_PF1_ENTRY_BAR_3,
	    		C_ENTRY_ADDR_3                => C_PF1_ENTRY_ADDR_3,
	    		C_ENTRY_MAJOR_VERSION_3       => C_PF1_ENTRY_MAJOR_VERSION_3,
	    		C_ENTRY_MINOR_VERSION_3       => C_PF1_ENTRY_MINOR_VERSION_3,
	    		C_ENTRY_VERSION_TYPE_3        => C_PF1_ENTRY_VERSION_TYPE_3,
	    		C_ENTRY_RSVD0_3               => C_PF1_ENTRY_RSVD0_3,
	    		C_ENTRY_TYPE_4                => C_PF1_ENTRY_TYPE_4,
	    		C_ENTRY_BAR_4                 => C_PF1_ENTRY_BAR_4,
	    		C_ENTRY_ADDR_4                => C_PF1_ENTRY_ADDR_4,
	    		C_ENTRY_MAJOR_VERSION_4       => C_PF1_ENTRY_MAJOR_VERSION_4,
	    		C_ENTRY_MINOR_VERSION_4       => C_PF1_ENTRY_MINOR_VERSION_4,
	    		C_ENTRY_VERSION_TYPE_4        => C_PF1_ENTRY_VERSION_TYPE_4,
	    		C_ENTRY_RSVD0_4               => C_PF1_ENTRY_RSVD0_4,
	    		C_ENTRY_TYPE_5                => C_PF1_ENTRY_TYPE_5,
	    		C_ENTRY_BAR_5                 => C_PF1_ENTRY_BAR_5,
	    		C_ENTRY_ADDR_5                => C_PF1_ENTRY_ADDR_5,
	    		C_ENTRY_MAJOR_VERSION_5       => C_PF1_ENTRY_MAJOR_VERSION_5,
	    		C_ENTRY_MINOR_VERSION_5       => C_PF1_ENTRY_MINOR_VERSION_5,
	    		C_ENTRY_VERSION_TYPE_5        => C_PF1_ENTRY_VERSION_TYPE_5,
	    		C_ENTRY_RSVD0_5               => C_PF1_ENTRY_RSVD0_5,
	    		C_ENTRY_TYPE_6                => C_PF1_ENTRY_TYPE_6,
	    		C_ENTRY_BAR_6                 => C_PF1_ENTRY_BAR_6,
	    		C_ENTRY_ADDR_6                => C_PF1_ENTRY_ADDR_6,
	    		C_ENTRY_MAJOR_VERSION_6       => C_PF1_ENTRY_MAJOR_VERSION_6,
	    		C_ENTRY_MINOR_VERSION_6       => C_PF1_ENTRY_MINOR_VERSION_6,
	    		C_ENTRY_VERSION_TYPE_6        => C_PF1_ENTRY_VERSION_TYPE_6,
	    		C_ENTRY_RSVD0_6               => C_PF1_ENTRY_RSVD0_6,
	    		C_ENTRY_TYPE_7                => C_PF1_ENTRY_TYPE_7,
	    		C_ENTRY_BAR_7                 => C_PF1_ENTRY_BAR_7,
	    		C_ENTRY_ADDR_7                => C_PF1_ENTRY_ADDR_7,
	    		C_ENTRY_MAJOR_VERSION_7       => C_PF1_ENTRY_MAJOR_VERSION_7,
	    		C_ENTRY_MINOR_VERSION_7       => C_PF1_ENTRY_MINOR_VERSION_7,
	    		C_ENTRY_VERSION_TYPE_7        => C_PF1_ENTRY_VERSION_TYPE_7,
	    		C_ENTRY_RSVD0_7               => C_PF1_ENTRY_RSVD0_7,
	    		C_ENTRY_TYPE_8                => C_PF1_ENTRY_TYPE_8,
	    		C_ENTRY_BAR_8                 => C_PF1_ENTRY_BAR_8,
	    		C_ENTRY_ADDR_8                => C_PF1_ENTRY_ADDR_8,
	    		C_ENTRY_MAJOR_VERSION_8       => C_PF1_ENTRY_MAJOR_VERSION_8,
	    		C_ENTRY_MINOR_VERSION_8       => C_PF1_ENTRY_MINOR_VERSION_8,
	    		C_ENTRY_VERSION_TYPE_8        => C_PF1_ENTRY_VERSION_TYPE_8,
	    		C_ENTRY_RSVD0_8               => C_PF1_ENTRY_RSVD0_8,
	    		C_ENTRY_TYPE_9                => C_PF1_ENTRY_TYPE_9,
	    		C_ENTRY_BAR_9                 => C_PF1_ENTRY_BAR_9,
	    		C_ENTRY_ADDR_9                => C_PF1_ENTRY_ADDR_9,
	    		C_ENTRY_MAJOR_VERSION_9       => C_PF1_ENTRY_MAJOR_VERSION_9,
	    		C_ENTRY_MINOR_VERSION_9       => C_PF1_ENTRY_MINOR_VERSION_9,
	    		C_ENTRY_VERSION_TYPE_9        => C_PF1_ENTRY_VERSION_TYPE_9,
	    		C_ENTRY_RSVD0_9               => C_PF1_ENTRY_RSVD0_9,
	    		C_ENTRY_TYPE_10               => C_PF1_ENTRY_TYPE_10,
	    		C_ENTRY_BAR_10                => C_PF1_ENTRY_BAR_10,
	    		C_ENTRY_ADDR_10               => C_PF1_ENTRY_ADDR_10,
	    		C_ENTRY_MAJOR_VERSION_10      => C_PF1_ENTRY_MAJOR_VERSION_10,
	    		C_ENTRY_MINOR_VERSION_10      => C_PF1_ENTRY_MINOR_VERSION_10,
	    		C_ENTRY_VERSION_TYPE_10       => C_PF1_ENTRY_VERSION_TYPE_10,
	    		C_ENTRY_RSVD0_10              => C_PF1_ENTRY_RSVD0_10,
	    		C_ENTRY_TYPE_11               => C_PF1_ENTRY_TYPE_11,
	    		C_ENTRY_BAR_11                => C_PF1_ENTRY_BAR_11,
	    		C_ENTRY_ADDR_11               => C_PF1_ENTRY_ADDR_11,
	    		C_ENTRY_MAJOR_VERSION_11      => C_PF1_ENTRY_MAJOR_VERSION_11,
	    		C_ENTRY_MINOR_VERSION_11      => C_PF1_ENTRY_MINOR_VERSION_11,
	    		C_ENTRY_VERSION_TYPE_11       => C_PF1_ENTRY_VERSION_TYPE_11,
	    		C_ENTRY_RSVD0_11              => C_PF1_ENTRY_RSVD0_11,
	    		C_ENTRY_TYPE_12               => C_PF1_ENTRY_TYPE_12,
	    		C_ENTRY_BAR_12                => C_PF1_ENTRY_BAR_12,
	    		C_ENTRY_ADDR_12               => C_PF1_ENTRY_ADDR_12,
	    		C_ENTRY_MAJOR_VERSION_12      => C_PF1_ENTRY_MAJOR_VERSION_12,
	    		C_ENTRY_MINOR_VERSION_12      => C_PF1_ENTRY_MINOR_VERSION_12,
	    		C_ENTRY_VERSION_TYPE_12       => C_PF1_ENTRY_VERSION_TYPE_12,
	    		C_ENTRY_RSVD0_12              => C_PF1_ENTRY_RSVD0_12,
	    		C_ENTRY_TYPE_13               => C_PF1_ENTRY_TYPE_13,
	    		C_ENTRY_BAR_13                => C_PF1_ENTRY_BAR_13,
	    		C_ENTRY_ADDR_13               => C_PF1_ENTRY_ADDR_13,
	    		C_ENTRY_MAJOR_VERSION_13      => C_PF1_ENTRY_MAJOR_VERSION_13,
	    		C_ENTRY_MINOR_VERSION_13      => C_PF1_ENTRY_MINOR_VERSION_13,
	    		C_ENTRY_VERSION_TYPE_13       => C_PF1_ENTRY_VERSION_TYPE_13,
	    		C_ENTRY_RSVD0_13              => C_PF1_ENTRY_RSVD0_13,
	    		C_ENTRY_TYPE_14               => C_PF1_ENTRY_TYPE_14,
	    		C_ENTRY_BAR_14                => C_PF1_ENTRY_BAR_14,
	    		C_ENTRY_ADDR_14               => C_PF1_ENTRY_ADDR_14,
	    		C_ENTRY_MAJOR_VERSION_14      => C_PF1_ENTRY_MAJOR_VERSION_14,
	    		C_ENTRY_MINOR_VERSION_14      => C_PF1_ENTRY_MINOR_VERSION_14,
	    		C_ENTRY_VERSION_TYPE_14       => C_PF1_ENTRY_VERSION_TYPE_14,
	    		C_ENTRY_RSVD0_14              => C_PF1_ENTRY_RSVD0_14,
	    		C_ENTRY_TYPE_15               => C_PF1_ENTRY_TYPE_15,
	    		C_ENTRY_BAR_15                => C_PF1_ENTRY_BAR_15,
	    		C_ENTRY_ADDR_15               => C_PF1_ENTRY_ADDR_15,
	    		C_ENTRY_MAJOR_VERSION_15      => C_PF1_ENTRY_MAJOR_VERSION_15,
	    		C_ENTRY_MINOR_VERSION_15      => C_PF1_ENTRY_MINOR_VERSION_15,
	    		C_ENTRY_VERSION_TYPE_15       => C_PF1_ENTRY_VERSION_TYPE_15,
	    		C_ENTRY_RSVD0_15              => C_PF1_ENTRY_RSVD0_15,     
	    		C_S_AXI_DATA_WIDTH            => C_PF1_S_AXI_DATA_WIDTH,
	    		C_S_AXI_ADDR_WIDTH            => C_PF1_S_AXI_ADDR_WIDTH,
	    		C_XDEVICEFAMILY               => C_XDEVICEFAMILY
	    	)
		    port map (
			    s_axi_aclk            		    => aclk_ctrl,
			    s_axi_aresetn         		    => aresetn_ctrl,		    	
		    	s_axi_awaddr              		=> s_axi_ctrl_pf1_awaddr,
		    	s_axi_awvalid             		=> s_axi_ctrl_pf1_awvalid,
		    	s_axi_awready             		=> s_axi_ctrl_pf1_awready,
		    	s_axi_wdata               		=> s_axi_ctrl_pf1_wdata,
		    	s_axi_wstrb               		=> s_axi_ctrl_pf1_wstrb,
		    	s_axi_wvalid              		=> s_axi_ctrl_pf1_wvalid,
		    	s_axi_wready              		=> s_axi_ctrl_pf1_wready,
		    	s_axi_bresp               		=> s_axi_ctrl_pf1_bresp,
		    	s_axi_bvalid              		=> s_axi_ctrl_pf1_bvalid,
		    	s_axi_bready              		=> s_axi_ctrl_pf1_bready,
		    	s_axi_araddr              		=> s_axi_ctrl_pf1_araddr,
		    	s_axi_arvalid             		=> s_axi_ctrl_pf1_arvalid,
		    	s_axi_arready             		=> s_axi_ctrl_pf1_arready,
		    	s_axi_rdata               		=> s_axi_ctrl_pf1_rdata,
		    	s_axi_rresp               		=> s_axi_ctrl_pf1_rresp,
		    	s_axi_rvalid              		=> s_axi_ctrl_pf1_rvalid,
		    	s_axi_rready              		=> s_axi_ctrl_pf1_rready
		    );
	
			end generate G_GENERATE_PF1;
			
		G_GENERATE_PF2: if (i = 2) generate

	  	-- Instantiate BAR Layout table
			BAR_LAYOUT_TABLE_inst_2 : entity hw_discovery_v1_0_0.hw_discovery_v1_0_0_bar_layout_table
	    	generic map (
	    	  C_NUM_SLOTS_BAR_LAYOUT_TABLE  => C_PF2_NUM_SLOTS_BAR_LAYOUT_TABLE,
	    	  C_ENTRY_TYPE_0                => C_PF2_ENTRY_TYPE_0,
	    	  C_ENTRY_BAR_0                 => C_PF2_ENTRY_BAR_0,
	    	  C_ENTRY_ADDR_0                => C_PF2_ENTRY_ADDR_0,
	    	  C_ENTRY_MAJOR_VERSION_0       => C_PF2_ENTRY_MAJOR_VERSION_0,
	    	  C_ENTRY_MINOR_VERSION_0       => C_PF2_ENTRY_MINOR_VERSION_0,
	    	  C_ENTRY_VERSION_TYPE_0        => C_PF2_ENTRY_VERSION_TYPE_0,
	    	  C_ENTRY_RSVD0_0               => C_PF2_ENTRY_RSVD0_0,
	    	  C_ENTRY_TYPE_1                => C_PF2_ENTRY_TYPE_1,
	    	  C_ENTRY_BAR_1                 => C_PF2_ENTRY_BAR_1,
	    	  C_ENTRY_ADDR_1                => C_PF2_ENTRY_ADDR_1,
	    	  C_ENTRY_MAJOR_VERSION_1       => C_PF2_ENTRY_MAJOR_VERSION_1,
	    	  C_ENTRY_MINOR_VERSION_1       => C_PF2_ENTRY_MINOR_VERSION_1,
	    	  C_ENTRY_VERSION_TYPE_1        => C_PF2_ENTRY_VERSION_TYPE_1,
	    	  C_ENTRY_RSVD0_1               => C_PF2_ENTRY_RSVD0_1,
	    	  C_ENTRY_TYPE_2                => C_PF2_ENTRY_TYPE_2,
	    	  C_ENTRY_BAR_2                 => C_PF2_ENTRY_BAR_2,
	    	  C_ENTRY_ADDR_2                => C_PF2_ENTRY_ADDR_2,
	    	  C_ENTRY_MAJOR_VERSION_2       => C_PF2_ENTRY_MAJOR_VERSION_2,
	    	  C_ENTRY_MINOR_VERSION_2       => C_PF2_ENTRY_MINOR_VERSION_2,
	    	  C_ENTRY_VERSION_TYPE_2        => C_PF2_ENTRY_VERSION_TYPE_2,
	    	  C_ENTRY_RSVD0_2               => C_PF2_ENTRY_RSVD0_2,
	    	  C_ENTRY_TYPE_3                => C_PF2_ENTRY_TYPE_3,
	    	  C_ENTRY_BAR_3                 => C_PF2_ENTRY_BAR_3,
	    	  C_ENTRY_ADDR_3                => C_PF2_ENTRY_ADDR_3,
	    	  C_ENTRY_MAJOR_VERSION_3       => C_PF2_ENTRY_MAJOR_VERSION_3,
	    	  C_ENTRY_MINOR_VERSION_3       => C_PF2_ENTRY_MINOR_VERSION_3,
	    	  C_ENTRY_VERSION_TYPE_3        => C_PF2_ENTRY_VERSION_TYPE_3,
	    	  C_ENTRY_RSVD0_3               => C_PF2_ENTRY_RSVD0_3,
	    	  C_ENTRY_TYPE_4                => C_PF2_ENTRY_TYPE_4,
	    	  C_ENTRY_BAR_4                 => C_PF2_ENTRY_BAR_4,
	    	  C_ENTRY_ADDR_4                => C_PF2_ENTRY_ADDR_4,
	    	  C_ENTRY_MAJOR_VERSION_4       => C_PF2_ENTRY_MAJOR_VERSION_4,
	    	  C_ENTRY_MINOR_VERSION_4       => C_PF2_ENTRY_MINOR_VERSION_4,
	    	  C_ENTRY_VERSION_TYPE_4        => C_PF2_ENTRY_VERSION_TYPE_4,
	    	  C_ENTRY_RSVD0_4               => C_PF2_ENTRY_RSVD0_4,
	    	  C_ENTRY_TYPE_5                => C_PF2_ENTRY_TYPE_5,
	    	  C_ENTRY_BAR_5                 => C_PF2_ENTRY_BAR_5,
	    	  C_ENTRY_ADDR_5                => C_PF2_ENTRY_ADDR_5,
	    	  C_ENTRY_MAJOR_VERSION_5       => C_PF2_ENTRY_MAJOR_VERSION_5,
	    	  C_ENTRY_MINOR_VERSION_5       => C_PF2_ENTRY_MINOR_VERSION_5,
	    	  C_ENTRY_VERSION_TYPE_5        => C_PF2_ENTRY_VERSION_TYPE_5,
	    	  C_ENTRY_RSVD0_5               => C_PF2_ENTRY_RSVD0_5,
	    	  C_ENTRY_TYPE_6                => C_PF2_ENTRY_TYPE_6,
	    	  C_ENTRY_BAR_6                 => C_PF2_ENTRY_BAR_6,
	    	  C_ENTRY_ADDR_6                => C_PF2_ENTRY_ADDR_6,
	    	  C_ENTRY_MAJOR_VERSION_6       => C_PF2_ENTRY_MAJOR_VERSION_6,
	    	  C_ENTRY_MINOR_VERSION_6       => C_PF2_ENTRY_MINOR_VERSION_6,
	    	  C_ENTRY_VERSION_TYPE_6        => C_PF2_ENTRY_VERSION_TYPE_6,
	    	  C_ENTRY_RSVD0_6               => C_PF2_ENTRY_RSVD0_6,
	    	  C_ENTRY_TYPE_7                => C_PF2_ENTRY_TYPE_7,
	    	  C_ENTRY_BAR_7                 => C_PF2_ENTRY_BAR_7,
	    	  C_ENTRY_ADDR_7                => C_PF2_ENTRY_ADDR_7,
	    	  C_ENTRY_MAJOR_VERSION_7       => C_PF2_ENTRY_MAJOR_VERSION_7,
	    	  C_ENTRY_MINOR_VERSION_7       => C_PF2_ENTRY_MINOR_VERSION_7,
	    	  C_ENTRY_VERSION_TYPE_7        => C_PF2_ENTRY_VERSION_TYPE_7,
	    	  C_ENTRY_RSVD0_7               => C_PF2_ENTRY_RSVD0_7,
	    	  C_ENTRY_TYPE_8                => C_PF2_ENTRY_TYPE_8,
	    	  C_ENTRY_BAR_8                 => C_PF2_ENTRY_BAR_8,
	    	  C_ENTRY_ADDR_8                => C_PF2_ENTRY_ADDR_8,
	    	  C_ENTRY_MAJOR_VERSION_8       => C_PF2_ENTRY_MAJOR_VERSION_8,
	    	  C_ENTRY_MINOR_VERSION_8       => C_PF2_ENTRY_MINOR_VERSION_8,
	    	  C_ENTRY_VERSION_TYPE_8        => C_PF2_ENTRY_VERSION_TYPE_8,
	    	  C_ENTRY_RSVD0_8               => C_PF2_ENTRY_RSVD0_8,
	    	  C_ENTRY_TYPE_9                => C_PF2_ENTRY_TYPE_9,
	    	  C_ENTRY_BAR_9                 => C_PF2_ENTRY_BAR_9,
	    	  C_ENTRY_ADDR_9                => C_PF2_ENTRY_ADDR_9,
	    	  C_ENTRY_MAJOR_VERSION_9       => C_PF2_ENTRY_MAJOR_VERSION_9,
	    	  C_ENTRY_MINOR_VERSION_9       => C_PF2_ENTRY_MINOR_VERSION_9,
	    	  C_ENTRY_VERSION_TYPE_9        => C_PF2_ENTRY_VERSION_TYPE_9,
	    	  C_ENTRY_RSVD0_9               => C_PF2_ENTRY_RSVD0_9,
	    	  C_ENTRY_TYPE_10               => C_PF2_ENTRY_TYPE_10,
	    	  C_ENTRY_BAR_10                => C_PF2_ENTRY_BAR_10,
	    	  C_ENTRY_ADDR_10               => C_PF2_ENTRY_ADDR_10,
	    	  C_ENTRY_MAJOR_VERSION_10      => C_PF2_ENTRY_MAJOR_VERSION_10,
	    	  C_ENTRY_MINOR_VERSION_10      => C_PF2_ENTRY_MINOR_VERSION_10,
	    	  C_ENTRY_VERSION_TYPE_10       => C_PF2_ENTRY_VERSION_TYPE_10,
	    	  C_ENTRY_RSVD0_10              => C_PF2_ENTRY_RSVD0_10,
	    	  C_ENTRY_TYPE_11               => C_PF2_ENTRY_TYPE_11,
	    	  C_ENTRY_BAR_11                => C_PF2_ENTRY_BAR_11,
	    	  C_ENTRY_ADDR_11               => C_PF2_ENTRY_ADDR_11,
	    	  C_ENTRY_MAJOR_VERSION_11      => C_PF2_ENTRY_MAJOR_VERSION_11,
	    	  C_ENTRY_MINOR_VERSION_11      => C_PF2_ENTRY_MINOR_VERSION_11,
	    	  C_ENTRY_VERSION_TYPE_11       => C_PF2_ENTRY_VERSION_TYPE_11,
	    	  C_ENTRY_RSVD0_11              => C_PF2_ENTRY_RSVD0_11,
	    	  C_ENTRY_TYPE_12               => C_PF2_ENTRY_TYPE_12,
	    	  C_ENTRY_BAR_12                => C_PF2_ENTRY_BAR_12,
	    	  C_ENTRY_ADDR_12               => C_PF2_ENTRY_ADDR_12,
	    	  C_ENTRY_MAJOR_VERSION_12      => C_PF2_ENTRY_MAJOR_VERSION_12,
	    	  C_ENTRY_MINOR_VERSION_12      => C_PF2_ENTRY_MINOR_VERSION_12,
	    	  C_ENTRY_VERSION_TYPE_12       => C_PF2_ENTRY_VERSION_TYPE_12,
	    	  C_ENTRY_RSVD0_12              => C_PF2_ENTRY_RSVD0_12,
	    	  C_ENTRY_TYPE_13               => C_PF2_ENTRY_TYPE_13,
	    	  C_ENTRY_BAR_13                => C_PF2_ENTRY_BAR_13,
	    	  C_ENTRY_ADDR_13               => C_PF2_ENTRY_ADDR_13,
	    	  C_ENTRY_MAJOR_VERSION_13      => C_PF2_ENTRY_MAJOR_VERSION_13,
	    	  C_ENTRY_MINOR_VERSION_13      => C_PF2_ENTRY_MINOR_VERSION_13,
	    	  C_ENTRY_VERSION_TYPE_13       => C_PF2_ENTRY_VERSION_TYPE_13,
	    	  C_ENTRY_RSVD0_13              => C_PF2_ENTRY_RSVD0_13,
	    	  C_ENTRY_TYPE_14               => C_PF2_ENTRY_TYPE_14,
	    	  C_ENTRY_BAR_14                => C_PF2_ENTRY_BAR_14,
	    	  C_ENTRY_ADDR_14               => C_PF2_ENTRY_ADDR_14,
	    	  C_ENTRY_MAJOR_VERSION_14      => C_PF2_ENTRY_MAJOR_VERSION_14,
	    	  C_ENTRY_MINOR_VERSION_14      => C_PF2_ENTRY_MINOR_VERSION_14,
	    	  C_ENTRY_VERSION_TYPE_14       => C_PF2_ENTRY_VERSION_TYPE_14,
	    	  C_ENTRY_RSVD0_14              => C_PF2_ENTRY_RSVD0_14,
	    	  C_ENTRY_TYPE_15               => C_PF2_ENTRY_TYPE_15,
	    	  C_ENTRY_BAR_15                => C_PF2_ENTRY_BAR_15,
	    	  C_ENTRY_ADDR_15               => C_PF2_ENTRY_ADDR_15,
	    	  C_ENTRY_MAJOR_VERSION_15      => C_PF2_ENTRY_MAJOR_VERSION_15,
	    	  C_ENTRY_MINOR_VERSION_15      => C_PF2_ENTRY_MINOR_VERSION_15,
	    	  C_ENTRY_VERSION_TYPE_15       => C_PF2_ENTRY_VERSION_TYPE_15,
	    	  C_ENTRY_RSVD0_15              => C_PF2_ENTRY_RSVD0_15,     
	    	  C_S_AXI_DATA_WIDTH            => C_PF2_S_AXI_DATA_WIDTH,
	    	  C_S_AXI_ADDR_WIDTH            => C_PF2_S_AXI_ADDR_WIDTH,
	    	  C_XDEVICEFAMILY               => C_XDEVICEFAMILY
	    	)
		    port map (
			    s_axi_aclk            		    => aclk_ctrl,
			    s_axi_aresetn         		    => aresetn_ctrl,		    	
		    	s_axi_awaddr              		=> s_axi_ctrl_pf2_awaddr,
		    	s_axi_awvalid             		=> s_axi_ctrl_pf2_awvalid,
		    	s_axi_awready             		=> s_axi_ctrl_pf2_awready,
		    	s_axi_wdata               		=> s_axi_ctrl_pf2_wdata,
		    	s_axi_wstrb               		=> s_axi_ctrl_pf2_wstrb,
		    	s_axi_wvalid              		=> s_axi_ctrl_pf2_wvalid,
		    	s_axi_wready              		=> s_axi_ctrl_pf2_wready,
		    	s_axi_bresp               		=> s_axi_ctrl_pf2_bresp,
		    	s_axi_bvalid              		=> s_axi_ctrl_pf2_bvalid,
		    	s_axi_bready              		=> s_axi_ctrl_pf2_bready,
		    	s_axi_araddr              		=> s_axi_ctrl_pf2_araddr,
		    	s_axi_arvalid             		=> s_axi_ctrl_pf2_arvalid,
		    	s_axi_arready             		=> s_axi_ctrl_pf2_arready,
		    	s_axi_rdata               		=> s_axi_ctrl_pf2_rdata,
		    	s_axi_rresp               		=> s_axi_ctrl_pf2_rresp,
		    	s_axi_rvalid              		=> s_axi_ctrl_pf2_rvalid,
		    	s_axi_rready              		=> s_axi_ctrl_pf2_rready
		    );
	
			end generate G_GENERATE_PF2;
			
		G_GENERATE_PF3 : if (i = 3) generate

	  	-- Instantiate BAR Layout table
			BAR_LAYOUT_TABLE_inst_3 : entity hw_discovery_v1_0_0.hw_discovery_v1_0_0_bar_layout_table
	    	generic map (
	    		C_NUM_SLOTS_BAR_LAYOUT_TABLE  => C_PF3_NUM_SLOTS_BAR_LAYOUT_TABLE,
	    		C_ENTRY_TYPE_0                => C_PF3_ENTRY_TYPE_0,
	    		C_ENTRY_BAR_0                 => C_PF3_ENTRY_BAR_0,
	    		C_ENTRY_ADDR_0                => C_PF3_ENTRY_ADDR_0,
	    		C_ENTRY_MAJOR_VERSION_0       => C_PF3_ENTRY_MAJOR_VERSION_0,
	    		C_ENTRY_MINOR_VERSION_0       => C_PF3_ENTRY_MINOR_VERSION_0,
	    		C_ENTRY_VERSION_TYPE_0        => C_PF3_ENTRY_VERSION_TYPE_0,
	    		C_ENTRY_RSVD0_0               => C_PF3_ENTRY_RSVD0_0,
	    		C_ENTRY_TYPE_1                => C_PF3_ENTRY_TYPE_1,
	    		C_ENTRY_BAR_1                 => C_PF3_ENTRY_BAR_1,
	    		C_ENTRY_ADDR_1                => C_PF3_ENTRY_ADDR_1,
	    		C_ENTRY_MAJOR_VERSION_1       => C_PF3_ENTRY_MAJOR_VERSION_1,
	    		C_ENTRY_MINOR_VERSION_1       => C_PF3_ENTRY_MINOR_VERSION_1,
	    		C_ENTRY_VERSION_TYPE_1        => C_PF3_ENTRY_VERSION_TYPE_1,
	    		C_ENTRY_RSVD0_1               => C_PF3_ENTRY_RSVD0_1,
	    		C_ENTRY_TYPE_2                => C_PF3_ENTRY_TYPE_2,
	    		C_ENTRY_BAR_2                 => C_PF3_ENTRY_BAR_2,
	    		C_ENTRY_ADDR_2                => C_PF3_ENTRY_ADDR_2,
	    		C_ENTRY_MAJOR_VERSION_2       => C_PF3_ENTRY_MAJOR_VERSION_2,
	    		C_ENTRY_MINOR_VERSION_2       => C_PF3_ENTRY_MINOR_VERSION_2,
	    		C_ENTRY_VERSION_TYPE_2        => C_PF3_ENTRY_VERSION_TYPE_2,
	    		C_ENTRY_RSVD0_2               => C_PF3_ENTRY_RSVD0_2,
	    		C_ENTRY_TYPE_3                => C_PF3_ENTRY_TYPE_3,
	    		C_ENTRY_BAR_3                 => C_PF3_ENTRY_BAR_3,
	    		C_ENTRY_ADDR_3                => C_PF3_ENTRY_ADDR_3,
	    		C_ENTRY_MAJOR_VERSION_3       => C_PF3_ENTRY_MAJOR_VERSION_3,
	    		C_ENTRY_MINOR_VERSION_3       => C_PF3_ENTRY_MINOR_VERSION_3,
	    		C_ENTRY_VERSION_TYPE_3        => C_PF3_ENTRY_VERSION_TYPE_3,
	    		C_ENTRY_RSVD0_3               => C_PF3_ENTRY_RSVD0_3,
	    		C_ENTRY_TYPE_4                => C_PF3_ENTRY_TYPE_4,
	    		C_ENTRY_BAR_4                 => C_PF3_ENTRY_BAR_4,
	    		C_ENTRY_ADDR_4                => C_PF3_ENTRY_ADDR_4,
	    		C_ENTRY_MAJOR_VERSION_4       => C_PF3_ENTRY_MAJOR_VERSION_4,
	    		C_ENTRY_MINOR_VERSION_4       => C_PF3_ENTRY_MINOR_VERSION_4,
	    		C_ENTRY_VERSION_TYPE_4        => C_PF3_ENTRY_VERSION_TYPE_4,
	    		C_ENTRY_RSVD0_4               => C_PF3_ENTRY_RSVD0_4,
	    		C_ENTRY_TYPE_5                => C_PF3_ENTRY_TYPE_5,
	    		C_ENTRY_BAR_5                 => C_PF3_ENTRY_BAR_5,
	    		C_ENTRY_ADDR_5                => C_PF3_ENTRY_ADDR_5,
	    		C_ENTRY_MAJOR_VERSION_5       => C_PF3_ENTRY_MAJOR_VERSION_5,
	    		C_ENTRY_MINOR_VERSION_5       => C_PF3_ENTRY_MINOR_VERSION_5,
	    		C_ENTRY_VERSION_TYPE_5        => C_PF3_ENTRY_VERSION_TYPE_5,
	    		C_ENTRY_RSVD0_5               => C_PF3_ENTRY_RSVD0_5,
	    		C_ENTRY_TYPE_6                => C_PF3_ENTRY_TYPE_6,
	    		C_ENTRY_BAR_6                 => C_PF3_ENTRY_BAR_6,
	    		C_ENTRY_ADDR_6                => C_PF3_ENTRY_ADDR_6,
	    		C_ENTRY_MAJOR_VERSION_6       => C_PF3_ENTRY_MAJOR_VERSION_6,
	    		C_ENTRY_MINOR_VERSION_6       => C_PF3_ENTRY_MINOR_VERSION_6,
	    		C_ENTRY_VERSION_TYPE_6        => C_PF3_ENTRY_VERSION_TYPE_6,
	    		C_ENTRY_RSVD0_6               => C_PF3_ENTRY_RSVD0_6,
	    		C_ENTRY_TYPE_7                => C_PF3_ENTRY_TYPE_7,
	    		C_ENTRY_BAR_7                 => C_PF3_ENTRY_BAR_7,
	    		C_ENTRY_ADDR_7                => C_PF3_ENTRY_ADDR_7,
	    		C_ENTRY_MAJOR_VERSION_7       => C_PF3_ENTRY_MAJOR_VERSION_7,
	    		C_ENTRY_MINOR_VERSION_7       => C_PF3_ENTRY_MINOR_VERSION_7,
	    		C_ENTRY_VERSION_TYPE_7        => C_PF3_ENTRY_VERSION_TYPE_7,
	    		C_ENTRY_RSVD0_7               => C_PF3_ENTRY_RSVD0_7,
	    		C_ENTRY_TYPE_8                => C_PF3_ENTRY_TYPE_8,
	    		C_ENTRY_BAR_8                 => C_PF3_ENTRY_BAR_8,
	    		C_ENTRY_ADDR_8                => C_PF3_ENTRY_ADDR_8,
	    		C_ENTRY_MAJOR_VERSION_8       => C_PF3_ENTRY_MAJOR_VERSION_8,
	    		C_ENTRY_MINOR_VERSION_8       => C_PF3_ENTRY_MINOR_VERSION_8,
	    		C_ENTRY_VERSION_TYPE_8        => C_PF3_ENTRY_VERSION_TYPE_8,
	    		C_ENTRY_RSVD0_8               => C_PF3_ENTRY_RSVD0_8,
	    		C_ENTRY_TYPE_9                => C_PF3_ENTRY_TYPE_9,
	    		C_ENTRY_BAR_9                 => C_PF3_ENTRY_BAR_9,
	    		C_ENTRY_ADDR_9                => C_PF3_ENTRY_ADDR_9,
	    		C_ENTRY_MAJOR_VERSION_9       => C_PF3_ENTRY_MAJOR_VERSION_9,
	    		C_ENTRY_MINOR_VERSION_9       => C_PF3_ENTRY_MINOR_VERSION_9,
	    		C_ENTRY_VERSION_TYPE_9        => C_PF3_ENTRY_VERSION_TYPE_9,
	    		C_ENTRY_RSVD0_9               => C_PF3_ENTRY_RSVD0_9,
	    		C_ENTRY_TYPE_10               => C_PF3_ENTRY_TYPE_10,
	    		C_ENTRY_BAR_10                => C_PF3_ENTRY_BAR_10,
	    		C_ENTRY_ADDR_10               => C_PF3_ENTRY_ADDR_10,
	    		C_ENTRY_MAJOR_VERSION_10      => C_PF3_ENTRY_MAJOR_VERSION_10,
	    		C_ENTRY_MINOR_VERSION_10      => C_PF3_ENTRY_MINOR_VERSION_10,
	    		C_ENTRY_VERSION_TYPE_10       => C_PF3_ENTRY_VERSION_TYPE_10,
	    		C_ENTRY_RSVD0_10              => C_PF3_ENTRY_RSVD0_10,
	    		C_ENTRY_TYPE_11               => C_PF3_ENTRY_TYPE_11,
	    		C_ENTRY_BAR_11                => C_PF3_ENTRY_BAR_11,
	    		C_ENTRY_ADDR_11               => C_PF3_ENTRY_ADDR_11,
	    		C_ENTRY_MAJOR_VERSION_11      => C_PF3_ENTRY_MAJOR_VERSION_11,
	    		C_ENTRY_MINOR_VERSION_11      => C_PF3_ENTRY_MINOR_VERSION_11,
	    		C_ENTRY_VERSION_TYPE_11       => C_PF3_ENTRY_VERSION_TYPE_11,
	    		C_ENTRY_RSVD0_11              => C_PF3_ENTRY_RSVD0_11,
	    		C_ENTRY_TYPE_12               => C_PF3_ENTRY_TYPE_12,
	    		C_ENTRY_BAR_12                => C_PF3_ENTRY_BAR_12,
	    		C_ENTRY_ADDR_12               => C_PF3_ENTRY_ADDR_12,
	    		C_ENTRY_MAJOR_VERSION_12      => C_PF3_ENTRY_MAJOR_VERSION_12,
	    		C_ENTRY_MINOR_VERSION_12      => C_PF3_ENTRY_MINOR_VERSION_12,
	    		C_ENTRY_VERSION_TYPE_12       => C_PF3_ENTRY_VERSION_TYPE_12,
	    		C_ENTRY_RSVD0_12              => C_PF3_ENTRY_RSVD0_12,
	    		C_ENTRY_TYPE_13               => C_PF3_ENTRY_TYPE_13,
	    		C_ENTRY_BAR_13                => C_PF3_ENTRY_BAR_13,
	    		C_ENTRY_ADDR_13               => C_PF3_ENTRY_ADDR_13,
	    		C_ENTRY_MAJOR_VERSION_13      => C_PF3_ENTRY_MAJOR_VERSION_13,
	    		C_ENTRY_MINOR_VERSION_13      => C_PF3_ENTRY_MINOR_VERSION_13,
	    		C_ENTRY_VERSION_TYPE_13       => C_PF3_ENTRY_VERSION_TYPE_13,
	    		C_ENTRY_RSVD0_13              => C_PF3_ENTRY_RSVD0_13,
	    		C_ENTRY_TYPE_14               => C_PF3_ENTRY_TYPE_14,
	    		C_ENTRY_BAR_14                => C_PF3_ENTRY_BAR_14,
	    		C_ENTRY_ADDR_14               => C_PF3_ENTRY_ADDR_14,
	    		C_ENTRY_MAJOR_VERSION_14      => C_PF3_ENTRY_MAJOR_VERSION_14,
	    		C_ENTRY_MINOR_VERSION_14      => C_PF3_ENTRY_MINOR_VERSION_14,
	    		C_ENTRY_VERSION_TYPE_14       => C_PF3_ENTRY_VERSION_TYPE_14,
	    		C_ENTRY_RSVD0_14              => C_PF3_ENTRY_RSVD0_14,
	    		C_ENTRY_TYPE_15               => C_PF3_ENTRY_TYPE_15,
	    		C_ENTRY_BAR_15                => C_PF3_ENTRY_BAR_15,
	    		C_ENTRY_ADDR_15               => C_PF3_ENTRY_ADDR_15,
	    		C_ENTRY_MAJOR_VERSION_15      => C_PF3_ENTRY_MAJOR_VERSION_15,
	    		C_ENTRY_MINOR_VERSION_15      => C_PF3_ENTRY_MINOR_VERSION_15,
	    		C_ENTRY_VERSION_TYPE_15       => C_PF3_ENTRY_VERSION_TYPE_15,
	    		C_ENTRY_RSVD0_15              => C_PF3_ENTRY_RSVD0_15,     
	    		C_S_AXI_DATA_WIDTH            => C_PF3_S_AXI_DATA_WIDTH,
	    		C_S_AXI_ADDR_WIDTH            => C_PF3_S_AXI_ADDR_WIDTH,
	    		C_XDEVICEFAMILY               => C_XDEVICEFAMILY
	    		)
		    port map (
			    s_axi_aclk            		    => aclk_ctrl,
			    s_axi_aresetn         		    => aresetn_ctrl,		    	
		    	s_axi_awaddr              		=> s_axi_ctrl_pf3_awaddr,
		    	s_axi_awvalid             		=> s_axi_ctrl_pf3_awvalid,
		    	s_axi_awready             		=> s_axi_ctrl_pf3_awready,
		    	s_axi_wdata               		=> s_axi_ctrl_pf3_wdata,
		    	s_axi_wstrb               		=> s_axi_ctrl_pf3_wstrb,
		    	s_axi_wvalid              		=> s_axi_ctrl_pf3_wvalid,
		    	s_axi_wready              		=> s_axi_ctrl_pf3_wready,
		    	s_axi_bresp               		=> s_axi_ctrl_pf3_bresp,
		    	s_axi_bvalid              		=> s_axi_ctrl_pf3_bvalid,
		    	s_axi_bready              		=> s_axi_ctrl_pf3_bready,
		    	s_axi_araddr              		=> s_axi_ctrl_pf3_araddr,
		    	s_axi_arvalid             		=> s_axi_ctrl_pf3_arvalid,
		    	s_axi_arready             		=> s_axi_ctrl_pf3_arready,
		    	s_axi_rdata               		=> s_axi_ctrl_pf3_rdata,
		    	s_axi_rresp               		=> s_axi_ctrl_pf3_rresp,
		    	s_axi_rvalid              		=> s_axi_ctrl_pf3_rvalid,
		    	s_axi_rready              		=> s_axi_ctrl_pf3_rready
		    );
	
			end generate G_GENERATE_PF3;
			
		end generate G_GENERATE;

end architecture rtl;
