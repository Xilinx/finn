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
    
library hw_discovery_v1_0_0;    

entity hw_discovery_v1_0_0_pcie_vsec is
    generic(
    	  C_NUM_PFS                 : integer range 1 to 4           := 1;
        C_CAP_BASE_ADDR           : std_logic_vector(11 downto 0)  := x"480"; -- 0x480 default for PCIE4
        C_NEXT_CAP_ADDR           : std_logic_vector(11 downto 0)  := (others => '0');
        C_PF0_BAR_INDEX           : integer range 0 to 6           := 0;
        C_PF0_LOW_OFFSET          : std_logic_vector(27 downto 0)  := (others => '0');
        C_PF0_HIGH_OFFSET         : std_logic_vector(31 downto 0)  := (others => '0');
        C_PF1_BAR_INDEX           : integer range 0 to 6           := 0;
        C_PF1_LOW_OFFSET          : std_logic_vector(27 downto 0)  := (others => '0');
        C_PF1_HIGH_OFFSET         : std_logic_vector(31 downto 0)  := (others => '0');
        C_PF2_BAR_INDEX           : integer range 0 to 6           := 0;
        C_PF2_LOW_OFFSET          : std_logic_vector(27 downto 0)  := (others => '0');
        C_PF2_HIGH_OFFSET         : std_logic_vector(31 downto 0)  := (others => '0');
        C_PF3_BAR_INDEX           : integer range 0 to 6           := 0;
        C_PF3_LOW_OFFSET          : std_logic_vector(27 downto 0)  := (others => '0');
        C_PF3_HIGH_OFFSET         : std_logic_vector(31 downto 0)  := (others => '0');
        C_XDEVICEFAMILY           : string                         := "no_family"
    );
    port(

        -----------------------------------------------------------------------
        -- Clocks & Resets
        -----------------------------------------------------------------------

        aclk_pcie                  								: in  std_logic;
        aresetn_pcie               								: in  std_logic;

        -----------------------------------------------------------------------
        -- pcie4_cfg_ext Interface (aclk_pcie)
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
        -- pcie4_cfg_ext Interface (aclk_pcie)
        -----------------------------------------------------------------------

        m_pcie4_cfg_ext_function_number           : out std_logic_vector(15 downto 0);
        m_pcie4_cfg_ext_read_data                 : in  std_logic_vector(31 downto 0);
        m_pcie4_cfg_ext_read_data_valid           : in  std_logic;
        m_pcie4_cfg_ext_read_received             : out std_logic;
        m_pcie4_cfg_ext_register_number           : out std_logic_vector(9 downto 0);
        m_pcie4_cfg_ext_write_byte_enable         : out std_logic_vector(3 downto 0);
        m_pcie4_cfg_ext_write_data                : out std_logic_vector(31 downto 0);
        m_pcie4_cfg_ext_write_received            : out std_logic        
        
    );
end entity hw_discovery_v1_0_0_pcie_vsec;

architecture rtl of hw_discovery_v1_0_0_pcie_vsec is

-------------------------------------------------------------------------------
--
--      CONSTANTS
--
-------------------------------------------------------------------------------

constant  CAP_ID                : std_logic_vector(15 downto 0)             								:= x"000B";
constant  CAP_VERSION           : std_logic_vector(3 downto 0)              								:= x"1";
constant  VSEC_ID               : std_logic_vector(15 downto 0)             								:= x"0020";
constant  VSEC_REV              : std_logic_vector(3 downto 0)              								:= x"0";
constant  VSEC_LENGTH           : std_logic_vector(11 downto 0)             								:= x"010";
constant  NEXT_CAP_BASE_ADDR    : integer                              											:= (to_integer(unsigned(C_CAP_BASE_ADDR)) + 16);
constant  NEXT_CAP_CONFIG_ADDR  : integer                              											:= (to_integer(unsigned(C_NEXT_CAP_ADDR)));
constant  CAP_BASE_BYTE_ADDR    : integer                                   								:= (to_integer(unsigned(C_CAP_BASE_ADDR)) / 4);
constant  CAP_BASE_ADDR         : integer                                   								:= (to_integer(unsigned(C_CAP_BASE_ADDR)) / 16);
constant  ALF_VSEC_NXT_REG      : std_logic_vector(11 downto 0)   													:= std_logic_vector(to_unsigned(NEXT_CAP_BASE_ADDR, 12));	
constant  ALF_VSEC_CONFIG_NXT   : std_logic_vector(11 downto 0)   													:= std_logic_vector(to_unsigned(NEXT_CAP_CONFIG_ADDR, 12));	
constant  ALF_VSEC_BASE_REG     : std_logic_vector(7 downto 0)   														:= std_logic_vector(to_unsigned(CAP_BASE_ADDR, 8));	
constant  ALF_VSEC_REG_0        : std_logic_vector(s_pcie4_cfg_ext_register_number'RANGE)   := std_logic_vector(to_unsigned(CAP_BASE_BYTE_ADDR, s_pcie4_cfg_ext_register_number'LENGTH));
constant  ALF_VSEC_REG_1        : std_logic_vector(s_pcie4_cfg_ext_register_number'RANGE)   := std_logic_vector(to_unsigned((CAP_BASE_BYTE_ADDR + 1), s_pcie4_cfg_ext_register_number'LENGTH));
constant  ALF_VSEC_REG_2        : std_logic_vector(s_pcie4_cfg_ext_register_number'RANGE)   := std_logic_vector(to_unsigned((CAP_BASE_BYTE_ADDR + 2), s_pcie4_cfg_ext_register_number'LENGTH));
constant  ALF_VSEC_REG_3        : std_logic_vector(s_pcie4_cfg_ext_register_number'RANGE)   := std_logic_vector(to_unsigned((CAP_BASE_BYTE_ADDR + 3), s_pcie4_cfg_ext_register_number'LENGTH));
constant  PF0_BAR_INDEX         : std_logic_vector(2 downto 0)              								:= std_logic_vector(to_unsigned(C_PF0_BAR_INDEX, 3));
constant  PF1_BAR_INDEX         : std_logic_vector(2 downto 0)              								:= std_logic_vector(to_unsigned(C_PF1_BAR_INDEX, 3));
constant  PF2_BAR_INDEX         : std_logic_vector(2 downto 0)              								:= std_logic_vector(to_unsigned(C_PF2_BAR_INDEX, 3));
constant  PF3_BAR_INDEX         : std_logic_vector(2 downto 0)              								:= std_logic_vector(to_unsigned(C_PF3_BAR_INDEX, 3));
constant  NUM_PFS               : std_logic_vector(1 downto 0)  														:= std_logic_vector(to_unsigned((C_NUM_PFS - 1), 2));
	
-------------------------------------------------------------------------------
--
--      SIGNALS
--
-------------------------------------------------------------------------------

signal cfg_ext_read_data   			: std_logic_vector(31 downto 0)		:= (others => '0');
signal cfg_ext_read_data_valid  : std_logic                     	:= '0';

signal enable_m_cfg_ext  : std_logic                     	:= '0';

begin 
	
	G_GENERATE_M_PCIE4_NXT_CFG_EXT : if (NEXT_CAP_CONFIG_ADDR >= NEXT_CAP_BASE_ADDR) generate
		
		CF:
		process(aclk_pcie)
		
			begin
		
	    	if (rising_edge(aclk_pcie)) then
	    		
					m_pcie4_cfg_ext_function_number   <= s_pcie4_cfg_ext_function_number;
					m_pcie4_cfg_ext_read_received     <= s_pcie4_cfg_ext_read_received;
					m_pcie4_cfg_ext_register_number   <= s_pcie4_cfg_ext_register_number;
					m_pcie4_cfg_ext_write_byte_enable <= s_pcie4_cfg_ext_write_byte_enable;
					m_pcie4_cfg_ext_write_data        <= s_pcie4_cfg_ext_write_data;
					m_pcie4_cfg_ext_write_received    <= s_pcie4_cfg_ext_write_received;
					
					if (enable_m_cfg_ext = '1') then
  					s_pcie4_cfg_ext_read_data								<= m_pcie4_cfg_ext_read_data;
  					s_pcie4_cfg_ext_read_data_valid					<= m_pcie4_cfg_ext_read_data_valid;
  				else
  					s_pcie4_cfg_ext_read_data         <= cfg_ext_read_data;
  					s_pcie4_cfg_ext_read_data_valid   <= cfg_ext_read_data_valid;
  				end if;
					
				end if;
		
		end process;	
	
	end generate G_GENERATE_M_PCIE4_NXT_CFG_EXT;
	
	G_GENERATE_M_PCIE4_CFG_EXT : if (NEXT_CAP_CONFIG_ADDR < NEXT_CAP_BASE_ADDR) generate
		
		CF:
		process(aclk_pcie)
		
			begin
		
	    	if (rising_edge(aclk_pcie)) then
	    		
					s_pcie4_cfg_ext_read_data         <= cfg_ext_read_data;
					s_pcie4_cfg_ext_read_data_valid   <= cfg_ext_read_data_valid;

					m_pcie4_cfg_ext_function_number   <= (others => '0');
					m_pcie4_cfg_ext_read_received     <= '0';
					m_pcie4_cfg_ext_register_number   <= (others => '0');
					m_pcie4_cfg_ext_write_byte_enable <= (others => '0');
					m_pcie4_cfg_ext_write_data        <= (others => '0');
					m_pcie4_cfg_ext_write_received    <= '0';
					
				end if;
		
		end process;	
	
	end generate G_GENERATE_M_PCIE4_CFG_EXT;	
								
	RD:
	process(aclk_pcie)
	
	  variable func_num_var : std_logic_vector(1 downto 0);
	  variable reg_num_var  : std_logic_vector(1 downto 0);	  	
	
	begin
	
	    if (rising_edge(aclk_pcie)) then
	
        if (aresetn_pcie = '0') then
	        cfg_ext_read_data_valid   <= '0';
	        cfg_ext_read_data         <= (others => '0');
	        func_num_var      				:= s_pcie4_cfg_ext_function_number(1 downto 0);
	        reg_num_var      					:= s_pcie4_cfg_ext_register_number(1 downto 0);	        	
				  enable_m_cfg_ext          <= '0';
          cfg_ext_read_data_valid <= '0';
          cfg_ext_read_data       <= (others => '0');
				  
				else
	
	        -- default assignment
	        cfg_ext_read_data_valid   <= '0';
	        cfg_ext_read_data         <= (others => '0');
	        func_num_var      				:= s_pcie4_cfg_ext_function_number(1 downto 0);
	        reg_num_var      					:= s_pcie4_cfg_ext_register_number(1 downto 0);	        	
	        
	        if (s_pcie4_cfg_ext_read_received = '1') then
	        	
  					  enable_m_cfg_ext          <= '0';
	        		if (s_pcie4_cfg_ext_register_number(9 downto 2) = ALF_VSEC_BASE_REG) then 
	
		            -- default read response
		            cfg_ext_read_data_valid <= '1';
		            cfg_ext_read_data       <= (others => '0');
		
		            case func_num_var is
		            	
									when "00" => -- PF0            		
		
		             		case reg_num_var is
		             		
		             		  when "00" =>
		             		
		             		    -- PF0 Extended Capability Header
		             		
		             		    cfg_ext_read_data   <= C_NEXT_CAP_ADDR & CAP_VERSION & CAP_ID;
		             		
		             		  when "01" =>
		             		
		             		    -- VSEC Header - Identifies as Xilinx Additional List of Features (ALF)
		             		
		             		 		cfg_ext_read_data   <= VSEC_LENGTH & VSEC_REV & VSEC_ID;
		             		    
		             		  when "10" =>
		             		
		             		    -- ALF Field 1 (BAR Index & Low Offset)
		             		
		             		  	cfg_ext_read_data   <= C_PF0_LOW_OFFSET & '0' & PF0_BAR_INDEX;
		             		
		             		  when others =>
		             		
		             		    -- ALF Field 2 (High Offset)
		             		
		             		    cfg_ext_read_data   <= C_PF0_HIGH_OFFSET;
		             		
		             		end case;
		             			
									when "01" => -- PF2  
										
										if (NUM_PFS > "00") then
		
		             			case reg_num_var is
		             			
		             			  when "00" =>
		             			
		             			    -- PF1 Extended Capability Header
		             			
		             			    cfg_ext_read_data   <= C_NEXT_CAP_ADDR & CAP_VERSION & CAP_ID;
		             			
		             			  when "01" =>
		             			
		             			    -- VSEC Header - Identifies as Xilinx Additional List of Features (ALF)
		             			
		             			 		cfg_ext_read_data   <= VSEC_LENGTH & VSEC_REV & VSEC_ID;
		             			    
		             			  when "10" =>
		             			
		             			    -- ALF Field 1 (BAR Index & Low Offset)
		             			
		             			  	cfg_ext_read_data   <= C_PF1_LOW_OFFSET & '0' & PF1_BAR_INDEX;
		             			
		             			  when others =>
		             			
		             			    -- ALF Field 2 (High Offset)
		             			
		             			    cfg_ext_read_data   <= C_PF1_HIGH_OFFSET;
		             			
		             			end case;
		             				
										end if;             				
		             			
									when "10" => -- PF2
		
										if (NUM_PFS(1) = '1') then
		
		             			case reg_num_var is
		             			
		             			  when "00" =>
		             			
		             			    -- PF0 Extended Capability Header
		             			
		             			    cfg_ext_read_data   <= C_NEXT_CAP_ADDR & CAP_VERSION & CAP_ID;
		             			
		             			  when "01" =>
		             			
		             			    -- VSEC Header - Identifies as Xilinx Additional List of Features (ALF)
		             			
		             			 		cfg_ext_read_data   <= VSEC_LENGTH & VSEC_REV & VSEC_ID;
		             			    
		             			  when "10" =>
		             			
		             			    -- ALF Field 1 (BAR Index & Low Offset)
		             			
		             			  	cfg_ext_read_data   <= C_PF2_LOW_OFFSET & '0' & PF2_BAR_INDEX;
		             			
		             			  when others =>
		             			
		             			    -- ALF Field 2 (High Offset)
		             			
		             			    cfg_ext_read_data   <= C_PF2_HIGH_OFFSET;
		             			
		             			end case;
		             				
										end if;             				
		             			
									when others => -- PF3
										
										if (NUM_PFS = "11") then								
		
		             			case reg_num_var is
		             			
		             			  when "00" =>
		             			
		             			    -- PF0 Extended Capability Header
		             			
		             			    cfg_ext_read_data   <= C_NEXT_CAP_ADDR & CAP_VERSION & CAP_ID;
		             			
		             			  when "01" =>
		             			
		             			    -- VSEC Header - Identifies as Xilinx Additional List of Features (ALF)
		             			
		             			 		cfg_ext_read_data   <= VSEC_LENGTH & VSEC_REV & VSEC_ID;
		             			    
		             			  when "10" =>
		             			
		             			    -- ALF Field 1 (BAR Index & Low Offset)
		             			
		             			  	cfg_ext_read_data   <= C_PF3_LOW_OFFSET & '0' & PF3_BAR_INDEX;
		             			
		             			  when others =>
		             			
		             			    -- ALF Field 2 (High Offset)
		             			
		             			    cfg_ext_read_data   <= C_PF3_HIGH_OFFSET;
		             			
		             			end case;
		             				
									end if;             				
		             			
		            end case;
		            	
						elsif (NEXT_CAP_CONFIG_ADDR < NEXT_CAP_BASE_ADDR) then
							
	            cfg_ext_read_data_valid <= '1';
	            cfg_ext_read_data       <= (others => '0');
							
						elsif (s_pcie4_cfg_ext_register_number(9 downto 2) >= ALF_VSEC_CONFIG_NXT(11 downto 4)) then
							
							enable_m_cfg_ext        <= '1';
		        	cfg_ext_read_data_valid <= '0';
		        	cfg_ext_read_data       <= (others => '0');
							
						else
							
	            cfg_ext_read_data_valid <= '1';
	            cfg_ext_read_data       <= (others => '0');
							
						end if;
	
	        end if;
        end if;
	
	    end if;
	
	end process;

end architecture rtl;
