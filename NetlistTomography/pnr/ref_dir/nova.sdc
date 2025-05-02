# ####################################################################

#  Created by Genus(TM) Synthesis Solution 21.10-p002_1 on Thu Nov 09 14:52:30 PST 2023

# ####################################################################

set sdc_version 2.0

set_units -capacitance 1fF
set_units -time 1000ps

# Set the current design
current_design nova

set clk_period $::env(CLK_PERIOD)

create_clock -name "clk" -period $clk_period [get_ports clk]
set_clock_gating_check -setup 0.0 
set_wire_load_mode "enclosed"
