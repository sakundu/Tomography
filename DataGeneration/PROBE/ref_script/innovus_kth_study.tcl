# This script was written and developed by XXXX. However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.

setMultiCpuUsage -localCpu 4

## [User setting] PDK path ##
set tech_dir    "./NanGate45"
####
set lef_dir     "${tech_dir}/lef"

set TECH_LEF    "${lef_dir}/NangateOpenCellLibrary.tech.lef"
set SC_LEF      "${lef_dir}/NangateOpenCellLibrary.macro.mod.lef"
set ADDITIONAL_LEFS ""

set lib_dir     "${tech_dir}/lib"
set ALL_LEFS    "${TECH_LEF} ${SC_LEF} ${ADDITIONAL_LEFS}"
set ALL_LIBS    "${lib_dir}/NangateOpenCellLibrary_typical.lib"
set LIB_BC      ${ALL_LIBS}
set LIB_WC      ${ALL_LIBS}


set netlist $::env(NETLIST_FILE)
set sdc $::env(DESIGN_SDC)
set design $::env(DESIGN_NAME)

### Kth Study Parameter ###
set util $::env(UTILIZATION)
set aspect_ratio "1"
set k $::env(K)
set no_row $::env(SIZE)
set no_col $::env(SIZE)

# default settings
set init_verilog "$netlist"
set init_design_netlisttype "Verilog"
set init_design_settop 1
set init_top_cell "$design"
set init_lef_file "${ALL_LEFS}"

set init_pwr_net "VDD"
set init_gnd_net "VSS"

# MCMM setup
create_library_set -name WC_LIB -timing ${LIB_WC}
create_library_set -name BC_LIB -timing ${LIB_BC}

create_delay_corner -name WC -library_set WC_LIB
create_delay_corner -name BC -library_set BC_LIB

create_constraint_mode -name CON -sdc_file $sdc

create_analysis_view -name WC_VIEW -delay_corner WC -constraint_mode CON
create_analysis_view -name BC_VIEW -delay_corner BC -constraint_mode CON

init_design -setup {WC_VIEW} -hold {BC_VIEW}

set width [dbget [dbget top.insts.name g_1_1 -p ].cell.size_x]
set height [dbget [dbget top.insts.name g_1_1 -p ].cell.size_y]
set box_height [ expr ($height*$no_row)]
set box_width [ expr ($width/$util)*$no_col]

puts "FPLAN BOX HEIGHT:$box_height WIDTH:$box_width"
# Add floorPlan command
floorPlan -s $box_width $box_height 0 0 0 0

# Add pin placement command
editPin -pin [dbget top.terms.name] -snap TRACK -layer 4

set inputs $::env(DETAILS)
## [User setting] python command ##
set python_exe "./python"
####
set cmd "exec $python_exe ./gen_innovus_swap_place.py $height $width $util $no_row $no_col $k $inputs" 
eval $cmd > ./place_innovus.tcl

# $k $height $width $util $sub_k > ./place_innovus.tcl

source ./place_innovus.tcl
source ./extract_rpt.tcl
source ./extract_data.tcl
defOut -floorplan -netlist ./innovus_place.def

setDesignMode -bottomRoutingLayer 2
setDesignMode -topRoutingLayer 10


saveDesign ./enc/${design}_place.enc
freeDesign
source ./enc/${design}_place.enc
write_features

deleteRouteBlk -all
globalDetailRoute

# reports
summaryReport  -noHtml -outfile ./rpt/invs_route_summary.rpt
summaryReport  -outdir ./rpt/invs_route_summary
drc_markers_rpt post_route_drc.rpt

saveDesign ./enc/${design}.enc

verify_drc -limit 0
ecoRoute -fix_drc
saveDesign ./enc/${design}_ecoRoute.enc
drc_markers_rpt post_eco_fix_drc.rpt
write_labels

set db_unit [dbget head.dbUnits]
set run_dir $::env(PWD)

set python_script "./extract_data_features.py"
set cmd "exec $python_exe $python_script $db_unit $run_dir"
eval $cmd

exit
