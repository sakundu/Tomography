# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
source lib_setup.tcl
source design_setup.tcl
set handoff_dir  "./"

#set util $env(utility)

set netlist ${handoff_dir}/${DESIGN}.v
set sdc ./${DESIGN}.sdc 
source mmmc_setup.tcl

set util $::env(UTIL)
setMultiCpuUsage -localCpu 16

set rptDir summaryReport/ 
set encDir enc/

if {![file exists $rptDir/]} {
  exec mkdir $rptDir/
}

if {![file exists $encDir/]} {
  exec mkdir $encDir/
}

# default settings
set init_pwr_net VDD
set init_gnd_net VSS

# default settings
set init_verilog "$netlist"
set init_design_netlisttype "Verilog"
set init_design_settop 1
set init_top_cell "$DESIGN"
set init_lef_file "$lefs"

# MCMM setup
init_design -setup {WC_VIEW} -hold {BC_VIEW}
set_power_analysis_mode -leakage_power_view WC_VIEW -dynamic_power_view WC_VIEW

set_interactive_constraint_modes {CON}
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation -cppr both

clearGlobalNets
globalNetConnect VDD -type pgpin -pin VDD -inst * -override
globalNetConnect VSS -type pgpin -pin VSS -inst * -override
globalNetConnect VDD -type tiehi -inst * -override
globalNetConnect VSS -type tielo -inst * -override


setOptMode -powerEffort low -leakageToDynamicRatio 0.5
setGenerateViaMode -auto true
generateVias

# basic path groups
createBasicPathGroups -expanded

## Generate the floorplan ##
if {[info exist ::env(PHY_SYNTH)] && $::env(PHY_SYNTH) == 1} {
  defIn ${handoff_dir}/${DESIGN}.def
} else {
  ## Check if the floorplan def exists or not
  #if {![file exists $floorplan_def]} {
    puts "Floorplan def does not exist. Initializing floorplan with utilization 0.5 and aspect ratio 1.0"
    floorPlan -r 1 $util 2 2 2 2
    puts "Placing the pins on the left boundary of the floorplan"
  #} else {
  #  defIn $floorplan_de../f
  #}
  ## Check if the macros are there or not ##
  set fixed_macro_count [llength [dbget [dbget top.insts.cell.subClass block -p2 ].pStatus fixed -e]]
  set macro_count [llength [dbget top.insts.cell.subClass block -e]]
  if { $macro_count > 0 && $fixed_macro_count == 0 } {
    source ${proj_dir}/scripts/util/place_pin.tcl
    place_pins
    setPlaceMode -place_global_align_macro true
    addHaloToBlock -allMacro $HALO_WIDTH $HALO_WIDTH $HALO_WIDTH $HALO_WIDTH
    place_design -concurrent_macros
    refine_macro_place
  } else {
    set fix_pin_count [llength [dbget top.terms.pStatus fixed -e]]
    if { $fix_pin_count == 0 } {
      setPlaceMode -place_global_place_io_pins true
    }
  }
}

### Write postSynth report ###
echo "Physical Design Stage, Core Area (um^2), Standard Cell Area (um^2), Macro Area (um^2), Total Power (mW), Wirelength(um), WS(ns), TNS(ns), Congestion(H), Congestion(V)" > ${DESIGN}_DETAILS.rpt
source ../util/extract_report.tcl
set rpt_post_synth [extract_report postSynth]
echo "$rpt_post_synth" >> ${DESIGN}_DETAILS.rpt

### Write out the def files ###
source write_required_def.tcl

### Add power plan ###
source ../util/pdn_config.tcl
source ../util/pdn_flow.tcl

saveDesign ${encDir}/${DESIGN}_floorplan.enc

setPlaceMode -place_detail_legalization_inst_gap 1
setFillerMode -fitGap true
setDesignMode -topRoutingLayer $TOP_ROUTING_LAYER
setDesignMode -bottomRoutingLayer 2 

## Fix here: We need to run place design only and then opt design ##
## Get the nets and instances with dontTouch false ##
set nets_ptr [dbget top.nets.dontTouch false -p -e ]
set ints_ptr [dbget top.insts.dontTouch false -p -e ]

foreach net_ptr $nets_ptr {
  dbset ${net_ptr}.dontTouch true
}

foreach int_ptr $ints_ptr {
  dbset ${int_ptr}.dontTouch true
}

## Check if seeded def exists or not ##
if { [info exists ::env(SEEDED_DEF)] && [file exists $::env(SEEDED_DEF)]} {
  defIn $::env(SEEDED_DEF)
  place_design -incremental
} else {
  place_design
}

## Write out the required info for blob placement ##
#source ${proj_dir}/../Scripts/gen_graph_invs.tcl
#mkdir -p blob_inputs
#cd blob_inputs
#write_blob_place_exp_info
#cd ..
saveDesign $encDir/${DESIGN}_only_placed.enc

## Update net dontTouch status ##
foreach net_ptr $nets_ptr {
  dbset ${net_ptr}.dontTouch false
}

foreach int_ptr $ints_ptr {
  dbset ${int_ptr}.dontTouch false
}

## Remove the Regions ##
if { [dbget top.fplan.groups -e] != "" } {
  deleteAllInstGroups
}

set rpt_pre_cts [extract_report preCTS]
echo "$rpt_pre_cts" >> ${DESIGN}_DETAILS.rpt

# place_opt_design -out_dir $rptDir -prefix place
optDesign -preCTS
saveDesign $encDir/${DESIGN}_placed.enc

set rpt_pre_cts [extract_report preCTS]
echo "$rpt_pre_cts" >> ${DESIGN}_DETAILS.rpt

set_ccopt_property post_conditioning_enable_routing_eco 1
set_ccopt_property -cts_def_lock_clock_sinks_after_routing true
setOptMode -unfixClkInstForOpt false

create_ccopt_clock_tree_spec
ccopt_design

set_interactive_constraint_modes [all_constraint_modes -active]
set_propagated_clock [all_clocks]
set_clock_propagation propagated

saveDesign $encDir/${DESIGN}_cts.enc
set rpt_post_cts [extract_report postCTS]
echo "$rpt_post_cts" >> ${DESIGN}_DETAILS.rpt

# ------------------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------------------
setNanoRouteMode -drouteVerboseViolationSummary 1
setNanoRouteMode -routeWithSiDriven true
setNanoRouteMode -routeWithTimingDriven true
setNanoRouteMode -routeUseAutoVia true

##Recommended by lib owners
# Prevent router modifying M1 pins shapes
setNanoRouteMode -routeWithViaInPin "1:1"
setNanoRouteMode -routeWithViaOnlyForStandardCellPin "1:1"

## limit VIAs to ongrid only for VIA1 (S1)
setNanoRouteMode -drouteOnGridOnly "via 1:1"
setNanoRouteMode -drouteAutoStop false
setNanoRouteMode -drouteExpAdvancedMarFix true
setNanoRouteMode -routeExpAdvancedTechnology true

#SM suggestion for solving long extraction runtime during GR
setNanoRouteMode -grouteExpWithTimingDriven false

routeDesign
#route_opt_design
saveDesign ${encDir}/${DESIGN}_route.enc

### Run DRC and LVS ###
verify_connectivity -error 0 -geom_connect -no_antenna
verify_drc -limit 0

set rpt_post_route [extract_report postRoute]
echo "$rpt_post_route" >> ${DESIGN}_DETAILS.rpt
defOut -netlist -floorplan -routing ${DESIGN}_route.def

#route_opt_design
#optDesign -postRoute
ecoRoute -fix_drc
set rpt_post_route [extract_report postRouteOpt]
echo "$rpt_post_route" >> ${DESIGN}_DETAILS.rpt

summaryReport -noHtml -outfile summaryReport/post_route.sum
saveDesign ${encDir}/${DESIGN}.enc
defOut -netlist -floorplan -routing ${DESIGN}.def

exit
