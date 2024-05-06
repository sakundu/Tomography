# This script was written and developed by xxx. However,
# the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help
# promote and foster the next generation of innovators.
set proj_dir "$::env(PROJ_DIR)"
set DESIGN $env(DESIGN) 
set TECH $env(TECH)
source ${proj_dir}/rtl/${DESIGN}/design_setup.tcl
source ${proj_dir}/inputs/${TECH}/util/lib_setup.tcl

setMultiCpuUsage -localCpu 16


set rptDir summaryReport/ 
set encDir enc/

if {![file exists $rptDir/]} {
  exec mkdir $rptDir/
}

if {![file exists $encDir/]} {
  exec mkdir $encDir/
}


source $env(DESIGN_DIR)/enc/${DESIGN}_cts.enc
source $env(ROUTE_BLKG)

### Write postSynth report ###
echo "Physical Design Stage, Core Area (um^2), Standard Cell Area (um^2), Macro Area (um^2), Total Power (mW), Wirelength(um), WS(ns), TNS(ns), Congestion(H), Congestion(V)" > ${DESIGN}_DETAILS.rpt
source ${proj_dir}/util/extract_report.tcl
set rpt_post_cts [extract_report postCTS]
echo "$rpt_post_cts" >> ${DESIGN}_DETAILS.rpt

# ------------------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------------------
setNanoRouteMode -drouteVerboseViolationSummary 1
setNanoRouteMode -routeWithSiDriven true
setNanoRouteMode -routeWithTimingDriven true
setNanoRouteMode -routeUseAutoVia true
setNanoRouteMode -drouteEndIteration 20

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

deleteRouteBlk -all

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
#defOut -netlist -floorplan -routing ${DESIGN}.def

### Run DRC and LVS ###
verify_connectivity -error 0 -geom_connect -no_antenna
verify_drc -limit 0

exit
