## Load the floorplan design
setMultiCpuUsage -localCpu 16
set design $::env(DESIGN)
set site_count_h [expr {[info exists ::env(SITE_COUNT_H)] ? $::env(SITE_COUNT_H) : 0}]
set site_count_v [expr {[info exists ::env(SITE_COUNT_V)] ? $::env(SITE_COUNT_V) : 0}]
set flip [expr {[info exists ::env(FLIP)] ? $::env(FLIP) : "s"}]
set ref_dir $::env(REF_DIR)
set floorplan_enc "${ref_dir}/enc/${design}_floorplan.enc"

if { ![file exists ${floorplan_enc}] } {
  puts "ERROR: Enc file $floorplan_enc does not exist"
  exit 1
}

## Load the helper functions
source ./gen_graph_invs.tcl

source $floorplan_enc
if { [file exists ${ref_dir}/place_pins.tcl] } {
    source ${ref_dir}/place_pins.tcl
    dbset [dbget top.terms.pStatus placed -p ].pStatus fixed
}

## Remove the powerplan from the floorplan
editDelete -net [dbget [dbget top.nets.isPwrOrGnd 1 -p -e ].name]

## Remove all placement and routing blockages
deletePlaceBlockage -all
deleteRouteBlk -all

## Set dont touch for instance and nets
dbset top.insts.dontTouch sizeOk
dbset top.nets.dontTouch true

## Write out IO and macros placement information
fp_io_macro_details

## Update the floorplan
expand_fp_by_site $site_count_h $site_count_v $flip

## Update the IO and macros placement information
if { [file exists macro_details.tcl] } {
  source macro_details.tcl
}

if { [file exists io_details.tcl] } {
  source io_details.tcl
}

## Place the design
setPlaceMode -place_global_align_macro true
setPlaceMode -place_global_place_io_pins true

## If macros are there then run refine_macro_place
if { [dbget top.insts.cell.subClass block -p -e] != "" } {
  dbset [dbget top.insts.cell.subClass block -p2 ].pStatus placed
  refine_macro_place
  dbset [dbget top.insts.cell.subClass block -p2 ].pStatus fixed
}

## Set dont touch for instance and nets
dbset top.insts.dontTouch sizeok

## Place design
place_opt_design

exec mkdir -p enc
saveDesign ./enc/${design}.enc

## Write down the graph information and features
exec mkdir -p blob_input
cd ./blob_input
write_blob_place_exp_info 1 0 0
cd ..

earlyGlobalRoute
set cong_inst_ptr_list [get_cong_inst_ptrs 0 1]

set python_script "/home/fetzfs_projects/PlacementCluster/sakundu/BlobPlacement/Clustering/gen_cluster.py"
set python_exe "/home/tool/anaconda/envs/cluster/bin/python"

${python_exe} $python_script ./blob_input $design 42 1 1.0 0 |& tee log/clustering.log

exit
