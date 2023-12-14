proc drc_box_rpt { file_name } {
  set fp [open $file_name w]
  foreach drc_ptr [dbget top.markers.type Geometry -p -e] {
    set bbox [concat {*}[dbget $drc_ptr.box]]
    puts $fp "$bbox"
  }
  close $fp
}

proc report_inst_terms { file_name } {
  set fp [open $file_name w]
  foreach instTermPtr [dbget top.insts.instTerms -u -e] {
    set ptx [dbget ${instTermPtr}.pt_x]
    set pty [dbget ${instTermPtr}.pt_y]
    puts $fp "$ptx $pty"
  }
  close $fp
}

proc report_inst_box { file_name } {
  set fp [open $file_name w]
  foreach instPtr [dbget top.insts -u -e] {
    set bbox [concat {*}[dbget ${instPtr}.box]]
    puts $fp "$bbox"
  }
  close $fp
}

# (100, 95, 90, 85, 80, 75, 70)

proc report_routing_resource_usage { file_name } {
  set route_screen_values [list 100 95 90 85 80 75 70]
  deleteRouteBlk -all
  foreach route_screen_value $route_screen_values {
    editDelete -net [dbget top.nets.name]
    createRouteBlk -box [dbget top.fplan.box] -layer all -partial $route_screen_value
    earlyGlobalRoute
    dumpCongestArea -all ${file_name}_${route_screen_value}.rpt
    deleteRouteBlk -all
  }
  editDelete -net [dbget top.nets.name]
}

proc write_features {  } {
  report_inst_terms "inst_terms.rpt"
  report_inst_box "inst_box.rpt"
  report_routing_resource_usage "routing_resource_usage"
}

proc write_labels { } {
  drc_box_rpt "drc_box.rpt"
}

proc extract_egr_net { net_ptr fp } {
  ## File Format:
  ## START NET <net_name>
  ## LAYERS
  ## <layer_name> <layer_box>
  ## VIAS
  ## <via_layer_name> <via_layer_pt>
  ## END NET <net_name>

  set net_name [dbget ${net_ptr}.name]
  puts $fp "START NET $net_name"
  puts $fp "LAYERS"
  foreach wire_ptr [dbget ${net_ptr}.wires] {
    set layer_name [dbget ${wire_ptr}.layer.name]
    set layer_box [concat {*}[dbget ${wire_ptr}.box]]
    puts $fp "$layer_name $layer_box"
  }

  puts $fp "VIAS"
  foreach via_ptr [dbget ${net_ptr}.vias] {
    set via_layer_name [dbget ${via_ptr}.via.cutLayer.name]
    set via_layer_pt [concat {*}[dbget ${via_ptr}.pt]]
    puts $fp "$via_layer_name $via_layer_pt"
  }
  puts $fp "END NET $net_name"
}

proc extract_egr_all_nets { file_name } {
  set fp [open $file_name w]
  foreach net_ptr [dbget top.nets.isPwrOrGnd 0 -p] {
    extract_egr_net $net_ptr $fp
  }
  close $fp
}
