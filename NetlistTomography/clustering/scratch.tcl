proc get_cong_inst_ptrs { {th -3} {write_cells 1} } {
  set inst_ptr_list {}
  get_db gcells -if {.vertical_remaining < $th || .horizontal_remaining < $th} -foreach {
    set box $obj(.rect)
    set inst_ptrs [dbQuery -overlap_only -area $box -objType inst]
    foreach inst_ptr $inst_ptrs {
      lappend inst_ptr_list $inst_ptr
    }
  }
  catch {set inst_ptr_list [lsort -unique $inst_ptr_list]}
  set inst_count [llength $inst_ptr_list]
  puts $inst_count

  ## Highlight these instances
  highlight $inst_ptr_list -color red

  if { $write_cells == 1 } {
    set design [dbget top.name]
    set fp [open "${design}_cong_cells.txt" w]
    puts $fp "Name"
    foreach inst_ptr $inst_ptr_list {
      set inst_typ [dbGet ${inst_ptr}.cell.subClass]
      if { $inst_typ != "block" } {
        set inst_name [dbGet ${inst_ptr}.name]
        puts $fp "$inst_name"
      }
    }
    close $fp
  }
  return $inst_ptr_list
}

proc add_cell_padding { inst_file inst_count is_left is_right } {
  ## inst_count should be in 0 to 100 and a float
  if { $inst_count < 0 || $inst_count > 100 } {
    puts "Invalid inst_count"
    return
  }
  
  ## One of is_left and is_right should be one
  if { $is_left == 0 && $is_right == 0 } {
    puts "Invalid is_left and is_right"
    return
  }
  setPlaceMode -place_detail_honor_inst_pad true
  ## Total inst count
  set total_inst [llength [dbGet top.insts]]
  set pad_inst_count [expr {int($inst_count * $total_inst / 100.0)}]
  set fp [open $inst_file r]
  set i 0
  while {[gets $fp line] != -1 && $i < $pad_inst_count} {
    set items [split $line ',']
    set inst_name [lindex $items 0]
    if { [dbGet top.insts.name ${inst_name} -e] != "" } {
      specifyInstPad $inst_name -left $is_left -right $is_right
    }
    incr i
  }
  close $fp
  reportInstPad -all > inst_pad.rpt
}

proc get_cell_in_hotspot_gcells { } {
  set inst_ptr_list {}
  set gcell_list {}
  foreach box [dbget [dbget top.markers.type Geometry -p ].box -u] {
    set llx [lindex $box 0]
    set lly [lindex $box 1]
    set urx [lindex $box 2]
    set ury [lindex $box 3]
    # get_db gcells  -if {.rect.ur.x < $llx || .rect.ll.x > $urx || \
    #       .rect.ur.y < $lly || .rect.ll.y > $ury} -invert -foreach {
    #   set bbox $obj(.rect)
    #   set inst_ptrs [dbQuery -overlap_only -area $bbox -objType inst]
    #   foreach inst_ptr $inst_ptrs {
    #     lappend inst_ptr_list $inst_ptr
    #   }
    # }
    set gcells [get_db gcells  -if {.rect.ur.x < $llx || .rect.ll.x > $urx || \
          .rect.ur.y < $lly || .rect.ll.y > $ury} -invert]
    foreach gcell $gcells {
      lappend gcell_list $gcell
    }
  }

  ## Uniqify the Gcell list
  catch {set gcell_list [lsort -unique $gcell_list]}
  set gcell_count [llength $gcell_list]
  puts "Number of Unique  Gcells $gcell_count"
  ## Create the inst list
  foreach gcell $gcell_list {
    set box [get_db $gcell .rect]
    set inst_ptrs [dbQuery -overlap_only -area $box -objType inst]
    foreach inst_ptr $inst_ptrs {
      lappend inst_ptr_list $inst_ptr
    }
  }

  catch {set inst_ptr_list [lsort -unique $inst_ptr_list]}
  set inst_count [llength $inst_ptr_list]
  puts $inst_count

  set design [dbget top.name]
  set fp [open "${design}_cong_cells.txt" w]
  puts $fp "Name"
  foreach inst_ptr $inst_ptr_list {
    set inst_typ [dbGet ${inst_ptr}.cell.subClass]
    if { $inst_typ != "block" } {
      set inst_name [dbGet ${inst_ptr}.name]
      puts $fp "$inst_name"
    }
  }
  close $fp
}

proc write_start_end_pairs { {fileName "" } {num_paths 500} {nworst 1} {path_group "reg2reg"}} {
  if { $fileName == "" } {
    set design [dbget top.name]
    set fileName "${design}_start_end_pairs.csv"
  }
  set fp [open $fileName w]
  puts $fp "Start,End,Slack"
  set paths [report_timing -max_paths $num_paths -nworst $nworst -path_group $path_group -collection]
  foreach_in_collection path $paths {
    set slack [get_property $path slack]
    set launch [get_cells -of_object [get_property $path launching_point]]
    set capture [get_cells -of_object [get_property $path capturing_point_name]]
    set launch_name [get_object_name $launch]
    set capture_name [get_object_name $capture]
    puts $fp "$launch_name,$capture_name,$slack"
  }
  close $fp
}

proc commonElements {list1 list2} {
  # Create a dictionary from the first list
  set dict1 {}
  foreach item $list1 {
    dict set dict1 $item 1
  }
  
  # Initialize the list for common elements
  set common {}
  
  # Check each element of the second list against the dictionary
  foreach item $list2 {
    if {[dict exists $dict1 $item]} {
      lappend common $item
    }
  }
  
  # Return the list of common elements
  return $common
}

proc create_soft_guide { {inst_count 25} } {
  foreach grp_ptr [dbget top.fplan.groups] {
    set inst_c [llength [dbget $grp_ptr.members]]
    if { $inst_c < $inst_count } {
      deleteInstGroup [dbget ${grp_ptr}.name]
      continue
    }
    createSoftGuide [dbget ${grp_ptr}.name]
  }
}

proc create_soft_guide_based_on_cluster { cluster_file {id 3} } {
  set fp [open $cluster_file r]
  while {[gets $fp line] != -1} {
    set items [split $line ',']
    set inst [lindex $items 0]
    set cluster_id [lindex $items $id]
    if { [dbget top.fplan.groups.name cluster_${cluster_id} -e] == "" } {
      createInstGroup cluster_${cluster_id} -softGuide
    }
    ## Ensure the inst exists
    if { [dbget top.insts.name $inst -e] == "" } {
      continue
    }
    addInstToInstGroup cluster_${cluster_id} $inst
  }
  close $fp

  ## Delete the empty clusters
  foreach cluster [dbget top.fplan.groups] {
    if { [llength [dbget $cluster.members -e]] == 0 } {
      deleteInstGroup [dbget ${cluster}.name]
    }
  }
}

proc get_path_insts_helper { nets } {
  set insts {}
  foreach_in_collection net $nets {
    set node_name [get_object_name [get_cells -of_objects \
              [get_property $net driver_pins] -quiet]]
    if { $node_name == "" } {
      continue
    }
    lappend insts $node_name
    set previou_net [get_object_name $net]
    set previous_node $node_name
  }
  return $insts
}

proc get_path_insts { start end } {
  set paths [report_timing -from $start -to $end -nworst 10 -collection]
  set sorted_paths [sort_collection $paths num_cell_arcs -descending]
  set path [index_collection $sorted_paths 0]
  set nets [get_property $path nets]
  set insts [get_path_insts_helper $nets]
  return $insts
}

proc create_fanin_soft_guide { start_end_file path_id } {
  ## Check that the path file exists
  if { [file exists $start_end_file] == 0 } {
    puts "Path file does not exist"
    return
  }
  set fp [open $start_end_file r]
  set id 0
  while {[gets $fp line] != -1} {
    set items [split $line ',']
    set start [lindex $items 0]
    set end [lindex $items 1]
    if { $id != $path_id } {
      incr id
      continue
    }
    set insts [all_fanin -to ${end}/D -only_cells]
    createInstGroup cluster_${id} -softGuide
    foreach_in_collection inst $insts {
      addInstToInstGroup cluster_${id} [get_object_name $inst]
    }
    break
  }
}

proc write_path_insts { path_file {path_inst_file "" } } {
  if { $path_inst_file == "" } {
    set design [dbget top.name]
    set path_inst_file "${design}_path_insts.csv"
  }
  ## Ensure path file exists
  if { [file exists $path_file] == 0 } {
    puts "Path file does not exist"
    return
  }
  
  set fp [open $path_file r]
  set inst_fp [open $path_inst_file w]
  puts $inst_fp "Inst,Id"
  set path_id 0
  ## Read the path file line by line and get the start and end points
  while {[gets $fp line] != -1} {
    set items [split $line ',']
    if { [dbget top.insts.name [lindex $items 0] -e] == "" || \
         [dbget top.insts.name [lindex $items 1] -e] == "" } {
      continue
    }
    set start [lindex $items 0]
    set end [lindex $items 1]
    set insts [get_path_insts $start $end]
    foreach inst $insts {
      if { $inst != "" } {
        puts $inst_fp "$inst,$path_id"
      }
    }
    incr path_id
  }
  close $fp
  close $inst_fp
}

proc write_edge_helper { nets edge_fp path_id capture_name} {
  set previous_net ""
  set previous_node ""
  set nodes {}
  foreach_in_collection net $nets {
    set node_name [get_object_name [get_cells -of_objects \
              [get_property $net driver_pins] -quiet]]
    if { $node_name == "" } {
      continue
    }
    if { $previous_net != "" } {
      puts $edge_fp "$previous_node,$node_name,1,$previous_net,$path_id"
    }
    set previous_net [get_object_name $net]
    set previous_node $node_name
    lappend nodes $node_name
  }
  if { $previous_net != "" } {
    puts $edge_fp "$previous_node,$capture_name,1,$previous_net,$path_id"
  }
  return $nodes
}

proc collection2list { collection } {
  set list {}
  foreach_in_collection item $collection {
    lappend list [get_object_name $item]
  }
  return $list
}

proc write_start_end_insts { file_name } {
  ## Read the file and get the start and end points
  set fp [open $file_name r]
  set start_points {}
  set end_points {}
  while {[gets $fp line] != -1} {
    set items [split $line ',']
    if { [dbget top.insts.name [lindex $items 0] -e] == "" || \
         [dbget top.insts.name [lindex $items 1] -e] == "" } {
      continue
    }
    lappend start_points [lindex $items 0]
    lappend end_points [lindex $items 1]
  }
  close $fp
  set design [dbget top.name]
  set output_file "${design}_start_end_insts.csv"
  set fp [open $output_file w]
  puts $fp "Inst,Id"
  set i 0
  foreach start_point $start_points {
    set end_point [lindex $end_points $i]
    set end_point_fan_ins [all_fanin -to ${end_point}/D -only_cells]
    set start_point_fan_outs [all_fanout -from ${start_point}/Q -only_cells]
    set end_point_list [collection2list $end_point_fan_ins]
    set start_point_list [collection2list $start_point_fan_outs]
    set common_list [commonElements $end_point_list $start_point_list]
    foreach inst $common_list {
      puts $fp "$inst,$i"
    }
    incr i
  }
  close $fp
}

proc report_start_end_slack { file_name } {
  ## Read the file and get the start and end points
  set fp [open $file_name r]
  set start_points {}
  set end_points {}
  set old_slack {}
  while {[gets $fp line] != -1} {
    set items [split $line ',']
    if { [dbget top.insts.name [lindex $items 0] -e] == "" || \
         [dbget top.insts.name [lindex $items 1] -e] == "" } {
      continue
    }
    lappend start_points [lindex $items 0]
    lappend end_points [lindex $items 1]
    lappend old_slack [lindex $items 2]
  }
  close $fp
  set design [dbget top.name]
  set output_file "${design}_start_end_slack.csv"
  set fp [open $output_file w]
  puts $fp "Start,End,Slack,OldSlack"
  set i 0
  foreach start_point $start_points {
    set end_point [lindex $end_points $i]
    set oldSlack [lindex $old_slack $i]
    set path [report_timing -from ${start_point} -to ${end_point} -nworst 1 -collection]
    set slack [get_property $path slack]
    puts $fp "$start_point,$end_point,$slack,$oldSlack"
    incr i
  }
  close $fp
}

proc get_reg2reg_timing_graph { {max_paths 200000} } {
  ## Update Parasitics and Timing Graph
  reset_path_group
  createBasicPathGroups
  timeDesign -preCTS
  
  ## Get All Reg2Reg Timing Paths
  set start_time [clock milliseconds]
  set paths [report_timing -begin_end_pair -path_group reg2reg -collection]
  set end_time [clock milliseconds]
  set runtime [expr {$end_time - $start_time}]
  set path_count [sizeof_collection $paths]
  puts "Runtime: $runtime milliseconds for $path_count paths"

  ## Write Path Details, Edge Details and Node details
  set design [dbget top.name]
  set path_fp [open "${design}_reg2reg_paths.csv" w]
  puts $path_fp "PathID,Start,End,Slack,Depth,Cumulative_Length"
  set edge_fp [open "${design}_reg2reg_edges.csv" w]
  set nodes {}
  puts $edge_fp "Source,Sink,Weight,Net,PathID"
  set path_id 0
  set sorted_paths [sort_collection $paths num_cell_arcs -descending]
  foreach_in_collection path $sorted_paths {
    set slack [get_property $path slack]
    set launch [get_cells -of_object [get_property $path launching_point]]
    set capture [get_cells -of_object [get_property $path capturing_point_name]]
    set depth [get_property $path num_cell_arcs]
    set cdist [get_property $path cumulative_manhattan_length]
    set launch_name [get_object_name $launch]
    set capture_name [get_object_name $capture]
    puts $path_fp "$path_id,$launch_name,$capture_name,$slack,$depth,$cdist"
    set nets [get_property $path nets]
    set path_nodes [write_edge_helper $nets $edge_fp $path_id $capture_name]
    eval lappend nodes $path_nodes
    incr path_id
    if { $path_id >= $max_paths } {
      break
    }
  }
  close $path_fp
  close $edge_fp

  ## Unique Nodes
  set nodes [lsort -unique $nodes]
  set node_fp [open "${design}_reg2reg_nodes.csv" w]
  puts $node_fp "Name,Type,Master,Height,Width"
  get_db insts $nodes -foreach {
    set inst_name $obj(.name)
    set inst_type $obj(.obj_type)
    set inst_master [lindex [split $obj(.lib_cells.name) "/"] end]
    set inst_height $obj(.bbox.dy)
    set inst_width $obj(.bbox.dx)
    puts $node_fp "$inst_name,$inst_type,$inst_master,$inst_height,$inst_width"
  }
  close $node_fp
}

proc report_cluster_details {} {
  ## Average Inst count max and minimum
  ## Number of clusters
  set ccount [llength [dbget top.fplan.groups]]
  set inst_count [llength [dbget top.fplan.groups.members]]
  set group_insts {}
  foreach grp_ptr [dbget top.fplan.groups] {
    set inst_count [llength [dbget $grp_ptr.members]]
    lappend group_insts $inst_count
  }

  ## Report Mean, STD, Max and Min of group_insts, median
  set mean [expr {double([tcl::mathop::+ {*}$group_insts]) / $ccount}]
  set std [expr {sqrt([expr {double([tcl::mathop::+ {*}[lmap x $group_insts {expr {($x - $mean) * ($x - $mean)}}]]) / $ccount}])}]
  set sorted_insts [lsort -integer -decreasing $group_insts]
  set max [lindex $sorted_insts 0]
  set min [lindex $sorted_insts end]
  set median [lindex $sorted_insts [expr {$ccount / 2}]]
  puts "Cluster Count: $ccount Mean: $mean, STD: $std, Max: $max, Min: $min Median: $median"
}

proc get_reg2reg_timing_paths { {max_paths 100000} {path_per_endpoint 20} \
  {stage "-preCTS"} } {
  ## Update Parasitics and Timing Graph
  reset_path_group
  createBasicPathGroups
  set cmd "timeDesign $stage"
  eval $cmd
  
  ## Get All Reg2Reg Timing Paths
  set start_time [clock milliseconds]
  set total_path [expr $max_paths * $path_per_endpoint]
  set paths [report_timing -max_paths $total_path -nworst $path_per_endpoint \
       -path_group reg2reg -collection]
  
  set sorted_paths [sort_collection $paths {capturing_point_name launching_point_name slack}]
  set unique_paths ""
  set previous_capture ""
  set previous_launch ""
  foreach_in_collection path $sorted_paths {
    set capturep [get_property $path capturing_point_name]
    set launchp [get_property $path launching_point_name]
    if { $capturep == $previous_capture && $launchp == $previous_launch } {
      continue
    }
    set previous_capture $capturep
    set previous_launch $launchp
    append_to_collection unique_paths $path
  }
  set sorted_paths [sort_collection $unique_paths slack]
  set end_time [clock milliseconds]
  set runtime [expr {$end_time - $start_time}]
  set path_count [sizeof_collection $sorted_paths]
  puts "Runtime: $runtime milliseconds for $path_count paths"

  ## Wriet Path Details Start, End, Slack
  set design [dbget top.name]
  set path_fp [open "${design}_reg2reg_paths.csv" w]
  if { $stage == "-preCTS" } {
    set path_fp [open "${design}_reg2reg_paths_preCTS.csv" w]
  } elseif { $stage == "-postRoute"} {
    set path_fp [open "${design}_reg2reg_paths_postRoute.csv" w]
  }

  puts $path_fp "Start,End,Slack"
  set i 0
  foreach_in_collection path $sorted_paths {
    set slack [get_property $path slack]
    set launch [get_cells -of_object [get_property $path launching_point]]
    set capture [get_cells -of_object [get_property $path capturing_point_name]]
    set launch_name [get_object_name $launch]
    set capture_name [get_object_name $capture]
    puts $path_fp "$launch_name,$capture_name,$slack"
    incr i
    if { $i >= $max_paths } {
      break
    }
  }
  close $path_fp
}

proc write_path_graphs { paths max_paths } {
  ## Write Path Details, Edge Details and Node details
  set design [dbget top.name]
  set path_fp [open "${design}_reg2reg_paths.csv" w]
  puts $path_fp "PathID,Start,End,Slack,Depth,Cumulative_Length"
  set edge_fp [open "${design}_reg2reg_edges.csv" w]
  set nodes {}
  puts $edge_fp "Source,Sink,Weight,Net,PathID"
  set path_id 0
  set sorted_paths [sort_collection $paths {num_cell_arcs cumulative_manhattan_length} -descending]
  foreach_in_collection path $sorted_paths {
    set slack [get_property $path slack]
    set launch [get_cells -of_object [get_property $path launching_point]]
    set capture [get_cells -of_object [get_property $path capturing_point_name]]
    set depth [get_property $path num_cell_arcs]
    set cdist [get_property $path cumulative_manhattan_length]
    set launch_name [get_object_name $launch]
    set capture_name [get_object_name $capture]
    puts $path_fp "$path_id,$launch_name,$capture_name,$slack,$depth,$cdist"
    set nets [get_property $path nets]
    set path_nodes [write_edge_helper $nets $edge_fp $path_id $capture_name]
    eval lappend nodes $path_nodes
    incr path_id
    if { $path_id >= $max_paths } {
      break
    }
  }
  close $path_fp
  close $edge_fp

  ## Unique Nodes
  set nodes [lsort -unique $nodes]
  set node_fp [open "${design}_nodes.csv" w]
  puts $node_fp "Name,Type,Master,Height,Width"
  get_db insts $nodes -foreach {
    set inst_name $obj(.name)
    set inst_type $obj(.obj_type)
    set inst_master [lindex [split $obj(.lib_cells.name) "/"] end]
    set inst_height $obj(.bbox.dy)
    set inst_width $obj(.bbox.dx)
    puts $node_fp "$inst_name,$inst_type,$inst_master,$inst_height,$inst_width"
  }
  close $node_fp
}

proc get_timing_graph_timing_report { {max_paths 10000} {nworst 20} {path_group "reg2reg"} } {
  set paths [report_timing -max_paths $max_paths -nworst $nworst -path_group $path_group -collection]
  set design [dbget top.name]
  write_path_graphs $paths [sizeof_collection $paths]
}

proc get_worst_endpoint_graph { {max_paths 10000} {nworst 20} } {
  set paths {}
  set worst_path [report_timing -max_paths 1 -nworst 1 -path_group reg2reg -collection]
  set end_point [get_property $worst_path capturing_point_name]
  set all_start_points [all_fanin -to $end_point -only_cells -startpoints_only]
  foreach_in_collection start_point $all_start_points {
    set path [report_timing -max_paths 1 -nworst 1 -from $start_point -to $end_point -collection]
    append_to_collection paths $path
  }
  set design [dbget top.name]
  write_path_graphs $paths [sizeof_collection $paths]
}

proc get_left_right_spacing { inst_ptr } {
  set box [dbget ${inst_ptr}.box]
  set pty [dbget ${inst_ptr}.pt_y]
  set ptx [dbget ${inst_ptr}.pt_x]
  set neighbor_inst_ptx [dbget [dbget [dbQuery -areas ${box} -abut_only \
               -objType inst].pt_y $pty -e -p ].pt_x ]
  if { $neighbor_inst_ptx == "" } {
    return {1 1}
  }

  set is_lpt_x 1
  set is_rpt_x 1
  foreach n_ptx $neighbor_inst_ptx {
    if { $n_ptx < $ptx } {
      set is_lpt_x 0
    }
    if { $n_ptx > $ptx } {
      set is_rpt_x 0
    }
  }

  return [list $is_lpt_x $is_rpt_x]
}

## Write left right spacing for all the instances
proc get_inst_spacing { file_name } {
  set fp [open $file_name w]
  puts $fp "Name,Left,Right"
  foreach inst_ptr [dbget top.insts] {
    set inst_name [dbGet ${inst_ptr}.name]
    set spacing [get_left_right_spacing $inst_ptr]
    set lspacing [lindex $spacing 0]
    set rspacing [lindex $spacing 1]
    puts $fp "$inst_name,$lspacing,$rspacing"
  }
  close $fp
}

proc expand_fp_by_site { site_count_h site_count_v row_flip} {
  set core_width [dbget top.fplan.coreBox_sizex]
  set core_height [dbget top.fplan.coreBox_sizey]
  set site_width [lindex [dbget head.sites.size_x] 0]
  set site_height [lindex [dbget head.sites.size_y] 0]
  set core2bot [dbget top.fplan.core2bot]
  set core2top [dbget top.fplan.core2top]
  set core2left [dbget top.fplan.core2left]
  set core2right [dbget top.fplan.core2right]

  set new_core_width [expr {$core_width + $site_count_h * $site_width}]
  set new_core_height [expr {$core_height + $site_count_v * $site_height}]
  floorPlan -s $new_core_width $new_core_height $core2left $core2bot \
               $core2right $core2top -adjustToSite -flip $row_flip
}

proc fp_io_macro_details { } {
  if { [dbget top.insts.cell.subClass block -e -u] == "block" } {
    deselectAll
    select_obj [dbget top.insts.cell.subClass block -p2]
    dbset selected.pStatus placed
    writeFPlanScript -fileName macro_details.tcl -selected
    deselectAll
  }

  if { [dbget top.terms.pStatus -e -u] == "fixed" } {
    select_obj [dbget top.terms]
    writeFPlanScript -fileName io_details.tcl -selected
    deselectAll
  }
}

proc read_cell_padding_list {padding_file perc} {
  set fp [open $padding_file r]
  set inst_count [llength [dbGet top.insts]]
  set target_count [expr {int($perc * $inst_count)}]
  set padding_list {}
  set i 0
  while {[gets $fp line] != -1 && $i < $target_count} {
    set items [split $line ',']
    lappend padding_list [lindex $items 0]
    incr i
  }
  close $fp
  return $padding_list
}

proc save_padded_cells {padding_file} {
  set design [dbget top.name]
  foreach perc {0.005 0.01 0.015 0.02} {
    set inst_list [read_cell_padding_list $padding_file $perc]
    dehighlight -all
    highlight $inst_list -color red
    gui_dump_picture ${design}_${perc}_pad.jpeg -format jpeg
  }
}


proc get_spacing { inst_ptr1 inst_ptr2 site_width } {
  set lly1 [dbget ${inst_ptr1}.box_lly]
  set lly2 [dbget ${inst_ptr2}.box_lly]
  if { $lly1 != $lly2 } {
    return 0
  }
  set urx1 [dbget ${inst_ptr1}.box_urx]
  set llx2 [dbget ${inst_ptr2}.box_llx]
  set spacing [expr {($llx2 - $urx1) / $site_width}]
  return $spacing
}

proc get_cell_spacing { {file_name ""} } {
  set design [dbget top.name]
  if { $file_name == "" } {
    set file_name "${design}_cell_spacing.csv"
  }
  set fp [open $file_name w]
  puts $fp "Name,Master,Orientation,LeftSpacing,RightSpacing"
  set sorted_insts [get_object_name [sort_collection [get_cells [dbget [dbget top.insts.cell.subClass block -v -p2 ].name ]] {y_coordinate_min x_coordinate_min}]]
  set inst_ptrs {}
  
  foreach inst_name $sorted_insts {
    lappend inst_ptrs [dbget top.insts.name $inst_name -p]
  }

  set site_width [lindex [lsort [dbget head.sites.size_x]] 0]
  set inst_count [expr [llength $sorted_insts] - 1]
  set i 0
  while { $i < [llength $sorted_insts] } {
    set master [dbget [lindex $inst_ptrs $i].cell.name]
    set inst_name [lindex $sorted_insts $i]
    set orientation [dbget [lindex $inst_ptrs $i].orient]
    if { $i > 0 } {
    set left_spacing [get_spacing [lindex $inst_ptrs [expr $i - 1]] [lindex $inst_ptrs $i] $site_width]
    } else {
      set left_spacing 0
    }

    if { $i < $inst_count } {
      set right_spacing [get_spacing [lindex $inst_ptrs $i] [lindex $inst_ptrs [expr $i + 1]] $site_width]
    } else {
      set right_spacing 0
    }

    puts $fp "$inst_name,$master,$orientation,$left_spacing,$right_spacing"
    incr i
  }

  close $fp
}

proc highligh_critical_insts { file_name } {
  if { [file exists $file_name] == 0 } {
    puts "File does not exist"
    return
  }
  set fp [open $file_name r]
  set i 0
  while {[gets $fp line] != -1} {
    set items [split $line ',']
    set inst_name [lindex $items 0]
    if { [dbget top.insts.name $inst_name -e] == "" } {
      continue
    }
    set inst_ptr [dbget top.insts.name $inst_name -p]
    highlight $inst_ptr -color red
    incr i
    if { [expr {$i % 100}] == 0 } {
      puts "Highlighted $i instances"
    }
  }
}

proc highligh_critical_insts_tg { file_name } {
  if { [file exists $file_name] == 0 } {
    puts "File does not exist"
    return
  }
  set fp [open $file_name r]
  set i 0
  while {[gets $fp line] != -1} {
    set items [split $line ',']
    set slack [lindex $items 2]
    if { $slack > 0 || $slack == "INFINITY" } {
      continue
    }
    set inst_name [lindex $items 0]
    if { [dbget top.insts.name $inst_name -e] == "" } {
      continue
    }
    set inst_ptr [dbget top.insts.name $inst_name -p]
    highlight $inst_ptr -color red
    incr i
    if { [expr {$i % 1000}] == 0 } {
      puts "Highlighted $i instances"
    }
  }
}

# set inst_ptr_list [get_inst_ptrs -1]


# foreach inst_ptr $inst_ptr_list {
#   set inst_name [dbGet ${inst_ptr}.name]
#   puts "$inst_name"
# }
