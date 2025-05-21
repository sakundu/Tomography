# This script was written and developed by ABKGroup students at UCSD. 
# However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help 
# promote and foster the next generation of innovators.

### Write out the node details in the node file ###
proc print_node_header { fp } {
  puts $fp "Name,Type,Master,Height,Width"
}

proc print_node { fp nPtr } {
  set name [concat {*}[dbget ${nPtr}.name]]
  set type [dbget ${nPtr}.objType]
  set master "NA"
  set height "0.0"
  set width "0.0"
  if { $type == "inst" } {
    set master [dbget ${nPtr}.cell.name]
    set height [dbget ${nPtr}.cell.size_y]
    set width [dbget ${nPtr}.cell.size_x]
  }
  puts $fp "$name,$type,$master,$height,$width"
}

proc print_edge_header { fp } {
  puts $fp "Source,Sink,Weight,Net"
}

proc get_net_fanout { nPtr } {
  set instTerm_count [llength [dbget ${nPtr}.instTerms.isInput 1 -e]]
  set bTerm_count [llength [dbget ${nPtr}.terms.direction output -e]]
  return [expr $instTerm_count + $bTerm_count]
}

proc get_net_source { nPtr {isPin 0} } {
  set term_source [dbget [dbget ${nPtr}.terms.direction input -p ].name -e]
  if { $isPin } {
    set inst_source [dbget [dbget ${nPtr}.instTerms.isOutput 1 -p ].name -e]
  } else {
    set inst_source [dbget [dbget ${nPtr}.instTerms.isOutput 1 -p \
        ].inst.name -e]
  }
  
  if { $term_source != "" } {
    return [concat {*}$term_source]
  } elseif { $inst_source != "" } {
    return [concat {*}$inst_source]
  } else {
    set net_name [dbget ${nPtr}.name]
    puts "Error: Check source of net:$net_name"
  }
  return ""
}

proc get_net_sinks { nPtr {isPin 0} } {
  set sinks {}
  
  ## Add term sinks ##
  foreach outputTermPtr [dbget ${nPtr}.terms.direction output -p -e] {
    set sink_name [dbget ${outputTermPtr}.name]
    lappend sinks [concat {*}$sink_name]
  }

  ## Add inst sinks ##
  foreach inputIntTermPtr [dbget ${nPtr}.instTerms.isInput 1 -p -e] {
    if { $isPin } {
      set sink_name [dbget ${inputIntTermPtr}.name]
    } else {
      set sink_name [dbget ${inputIntTermPtr}.inst.name]
    }
    lappend sinks [concat {*}$sink_name]
  }
  return [lsort -unique $sinks]
}

proc print_edge { fp nPtr {fanout_threshold 50}} {
  set net_fanout [get_net_fanout $nPtr]
  set net_name [concat {*}[dbget ${nPtr}.name]]
  set source_name [get_net_source $nPtr]
  
  # Ignore power nets or high fanout nets #
  if { [dbget ${nPtr}.isPwrOrGnd] || $net_fanout > $fanout_threshold || 
        $net_fanout == 0 || $source_name == "" } {
    return
  }

  set edge_weight [expr 1.0/$net_fanout]
  set sinks [get_net_sinks $nPtr]
  if { [llength $sinks] != $net_fanout } {
    puts "Error: Fanout list of $net_name is not unique."
  }

  foreach sink $sinks {
    puts $fp "$source_name,$sink,$edge_weight,$net_name"
  }
}

proc write_graph { {file_name ""} } {
  # This function writes out the graph in the form of nodes and edges in 
  # csv format. The CSV format for the node file is as follows:
  # Name, Type, Master
  # The CSV format for the edge file is as follows:
  # Source, Sink, Weight, Net
  # The output files are <file_name>_nodes.csv and <file_name>_edges.csv
  # Usage: First source the file and the use the below command
  # write_graph <file_name>
  # If file_name is not specified, then the name of the top cell is used.
  
  if {$file_name == ""} {
    set file_name [dbget top.name]
  }

  set node_file "${file_name}_nodes.csv"
  set edge_file "${file_name}_edges.csv"
  set node_fp [open $node_file w]
  set edge_fp [open $edge_file w]

  print_node_header $node_fp
  print_edge_header $edge_fp

  # Write out the instance details as nodes #
  foreach nPtr [dbget top.insts ] {
    print_node $node_fp $nPtr
  }

  # Write out the terminal details as nodes #
  foreach nPtr [dbget top.terms ] {
    print_node $node_fp $nPtr
  }

  # Write out the net details as edges #
  foreach nPtr [dbget top.nets ] {
    print_edge $edge_fp $nPtr
  }

  close $node_fp
  close $edge_fp
}

proc print_hyperedge { fp nPtr {isPin 0}} {
  set net_name [concat {*}[dbget ${nPtr}.name]]
  set source_name [get_net_source $nPtr $isPin]
  set sink_names [get_net_sinks $nPtr $isPin]
  set sinks [join $sink_names " "]
  puts $fp "$net_name $source_name $sinks"
}

proc write_hypergraph { {file_name ""} {isPin 0}} {
  if {$file_name == ""} {
    set file_name [dbget top.name]
  }

  set hgr_file "${file_name}.hgr"
  set hgr_fp [open $hgr_file w]
  foreach nPtr [dbget top.nets] {
    if { [dbget ${nPtr}.isPwrOrGnd] || [get_net_fanout $nPtr] == 0 } {
      continue
    }
    print_hyperedge $hgr_fp $nPtr $isPin
  }
  close $hgr_fp
}

proc write_macro_location { {file_name ""} } {
  if {$file_name == ""} {
    set file_name [dbget top.name]
  }
  set db_unit [dbget head.dbUnits]
  set macro_file "${file_name}_macro.csv"
  set macro_fp [open $macro_file w]
  puts $macro_fp "Name,llx,lly"
  foreach macro_ptr [dbget top.insts.cell.subClass block -p2] {
    set macro_name [concat {*}[dbget ${macro_ptr}.name]]
    set macro_x [expr [dbget ${macro_ptr}.pt_x] * $db_unit]
    set macro_y [expr [dbget ${macro_ptr}.pt_y] * $db_unit]
    # Convert macro_x and macro_y to integer #
    set macro_x [expr int($macro_x)]
    set macro_y [expr int($macro_y)]
    puts $macro_fp "$macro_name,$macro_x,$macro_y"
  }
  close $macro_fp
}

proc write_slack_info_multi_clk { file_name } {
  set fp [open $file_name w]
  puts $fp "Source,Slack,ClockPeriod"
  foreach netPtr [dbget top.nets {.isClock == 0 && .isPwrOrGnd == 0}] {
    set net_source [get_net_source $netPtr]
    set is_inst [dbget [dbget ${netPtr}.instTerms.isOutput 1 -p -e ].name -e]
    
    if { $net_source == "" } {
      # set net_name [dbget ${netPtr}.name]
      # puts "Error: Check source of net: ${net_name}"
      continue
    }

    ## We do not care about IO ports ##
    if { $is_inst == "" } {
      puts $fp "$net_source,0.0,1.0"
      continue
    }

    set slack [get_property [get_pins $is_inst] slack_max]
    set clock_ptr [get_property [get_pins $is_inst] arrival_clocks]
    set clk_period [get_property $clock_ptr period]
    puts $fp "$net_source,$slack,$clk_period"
  }
  close $fp
}

proc write_slack_pt_info { { file_name ""} } {
  ## If file_name is not specified set it to the top cell name ##
  if {$file_name == ""} {
    set design [dbget top.name]
    set file_name "${design}_slack_pt.csv"
  }

  ## First get all the output terms ##
  set terms_ptr [dbget top.insts.instTerms.isoutput 1 -p]
  set insts_name [dbget ${terms_ptr}.inst.name]
  set cells_name [dbget ${terms_ptr}.inst.cell.name]
  set slacks [get_property [get_pins [dbget ${terms_ptr}.name]] slack_max]
  set i 0
  set fp [open $file_name w]
  puts $fp "Instance,Cell,Slack,ClockPeriod,PT_X,PT_Y"
  foreach term_ptr $terms_ptr {
    set term_name [dbget ${term_ptr}.name]
    set inst_name [lindex $insts_name $i]
    set cell_name [lindex $cells_name $i]
    set slack [lindex $slacks $i]
    set clks [get_property [get_pins $term_name] arrival_clocks]
    set clk_period [lindex [get_property $clks period] 0]
    set pt_x [dbget ${term_ptr}.pt_x]
    set pt_y [dbget ${term_ptr}.pt_y]
    puts $fp "$inst_name,$cell_name,$slack,$clk_period,$pt_x,$pt_y"
    incr i
  }
  close $fp
}

proc get_net_source_map {} {
  # This function returns a dictionary mapping net name to net source
  set net_source_map {}
  foreach netPtr [dbget top.nets {.isClock == 0 && .isPwrOrGnd == 0}] {
    set net_source [get_net_source $netPtr]
    if {$net_source == ""} {
      continue
    }    
    ## If net source is term skip it ##
    if { [dbget top.terms.name $net_source -e] != "" } {
      continue
    }
    set net_name [concat {*}[dbget ${netPtr}.name]]
    dict set net_source_map $net_name $net_source
  }
  return $net_source_map
}

proc write_edge_slack_length_info { {file_name "" } } {
  ## If file_name is not specified set it to the top cell name ##
  if {$file_name == ""} {
    set design [dbget top.name]
    set file_name "${design}_edge_slack_length.csv"
  }

  ## First get all the input terms ##
  set terms_ptr [dbget top.insts.instTerms.isinput 1 -p]
  set terms_name [dbget ${terms_ptr}.name]
  set slacks [get_property [get_pins $terms_name] slack_max]
  set net_source_map [get_net_source_map]
  set i 0
  set fp [open $file_name w]
  puts $fp "Source,Sink,Slack,ClockPeriod"
  
  foreach term_ptr $terms_ptr {
    set term_name [dbget ${term_ptr}.name]
    set net_name [dbget ${term_ptr}.net.name]
    set net_sink [concat {*}[dbget ${term_ptr}.inst.name]]
    
    ## If term net then skip ##
    if { ![dict exists $net_source_map $net_name] } {
      incr i
      continue
    }

    set net_source [dict get $net_source_map $net_name]
    set slack [lindex $slacks $i]
    set clks [get_property [get_pins $term_name] arrival_clocks]
    set clk_period [lindex [get_property $clks period] 0]
    puts $fp "$net_source,$net_sink,$slack,$clk_period"
    incr i
  }
  close $fp
}

proc write_slack_info { file_name } {
  ## First fix DRVs
  optDesign -preCTS -drv

  ## Now write out the slack information for each net source
  ## Slack weight is (1 - slack / clock period)^2
  set fp [open $file_name w]
  puts $fp "Source,Slack,Weight"
  foreach netPtr [dbget top.nets {.isClock  == 0 && .isPwrOrGnd == 0}] {
    set net_source [get_net_source $netPtr]
    if { $net_source == "" } {
      set net_name [dbget ${netPtr}.name]
      puts "Error: Check source of net: ${net_name}"
      continue
    }
    set path [report_timing -through $net_source -collection]
    set slack [get_property $path slack]
    set clk_period [get_property $path phase_shift]
    if { $clk_period == 0.0 || $clk_period == "" } {
      set slack_weight 0.0
    } else {
      set slack_weight [expr (1.0 - $slack/$clk_period)**2]
    }
    puts $fp "$net_source,$slack,$slack_weight"
  }
  close $fp
  deleteBufferTree
}

proc highlight_only_clusters { file_name } {
  # This function highlights the clusters in the design.
  # The input file is the cluster file which has the following format:
  # cluster_name, color
  
  # First dehighlight everything #
  dehighlight -all

  if { [file exists $file_name] == 0 } {
    puts "Error: $file_name does not exist."
    return
  }
  set fp [open $file_name r]
  while { [gets $fp line] >= 0 } {
    set cluster_name [lindex $line 0]
    set color [lindex $line 1]
    highlight $cluster_name -color $color
  }
}

proc highlight_only_avail_clusters { file_name } {
  # This function highlights the clusters in the design.
  # The input file is the cluster file which has the following format:
  # cluster_name, color
  
  # First dehighlight everything #
  dehighlight -all

  if { [file exists $file_name] == 0 } {
    puts "Error: $file_name does not exist."
    return
  }
  set fp [open $file_name r]
  while { [gets $fp line] >= 0 } {
    set cluster_name [lindex $line 0]
    set color [lindex $line 1]
    if { [dbget top.fplan.groups.name $cluster_name -e ] != "" } {
      highlight $cluster_name -color $color
    }
  }
}

proc highlight_clusters { file_name {highlight_io 0} {sortedCluster 0}} {
  # This function highlights the clusters in the design. 
  # The input file is the cluster file which has the following format:
  # Instance / Terminal Name, Cluster ID, Color
  # This function also creates groups for each cluster.
  # Usage: First source the file and the use the below command
  # highlight_clusters <cluster_file> <highlight_io>
  # If highlight_io is set to 1, then the IOs are also highlighted.
  # If highlight_io is set to 0, then only the instances are highlighted.

  # First remove all the existing groups #
  foreach group [dbget top.fplan.groups.name cluster* -e ] {
    deleteInstGroup $group
  }

  # Dehighlight all the instances #
  dehighlight -all

  set fp [open $file_name r]
  set previous_cluster_id -1
  while { [gets $fp line] >= 0 } {
    set inst_name [lindex $line 0]
    set cluster_id [lindex $line 1]
    set color [lindex $line 2]

    if { $sortedCluster == 1 && $cluster_id != $previous_cluster_id } {
      createInstGroup cluster_${cluster_id}
    } elseif { [dbget top.fplan.groups.name cluster_${cluster_id} -e] == "" } {
      createInstGroup cluster_${cluster_id}
    }

    if { [dbget top.insts.name $inst_name -e] != "" } {
      addInstToInstGroup cluster_${cluster_id} $inst_name
      highlight $inst_name -color $color
    }
    
    set previous_cluster_id $cluster_id

    if { $highlight_io == 0 } {
      continue
    }

    highlight $inst_name -color $color
  }

  close $fp
}

proc write_placement_info_helper { fp nPtr } {
  # This is a helper function for the write_placement_info function.
  set name [concat {*}[dbget ${nPtr}.name]]
  set type [dbget ${nPtr}.objType]
  set master "NA"
  if { $type == "inst" } {
    set master [dbget ${nPtr}.cell.name]
  }
  set x [dbget ${nPtr}.pt_x]
  set y [dbget ${nPtr}.pt_y]
  puts $fp "$name,$type,$master,$x,$y"
}

proc write_placement_info { {file_name ""} } {
  # This function writes out the placement information in csv format. 
  # The csv format is as follows:
  # Instance Name, Inst/Term, Cell Master for Instances, X, Y
  # Usage: First source this script then use the below command
  # write_placement_info <file_name>
  # If file_name is not specified, then the name of the top cell is used.

  if {$file_name == ""} {
    set file_name [dbget top.name]
    set file_name "${file_name}_placement.csv"
  }
  
  set fp [open $file_name w]
  puts $fp "Name,Type,Master,X,Y"
  
  foreach instPtr [dbget top.insts ] {
    write_placement_info_helper $fp $instPtr
  }

  foreach termPtr [dbget top.terms] {
    write_placement_info_helper $fp $termPtr
  }

  close $fp
}

proc write_blob_place_exp_info { {write_place 1} {write_slack 0} {buf_remove 1} } {
  # {db_dir "./blo_placement"}
  # exec mkdir $db_dir -p
  # exec cd $db_dir

  # Write out the placement def file #
  set design [dbget top.name]
  if { $buf_remove == 1 } {
    deleteBufferTree
  }

  saveNetlist ${design}_buf_removed.v
  defOut -netlist ${design}_placed.def

  # Generate the graph file
  write_graph

  # Generate the hypergraph file
  write_hypergraph

  # Write out the placement file
  if { $write_place == 1 } {
    write_placement_info
  }

  # exec cd -
  # Write Slack infor
  if { $write_slack == 1 } {
    write_slack_info ${design}_slack.csv
  }

  # If the design contains macro then write out the macros
  if { [llength [dbget top.insts.cell.subClass block -p -e]] > 0 } {
    write_macro_location
  }
}

proc extract_clustered_netlist_details { {file_name ""} } {
  # This function extracts the clustered netlist details
  # Writes out three files
  
  #  1. cluster files: <file_name>_cluster.csv
  #                     cluster_name, width, height
  
  #  2. pin file: <file_name>_pin.csv
  #                pin_name, location

  #  3. net file: <file_name>_net.csv
  #                net_name, source, sink1, sink2, ...

  #  4. net weight file: <file_name>_net_weight.csv
  #                       net_name, weight

  if {$file_name == ""} {
    set file_name [dbget top.name]
  }
  
  ## First write out the cluster file ##
  set cluster_file "${file_name}_cluster.csv"
  set cluster_fp [open $cluster_file w]
  puts $cluster_fp "Name,Width,Height,isFixed"
  foreach inst_ptr [dbget top.insts] {
    set inst_name [concat {*}[dbget ${inst_ptr}.name]]
    set width [dbget ${inst_ptr}.box_sizex]
    set height [dbget ${inst_ptr}.box_sizey]
    set pstat [dbget ${inst_ptr}.pStatus]
    set isFixed 0
    if { $pstat == "fixed" } {
      set isFixed 1
    }
    puts $cluster_fp "$inst_name,$width,$height,$isFixed"
  }
  close $cluster_fp

  ## Next write out the pin file ##
  set pin_file "${file_name}_pin.csv"
  set pin_fp [open $pin_file w]
  puts $pin_fp "Name,X,Y"

  foreach term_ptr [dbget top.terms] {
    set term_name [concat {*}[dbget ${term_ptr}.name]]
    set x [dbget ${term_ptr}.pt_x]
    set y [dbget ${term_ptr}.pt_y]
    puts $pin_fp "$term_name,$x,$y"
  }

  close $pin_fp

  ## Next write out the net file ##
  set net_file "${file_name}_net.csv"
  set net_fp [open $net_file w]
  foreach net_ptr [dbget top.nets.isPwrOrGnd 0 -p] {
    set net_name [concat {*}[dbget ${net_ptr}.name]]
    set source [get_net_source $net_ptr]
    set sinks [get_net_sinks $net_ptr]
    set sinks [join $sinks ","]
    if { $source == "" && $sinks == "" } {
      continue
    } elseif { $source == "" } {
      puts $net_fp "$net_name,$sinks"
    } elseif { $sinks == "" } {
      puts $net_fp "$net_name,$source"
    } else {
      puts $net_fp "$net_name,$source,$sinks"
    }
  }
  close $net_fp
}

proc gen_ariane_floorplan_def { util halo } {
  set design [dbget top.name]
  floorplan -r 1 $util 2 2 2 2
  place_pins
  place_ariane_macro_config1 $halo
  ## Convert $util to integer ##
  set util [expr int($util*100)]
  dbset [dbget top.insts.cell.subClass block -p2 ].pstatus fixed
  defOut ${design}_floorplan_${util}.def
}

proc add_tie_hi_tie_lo_to_netlist {} {
  set design [dbget top.name]
  place_pins

  # remove bufferes #
  deleteBufferTree

  if { [dbget top.nets.dontTouch true -v -p -e ] != "" } {
    dbset [dbget top.nets.dontTouch true -v -p -e ].dontTouch true
  }

  if { [dbget top.insts.dontTouch true -v -p -e ] != "" } {
    dbset [dbget top.insts.dontTouch true -v -p -e ].dontTouch true
  }
  
  place_design -concurrent_macros
  refinePlace
  setTieHiLoMode -cell {LOGIC1_X1 LOGIC0_X1}
  addTieHiLo
  saveNetlist ${design}.v
}

proc write_or_input { run_dir design } {
    source ${run_dir}/enc/${design}_placed.enc
    deleteBufferTree
    setTieHiLoMode -cell {LOGIC1_X1 LOGIC0_X1}
    addTieHiLo
    mkdir -p ${run_dir}/or_input
    saveNetlist ${run_dir}/or_input/${design}.v
    write_sdc ${run_dir}/or_input/${design}.sdc
}

proc add_tie_hi_tie_lo_to_netlist_mempool_cluster {} {
  set design [dbget top.name]
  source /home/fetzfs_projects/MacroPlacement/flow_scripts_run/MacroPlacement/Flows/util/place_mempool_cluster_macros.tcl
  mempool_cluster_macro_placement
  defIn  ../mem_pool_cluster_manual_run2/mempool_cluster_floorplan.def
  # remove bufferes #
  deleteBufferTree

  if { [dbget top.nets.dontTouch true -v -p -e ] != "" } {
    dbset [dbget top.nets.dontTouch true -v -p -e ].dontTouch true
  }

  if { [dbget top.insts.dontTouch true -v -p -e ] != "" } {
    dbset [dbget top.insts.dontTouch true -v -p -e ].dontTouch true
  }
  
  # setPlaceMode -place_global_place_io_pins true
  place_design
  # refinePlace
  setTieHiLoMode -cell {LOGIC1_X1 LOGIC0_X1}
  addTieHiLo
  saveNetlist ${design}.v
}

proc gen_bp_floorplan_def { util halo } {
  set design [dbget top.name]
  floorplan -r 1 $util 2 2 2 2
  place_pins 3
  place_macros_bp $halo
  ## Convert $util to integer ##
  set util [expr int($util*100)]
  dbset [dbget top.insts.cell.subClass block -p2 ].pstatus fixed
  defOut ${design}_floorplan_${util}.def
}

proc report_rent_con { cluster_ptr } {
  set inst_count [llength [ dbget ${cluster_ptr}.members ]]
  set external_nets 0
  foreach net_ptr [dbget ${cluster_ptr}.members.instTerms.net.isPwrOrGnd 0 -p ] {
    set cluster_count [llength [dbget ${net_ptr}.instTerms.inst.group.name -u]]
    if { $cluster_count > 1 } {
      incr external_nets
    }
  }
  set term_ptrs [dbget ${cluster_ptr}.members.instTerms.net.isPwrOrGnd 0 -p2]
  set pin_count [llength  $term_ptrs]
  set rent [expr log(${external_nets}*1.0/${pin_count})*1.0/log(${inst_count}*1.0)]
  return [expr $rent + 1]
}

proc report_rent_con_area { cluster_ptr } {
  set inst_count [llength [ dbget ${cluster_ptr}.members ]]
  set inst_area [expr [join [dbget ${cluster_ptr}.members.box_area] +]]
  set external_nets 0
  foreach net_ptr [dbget ${cluster_ptr}.members.instTerms.net.isPwrOrGnd 0 -p ] {
    set cluster_count [llength [dbget ${net_ptr}.instTerms.inst.group.name -u]]
    if { $cluster_count > 1 } {
      incr external_nets
    }
  }
  set term_ptrs [dbget ${cluster_ptr}.members.instTerms.net.isPwrOrGnd 0 -p2]
  set avg_pin_count [expr [llength  $term_ptrs]*1.0/$inst_count]
  set rent [expr log(${external_nets}*1.0/${avg_pin_count})*1.0/log(${inst_area}*1.0)]
  set rent_gf12 [expr log(${external_nets}*1.0/${avg_pin_count})*1.0/log(${inst_area}*78.0)]
  set rent_ng45 [expr log(${external_nets}*1.0/${avg_pin_count})*1.0/log(${inst_area}*27.0)]
  return [list $rent $rent_gf12 $rent_ng45]
}

proc count_below_threshold { rent_params threshold } {
  set count 0
  foreach rent $rent_params {
    if { $rent < $threshold } {
      incr count
    }
  }
  return $count
}

proc create_soft_guide { {top 1} } {
  set cluster_ptrs [dbget top.fplan.groups]
  set rent_params {}
  foreach cluster_ptr $cluster_ptrs {
    # puts [dbget ${cluster_ptr}.name]
    if { [llength [dbget ${cluster_ptr}.members]] <= 1 } {
      continue
    }
    set rent [report_rent_con $cluster_ptr]
    if { $rent <= 0.0 } {
      continue
    }
    lappend rent_params $rent
  }

  ## Compute mean and std
  set mean [expr [tcl::mathop::+ {*}$rent_params]*1.0 / [llength $rent_params]]
  set variance 0
  foreach rent $rent_params {
    set variance [expr $variance + ($rent - $mean)*($rent - $mean)]
  }
  set std [expr sqrt(${variance}*1.0/[llength $rent_params])]
  ## Return the cluster with rent below mean - top*std
  set threshold [expr $mean - $top*$std]
  ## Min value in the list
  set min_rent_value [lindex $rent_params 0]
  set max_rent_value [lindex $rent_params 0]

  foreach rent $rent_params {
    if { $rent < $min_rent_value } {
      set min_rent_value $rent
    }
    if { $rent > $max_rent_value } {
      set max_rent_value $rent
    }
  }

  ## If the number of element in rent_params less than threshold is less than 10% of total element in rent_params then reduce top by 0.25 and recompute
  while { [count_below_threshold $rent_params $threshold] < [expr [llength $rent_params]*0.1] } {
    set top [expr $top - 0.25]
    set threshold [expr $mean - $top*$std]
  }

  echo "Threshold: $threshold Mean: $mean Std: $std Min Rent Params: ${min_rent_value} Max Rent Params: ${max_rent_value}" > clustering_details

  set i 0
  set j 0
  set cluster_list {}
  while { $i < [llength $rent_params] } {
    if { [lindex $rent_params $i] < $threshold } {
      lappend cluster_list [lindex $cluster_ptrs $i]
      createSoftGuide [dbget [lindex $cluster_ptrs $i].name]
      incr j
    } else {
      deleteInstGroup [dbget [lindex $cluster_ptrs $i].name]
    }
    incr i
  }
  echo "Total Cluster count: $i Selected Cluster count: $j" >> clustering_details
}

proc add_cell_pad { cluster_ptr {top 0} {io_count 4} } {
  set inst_ptrs [dbget ${cluster_ptr}.members]
  set pin_density {}
  foreach inst_ptr $inst_ptrs {
    set area [dbget ${inst_ptr}.area]
    set pin_count [llength [dbget ${inst_ptr}.instTerms]]
    set density [expr $pin_count*1.0/$area]
    lappend pin_density $density
  }
  # puts "$pin_density"
  ## Compute mean and std
  set mean [expr [tcl::mathop::+ {*}$pin_density]*1.0 / [llength $pin_density]]
  set variance 0
  foreach density $pin_density {
    set variance [expr $variance + ($density - $mean)*($density - $mean)]
  }
  set std [expr sqrt(${variance}*1.0/[llength $pin_density])]
  set max_density [lindex [lsort -real $pin_density] end]
  ## Add cell padding for mean + top*std density
  set threshold [expr $mean + $top*$std]
  echo "Threshold: $threshold Mean: $mean Std: $std Max Density: $max_density" >> cell_padding_details
  set i 0
  while { $i < [llength $pin_density] } {
    set pd [lindex $pin_density $i]
    set term_count [llength [dbget [lindex $inst_ptrs $i].instTerms]]
    if { $pd > $threshold  && $term_count > $io_count } {
      set inst_name [dbget [lindex $inst_ptrs $i].name]
      set cell_name [dbget [lindex $inst_ptrs $i].cell.name]
      echo "Adding padding to $inst_name ($cell_name)  Pin Density: $pd Threshold: $threshold" >> cell_padding_details
      specifyInstPad $inst_name -right 1 -left 1
    }
    incr i
  }
}

proc add_cell_pad_test { cluster_ptr {top 0} {io_count 4} } {
  set inst_ptrs [dbget ${cluster_ptr}.members]
  set pin_density {}
  set inst_list {}
  set pin_counts {}
  foreach inst_ptr $inst_ptrs {
    set area [dbget ${inst_ptr}.area]
    set pin_count [llength [dbget ${inst_ptr}.instTerms]]
    if { $pin_count <= $io_count } {
      continue
    }
    set density [expr $pin_count*1.0/$area]
    lappend pin_density $density
    lappend inst_list $inst_ptr
    lappend pin_counts $pin_count
  }
  # puts "$pin_density"
  ## Compute mean and std
  set mean [expr [tcl::mathop::+ {*}$pin_density]*1.0 / [llength $pin_density]]
  set variance 0
  foreach density $pin_density {
    set variance [expr $variance + ($density - $mean)*($density - $mean)]
  }
  set std [expr sqrt(${variance}*1.0/[llength $pin_density])]
  set max_density [lindex [lsort -real $pin_density] end]
  ## Add cell padding for mean + top*std density
  set threshold [expr $mean + $top*$std]
  echo "Threshold: $threshold Mean: $mean Std: $std Max Density: $max_density Number of instances: [llength $pin_density]" >> clustering_details_pad
  set i 0
  while { $i < [llength $pin_density] } {
    set pd [lindex $pin_density $i]
    set term_count [lindex $pin_counts $i]
    if { $pd > $threshold  && $term_count > $io_count } {
      set inst_name [dbget [lindex $inst_list $i].name]
      set cell_name [dbget [lindex $inst_list $i].cell.name]
      echo "$i Adding padding to $inst_name ($cell_name) Pin Density: $pd Threshold: $threshold" >> clustering_details_pad
      # specifyInstPad $inst_name -right 1 -left 1
      specifyInstPad $inst_name -right 1
    }
    incr i
  }
}

proc add_cell_pad_to_all_cluster { {top 1} {delete_cluster 0} {io_count 4} } {
  set cluster_ptrs [dbget top.fplan.groups]
  setPlaceMode -place_detail_honor_inst_pad true
  foreach cluster_ptr $cluster_ptrs {
    add_cell_pad $cluster_ptr $top $io_count
  }

  if { $delete_cluster } {
    deleteAllInstGroups
  }
}


proc report_cluster_details {} {
  set cluster_ptrs {}
  foreach box [dbget [dbget top.markers.type Geometry -p ].box] {
    foreach cluster_ptr [dbget [dbQuery -areas ${box} -objType inst ].group -u -e] {
      lappend cluster_ptrs $cluster_ptr
    }
  }
  
  # Initialize an associative array to store counts
  array unset counts  ;# Ensure the array is empty
  foreach item $cluster_ptrs {
    incr counts($item)
  }

  ## Unique cluster_ptrs list
  set cluster_ptrs [lsort -unique $cluster_ptrs]

  foreach cluster_ptr $cluster_ptrs {
    set inst_count [llength [ dbget ${cluster_ptr}.members ]]
    set rent [report_rent_con $cluster_ptr]
    set rent_area [report_rent_con_area $cluster_ptr]
    set cluster_name [dbget ${cluster_ptr}.name]
    puts "Cluster: $cluster_name #DRC:$counts($cluster_ptr) Inst Count: $inst_count Rent: $rent Rent Area: $rent_area"
  }
}

proc report_cluster_details_wo_drc { {inst_threshold 3000} } {
  set cluster_ptrs [dbget top.fplan.groups]
  echo "Number of cluster:[llength $cluster_ptrs]" > clustering_details
  # Initialize an associative array to store counts
  array unset counts  ;# Ensure the array is empty
  foreach item $cluster_ptrs {
    incr counts($item)
  }

  ## Unique cluster_ptrs list
  set cluster_ptrs [lsort -unique $cluster_ptrs]

  foreach cluster_ptr $cluster_ptrs {
    set inst_count [llength [ dbget ${cluster_ptr}.members ]]
    set rent [report_rent_con $cluster_ptr]
    set rent_area [report_rent_con_area $cluster_ptr]
    set cluster_name [dbget ${cluster_ptr}.name]
    echo "Cluster: $cluster_name #DRC:$counts($cluster_ptr) Inst Count: $inst_count Rent: $rent Rent Area: $rent_area" >> clustering_details
    
    if { $inst_count > $inst_threshold } {
      if { [info exist ::env(ADd_CELL_PAD)] && $::env(ADd_CELL_PAD) == 1 } {
        if { $rent >= 0.83 || $rent < 0.61 } {
          add_cell_pad_test $cluster_ptr 0.2 4
        }
        deleteInstGroup $cluster_name
      } elseif { [info exist ::env(ADD_SOFT_GUIDE_SMALL)] \
        && $::env(ADD_SOFT_GUIDE_SMALL) == 1 } {
        if { $rent < 0.83 } {
          echo "Soft Guide Cluster: $cluster_name Rent: $rent" >> clustering_details
          createSoftGuide $cluster_name
        } else {
          deleteInstGroup $cluster_name
        }
      } elseif { [info exist ::env(ADD_SOFT_GUIDE_LARGE)] \
        && $::env(ADD_SOFT_GUIDE_LARGE) == 1 } {
        if { $rent >= 0.83 } {
          echo "Soft Guide Cluster: $cluster_name Rent: $rent" >> clustering_details
          createSoftGuide $cluster_name
        } else {
          deleteInstGroup $cluster_name
        }
      } elseif { [info exist ::env(ADD_SOFT_GUIDE_ALL)] \
        && $::env(ADD_SOFT_GUIDE_ALL) == 1 } {
          echo "Soft Guide Cluster: $cluster_name Rent: $rent" >> clustering_details
          createSoftGuide $cluster_name
        }  else {
        deleteInstGroup $cluster_name
      }
    } else {
      deleteInstGroup $cluster_name
    }
  }
}

## Write left right spacing for all the instances
proc get_left_right_spacing { inst_ptr } {
  set box [dbget ${inst_ptr}.box]
  set pty [dbget ${inst_ptr}.pt_y]
  set ptx [dbget ${inst_ptr}.pt_x]
  set neighbor_inst_ptx [dbget [dbget [dbQuery -areas ${box} -abut_only \
               -objType inst].pt_y $pty -e -p ].pt_x -e]
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
    set design [dbGet top.name]
    set fp [open "${design}_cong_cells.txt" w]
    puts $fp "Name,PinC,Left,Right"
    foreach inst_ptr $inst_ptr_list {
      set inst_typ [dbGet ${inst_ptr}.cell.subClass]
      if { $inst_typ != "block" } {
        set inst_name [dbGet ${inst_ptr}.name]
        set pin_count [llength [dbGet ${inst_ptr}.instTerms]]
        set spacing [get_left_right_spacing $inst_ptr]
        set lspacing [lindex $spacing 0]
        set rspacing [lindex $spacing 1]
        puts $fp "$inst_name,$pin_count,$lspacing,$rspacing"
      }
    }
    close $fp
  }
  return $inst_ptr_list
}

## Floorplan resize functions
proc expand_fp_by_site { site_count_h site_count_v row_flip} {
  set core_width [dbget top.fplan.coreBox_sizex]
  set core_height [dbget top.fplan.coreBox_sizey]
  set site_width [lindex [lsort [dbget head.sites.size_x]] 0]
  set site_height [lindex [lsort [dbget head.sites.size_y]] 0]
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

proc write_pin_details { file_name } {
  set fp [open $file_name w]
  puts $fp "Name,X,Y,Layer"
  foreach term_ptr [dbget top.terms] {
    set term_name [concat {*}[dbget ${term_ptr}.name]]
    set x [dbget ${term_ptr}.pt_x]
    set y [dbget ${term_ptr}.pt_y]
    set layer [dbget ${term_ptr}.layer.name]
    puts $fp "$term_name,$x,$y,$layer"
  }
  close $fp
}

proc create_fplan { width height c2d x_mul y_mul } {
  set new_width [expr $width * $x_mul]
  set new_height [expr $height * $y_mul]
  floorPlan -s $new_width $new_height $c2d $c2d $c2d $c2d
}

proc place_pins { pin_file suffix prefix x_shift x_flip y_shit y_flip width height } {
  ## Read the pin file
  if {![file exists $pin_file]} {
    puts "Error: Pin file '$pin_file' not found."
    return
  }

  set pin_fp [open $pin_file r]
  setPinAssignMode -pinEditInBatch true

  while {[gets $pin_fp line] >= 0} {
    # Skip empty lines and comments
    set items [split $line ',']
    if {$line eq "" || [string match "Name*" $line]} {
      continue
    }
    set pin_name [lindex $items 0]
    ## New pin_name is ${prefix}${pin_name}${suffix} when pin is not a bus.
    ## If pin is a bus e.g., xx[1] then new pin name is ${prefix}xx${suffix}[1]
    ## If ${prefix} is empty then ignore prefix similarly for $suffix
    if {[regexp {^(.*?)(\[[0-9]+\])$} $pin_name match base bus_index]} {
      set new_pin "${prefix}${base}_${suffix}${bus_index}"
    } else {
      set new_pin "${prefix}${pin_name}_${suffix}"
    }

    set pt_x [lindex $items 1]
    set pt_y [lindex $items 2]
    set layer [lindex $items 3]

    ## Check the new pin exists otherwise place it on center of the left boundary
    set isNewPin [dbget top.terms.name $new_pin -e]
    if { $isNewPin == "" } {
      puts "Error: Pin $new_pin does not exist."
      continue
    }
    
    ## New pin location if x_shift is 1 then shift by width, if x_flip is 1 then
    ## new location will be 2*height - pt_y similarly do it for y_shift 
    ## and y_flip
    set new_pt_x $pt_x
    set new_pt_y $pt_y
    if { $x_shift == 1 } {
      set new_pt_x [expr {$pt_x + $width}]
    }
    if { $x_flip == 1 } {
      set new_pt_y [expr {2*$height - $pt_y}]
    }
    if { $y_shit == 1 } {
      set new_pt_y [expr {$pt_y + $height}]
    }
    if { $y_flip == 1 } {
      set new_pt_x [expr {2*$width - $pt_x}]
    }

    set fp_width [dbget top.fplan.box_sizex]
    set fp_height [dbget top.fplan.box_sizey]
    ## Now find the side based on bbox. If the new pin is on the left boundary
    ## side is LEFT, on the right it is RIGHT and so on. Now find the nearest
    ## edge and assing side for the pin
    set dist_left $new_pt_x
    set dist_right [expr {$fp_width - $new_pt_x}]
    set dist_bottom $new_pt_y
    set dist_top [expr {$fp_height - $new_pt_y}]
    
    set side "LEFT"
    set min_dist $dist_left
    if { $dist_right < $min_dist } {
      set min_dist $dist_right
      set side "RIGHT"
    }
    if { $dist_bottom < $min_dist } {
      set min_dist $dist_bottom
      set side "BOTTOM"
    }
    if { $dist_top < $min_dist } {
      set min_dist $dist_top
      set side "TOP"
    }
    
    editPin -pin $new_pin -layer $layer -assign [list $new_pt_x $new_pt_y] -side $side
  }
  close $pin_fp
  setPinAssignMode -pinEditInBatch false
}

proc place_common_pins { pin_list side layer } {
  # If side is LEFT then put it at the center of left boundary, similarly for other sides
  set fp_width [dbget top.fplan.box_sizex]
  set fp_height [dbget top.fplan.box_sizey]
  switch -- $side {
    LEFT {
      set x 0
      set y [expr {$fp_height / 2}]
    }
    RIGHT {
      set x $fp_width
      set y [expr {$fp_height / 2}]
    }
    BOTTOM {
      set x [expr {$fp_width / 2}]
      set y 0
    }
    TOP {
      set x [expr {$fp_width / 2}]
      set y $fp_height
    }
    default {
      puts "Error: Unknown side $side"
      return
    }
  }
  foreach pin $pin_list {
    editPin -pin $pin -assign [list $x $y] -side $side -layer $layer -fixOverlap 1
  }
}