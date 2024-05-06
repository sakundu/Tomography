proc gen_rpt {} {
  set flop_count [ llength [dbget top.insts.cell.name *DFF* ] ]
  set buffer_count [llength [dbget top.insts.cell.isBuffer 1 ] ]
  set inv_count [ llength [dbget top.insts.cell.isInverter 1] ]
  set seq_count [ llength  [dbget top.insts.cell.isSequential 1] ]
  set total_cell_count [ llength [dbget top.insts] ]
  set comb_cell_count [ expr $total_cell_count - $seq_count - $inv_count - $buffer_count ]
  set buf_invs_count [ expr $inv_count + $buffer_count ]
  puts "$flop_count $comb_cell_count $buf_invs_count"

  ## Net Fanout
  set netFanout {}
  foreach netPtr [dbget top.nets.isPwrOrGnd 0 -p ] {
      set c1 [llength [dbget ${netPtr}.instTerms.name -e ]]
      set c2 [llength [dbget ${netPtr}.terms.name -e]]
      set fanout [ expr $c1 + $c2]
      if { $fanout == 0 } {
        continue
      }
      lappend netFanout $fanout
  }

  set uniq_fanout [lsort -unique $netFanout]

  puts "FanOut Count"
  foreach uf $uniq_fanout {
      set count [llength [lsearch -all $netFanout $uf]]
      puts "$uf $count"
  }
}

proc extract_congestion { {count 100} {suffix ""} } {
  ## Write out the congestion report in Congestion_PID.rpt file
  if { $suffix == "" } {
    set suffix [pid]
  }
  set rpt_file "congestion_${suffix}.rpt"
  reportCongestion -hotSpot -num_hotspot $count > $rpt_file
  
  ## Read the report file and extract the congestion information
  set fp [open $rpt_file r]
  while { [gets $fp line] >= 0 } {
    set items [regexp -all -inline {[0-9,\.]+} $line]
    if { [llength $items] == 6 } {
      set id [lindex $items 0]
      set box [lrange $items 1 4]
      set congestion [lindex $items 5]
      set type_text "Congestion Id: ${id} Score: ${congestion}"
      createMarker -bbox $box -type $type_text
    }
  }
  close $fp
  exec rm -rf $rpt_file
}

proc drc_markers_rpt { file_name } {
  set fp [open $file_name w]
  foreach drc_ptr [dbget top.markers.type Geometry -p -e] {
    set bbox [concat {*}[dbget $drc_ptr.box]]
    set box_x [expr ([lindex $bbox 2] + [lindex $bbox 0])/2.0]
    set box_y [expr ([lindex $bbox 3] + [lindex $bbox 1])/2.0]
    puts $fp "$box_x $box_y"
  }
  close $fp
}

proc drc_box_rpt { file_name } {
  set fp [open $file_name w]
  foreach drc_ptr [dbget top.markers.type Geometry -p -e] {
    set bbox [concat {*}[dbget $drc_ptr.box]]
    puts $fp "$bbox"
  }
  close $fp
}

proc gen_rpt { prefix } {
  dumpCongestArea -all congest_area_${prefix}.rpt
  reportCongestion -hotSpot -num_hotspot 100 > congestion_${prefix}.rpt
  # drc_box_rpt drc_box_${prefix}.rpt
}

proc add_checker_board_r_blockage { llx lly urx ury size_x size_y v1 v2} {
  set x $llx
  set y $lly
  set v [list $v1 $v2]
  set j 0
  while { $y < $ury } {
    set y1 [expr $y + $size_y]
    set i $j
    while { $x < $urx } {
      set x1 [expr $x + $size_x]
      set box [list $x $y $x1 $y1]
      createRouteBlk -box $box -partial [lindex $v $i]
      set x $x1
      set i [expr ($i + 1) % 2]
    }
    set x $llx
    set y $y1
    set j [expr ($j + 1) % 2]
  }
}
