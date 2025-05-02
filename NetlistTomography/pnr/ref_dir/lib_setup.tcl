# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
# lib and lef, RC setup

set libdir "../ng45/lib"
set lefdir "../ng45/lef"
set qrcdir "../ng45/qrc"

set_db init_lib_search_path { \
    ${libdir} \
    ${lefdir} \
}

set libworst "  
    ${libdir}/NangateOpenCellLibrary_typical.lib \
    ${libdir}/fakeram45_1024x39.lib \
    ${libdir}/fakeram45_128x32.lib  \
    ${libdir}/fakeram45_128x50.lib  \
    ${libdir}/fakeram45_128x60.lib  \
    ${libdir}/fakeram45_160x118.lib \
    ${libdir}/fakeram45_2048x42.lib \
    ${libdir}/fakeram45_256x12.lib  \
    ${libdir}/fakeram45_256x32.lib  \
    "

set libbest " 
    ${libdir}/NangateOpenCellLibrary_typical.lib \
    ${libdir}/fakeram45_1024x39.lib \
    ${libdir}/fakeram45_128x32.lib  \
    ${libdir}/fakeram45_128x50.lib  \
    ${libdir}/fakeram45_128x60.lib  \
    ${libdir}/fakeram45_160x118.lib \
    ${libdir}/fakeram45_2048x42.lib \
    ${libdir}/fakeram45_256x12.lib  \
    ${libdir}/fakeram45_256x32.lib  \
    "

set lefs "  
    ${lefdir}/NangateOpenCellLibrary.tech.lef \
    ${lefdir}/NangateOpenCellLibrary.macro.mod.lef \
    ${lefdir}/fakeram45_1024x39.lef \
    ${lefdir}/fakeram45_128x32.lef  \
    ${lefdir}/fakeram45_128x50.lef  \
    ${lefdir}/fakeram45_128x60.lef  \
    ${lefdir}/fakeram45_160x118.lef \
    ${lefdir}/fakeram45_2048x42.lef \
    ${lefdir}/fakeram45_256x12.lef  \
    ${lefdir}/fakeram45_256x32.lef  \
    "

set qrc_max "${qrcdir}/NG45.tch"
set qrc_min "${qrcdir}/NG45.tch"
#
# Ensures proper and consistent library handling between Genus and Innovus
#set_db library_setup_ispatial true
setDesignMode -process 45
