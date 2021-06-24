open_project MIBCI-QCNN-synth
set_top MIBCI_QCNN
add_files MIBCI-QCNN.cpp
open_solution "solution1"
set_part {xc7z010clg400-1}
create_clock -period 10 -name default
source "pipeline-inner-loops-directives.tcl"
csynth_design