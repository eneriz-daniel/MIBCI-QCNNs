open_project MI-QCNN-project-4-5
set_top MIBCI_QCNN
add_files MIBCI-QCNN.cpp
add_files -tb MIBCI-QCNN-tb.cpp
open_solution "solution1" -reset
set_part {xc7z010-clg400-1}
create_clock -period 10 -name default
#source "./MI-QCNN-project-4-5/solution1/directives.tcl"
csim_design