# Copyright (C) 2021 Daniel Enériz Orta and Ana Caren Hernández Ruiz
# 
# This file is part of MIBCI-QCNNs.
# 
# MIBCI-QCNNs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# MIBCI-QCNNs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with MIBCI-QCNNs.  If not, see <http://www.gnu.org/licenses/>.


from os import system
from string import Template
from time import sleep


def launch_csim(modelpath: str, T: int, ds: int, Nchans: int, Nclasses: int, initial_fold: int = 0, max_fold: int = 5) -> None:
    """Launches HLS C simulations to validate the model. It splits the
    simulations per folds, so shorter time is required to complete the
    validation. Each simulation is launched as a background process using
    `screen`. Make sure to call this program on a bash with a `vivado_hls`
    command activated. This can be done with:
    ````
    source /directory/to/Vivado/install/settings64.sh
    ```

    Args:
        modelpath (str): Entire path to the model directory. It must contain the
            validation dataset `valdiationDS` and model parameters `npyparams`
            subfolders in each fold. These can be created using the `createnpys`
            function. 
        T (int): Time window used for this model.
        ds (int): Downsampling factor used for this model.
        Nchans (int): Number of channels used for this model.
        Nclasses (int): Number of classes used for this model.
        initial_fold (int, optional): Initial fold to launch. Defaults to 0.
        max_fold (int, optional): Maximum fold to launch. Defaults to 5.
    """
    fs=160

    for fold in range(initial_fold, max_fold):

        # Change the console output of the testbench and the model path
        with open('MIBCI-QCNN-tb-template.txt') as template:
            tb_template = Template(template.read())
        with open('MIBCI-QCNN-tb.cpp', 'w') as tb:
            tb.write(tb_template.substitute(model_path=modelpath, initial_fold=fold, max_fold=fold+1))

        # Change the Conv's layer tripcount details in the source file
        with open('MIBCI-QCNN-template.txt') as template:
            s_template = Template(template.read())
        with open('MIBCI-QCNN.cpp', 'w') as s:
            s.write(s_template.substitute(min_tripcount_conv=int(fs/(4*ds)), max_tripcount_conv=int(fs/(2*ds))))
        
        # Change the dataset parameters in the header file
        with open('MIBCI-QCNN-h-template.txt') as template:
            h_template = Template(template.read())
        with open('MIBCI-QCNN.h', 'w') as h:
            h.write(h_template.substitute(Nclasses=Nclasses, ds=ds, t=T, Nchans=Nchans))
        
        # Change the project folder name in the tcl file
        with open('csim-launcher-template.txt') as template:
            tcl_template = Template(template.read())
        with open('csim-launcher.tcl', 'w') as tcl:
            tcl.write(tcl_template.substitute(initial_fold=fold, max_fold=fold+1))
        
        # Launch the simulation
        system('screen -d -m -S fold{} -L -Logfile fold{}.log vivado_hls csim-launcher.tcl'.format(fold, fold))

        sleep(10)

def launch_csim_noparallel(modelpath: str, T: int, ds: int, Nchans: int, Nclasses: int) -> None:
    """Launches HLS C simulations to validate the model. Make sure to call this
    function on a bash with a `vivado_hls` command activated. This can be done with:
    ````
    source /directory/to/Vivado/install/settings64.sh
    ```

    Args:
        modelpath (str): Entire path to the model directory. It must contain the
            validation dataset `valdiationDS` and model parameters `npyparams`
            subfolders in each fold. These can be created using the `createnpys`
            function. 
        T (int): Time window used for this model.
        ds (int): Downsampling factor used for this model.
        Nchans (int): Number of channels used for this model.
        Nclasses (int): Number of classes used for this model.
    """
    fs=160

    # Change the console output of the testbench and the model path
    with open('MIBCI-QCNN-tb-template.txt') as template:
        tb_template = Template(template.read())
    with open('MIBCI-QCNN-tb.cpp', 'w') as tb:
        tb.write(tb_template.substitute(model_path=modelpath, initial_fold=0, max_fold=5))

    # Change the Conv's layer tripcount details in the source file
    with open('MIBCI-QCNN-template.txt') as template:
        s_template = Template(template.read())
    with open('MIBCI-QCNN.cpp', 'w') as s:
        s.write(s_template.substitute(min_tripcount_conv=int(fs/(4*ds)), max_tripcount_conv=int(fs/(2*ds))))
    
    # Change the dataset parameters in the header file
    with open('MIBCI-QCNN-h-template.txt') as template:
        h_template = Template(template.read())
    with open('MIBCI-QCNN.h', 'w') as h:
        h.write(h_template.substitute(Nclasses=Nclasses, ds=ds, t=T, Nchans=Nchans))
    
    # Change the project folder name in the tcl file
    with open('csim-launcher-template.txt') as template:
        tcl_template = Template(template.read())
    with open('csim-launcher.tcl', 'w') as tcl:
        tcl.write(tcl_template.substitute(initial_fold=0, max_fold=5))
    
    # Launch the simulation
    system('vivado_hls csim-launcher.tcl')

def launch_synth(T: int, ds: int, Nchans: int, Nclasses: int) -> None:
    """This allows to launch an Vivado HLS synthesis. Must be run in a bash with
    the `vivado_hls` command activated. This can be done with:
    ````
    source /directory/to/Vivado/install/settings64.sh
    ```

    Args:
        T (int): Time window of the model.
        ds (int): Downsampling factor of the model.
        Nchans (int): Number of channels of the model.
        Nclasses (int): Number of classes of the model.
    """


    fs = 160
    
    # Change the dataset parameters in the header file
    with open('MIBCI-QCNN-h-template.txt') as template:
        h_template = Template(template.read())
    with open('MIBCI-QCNN.h', 'w') as h:
        h.write(h_template.substitute(Nclasses=Nclasses, ds=ds, t=T, Nchans=Nchans))

    # Change the Conv's layer tripcount details in the source file
    with open('MIBCI-QCNN-template.txt') as template:
        s_template = Template(template.read())
    with open('MIBCI-QCNN.cpp', 'w') as s:
        s.write(s_template.substitute(min_tripcount_conv=int(fs/(4*ds)), max_tripcount_conv=int(fs/(2*ds))))

    # Launch the simulation
    system('vivado_hls synth-launcher.tcl')
