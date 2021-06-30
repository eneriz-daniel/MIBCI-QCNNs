# FPGA implementation of BCIs using QCNNs

![EEGNet-based model architecture](img/EEGNet.svg)

This repo contains the source code of the project "FPGA implementation of BCIs using QCNNs" submitted to the [Xilinx Open Hardware Design Competition 2021](http://www.openhw.eu/).


## Index

- [Project submission details](#project-submission-details)
- [Brief description of the project](#brief-description-of-the-project)
- [Description of the archive](#description-of-the-archive)
- [Instructions to build and test project](#instructions-to-build-and-test-project)

## Project submission details

- Team number: **xohw21-129**
- Project name: **FPGA implementation of BCIs using QCNNs**
- [Link to YouTube Video](https://youtu.be/s-FPJzREJCY)
- [Link to project repository](https://github.com/eneriz-daniel/MIBCI-QCNNs)

- University name: [University of Zaragoza](unizar.es)
- Participants:
    - Daniel Enériz Orta (eneriz@unizar.es)
    - Ana Caren Hernández Ruiz (anaacaren@unizar.es)

- Supervisor: Nicolás Medrano Marqués (nmedrano@unizar.es)


- Board used: Red Pitaya STEMLab 125-10 (XC7Z010)
- Software Version: Red Pitaya OS 1.04-7 

## Brief description of the project

In this project a Brain Computer Interface (BCI) is explored using Quantized Convolutional Neural Networks (QCNNs) that process Electroencephalograph (EEG) data in order to recognize a Motor Imagery (MI) task. Concretely, the model is based on the [EEGNet](https://arxiv.org/abs/1611.08024) and trained over the [Physionet Motor Movement/Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/). This is a publicly available dataset composed of 64 EEG recordings of 109 subjects that performed the following motor imagery tasks:
1. Opening and closing the right fist (R).
1. Opening and closing the left fist (L).
1. Opening and closing both fists (B).
1. Opening and closing both feet (F).

Additionally a baseline data was acquired for the subjects when they were resting, forming a resting class (0). As other works that used this dataset, 2-, 3- and 4-classes classification task are explored using the tasks subsets L/R, L/R/0 and L/R/0/F, respectively.

The [Red Pitaya STEMLab 125-10](https://www.redpitaya.com/f130/STEMlab-board) board is the targeted platform to run the hardware design of the model. It mounts a [Xilinx Zynq 7010](https://www.xilinx.com/support/documentation/data_sheets/ds190-Zynq-7000-Overview.pdf) System on Chip (SoC) with the low-spec XC7Z010 FPGA and a dual-core ARM Cortex-A9 CPU. Three strategies are followed to reduce the FPGA's resources consumption:

1. The reduction of the model's input data size using the data-reduction methods presented [in this work](https://arxiv.org/abs/2004.00077) that consist on the reduction of the input's time window, the downsampling of the EEG recordings and the reduction of the number of EEG channels.
1. The use of fixed-point datatypes to represent the inputs, parameters, feature maps and outputs in the FPGA design. Due to it gets the best accuracy-resources footprint, a 16 bits fixed-point datatype with 8 bits for the integer part is selected. The Vivado HLS `ap_fixed.h` library was used for this purpose.
1. The substitution of the original ELU activation fucntion for the LeakyReLU, a much cheaper when implementing in hardware.

The hardware has been developed in Vivado HLS using an own-developed of the model in C++. Once the design is synthesized, an IP is exported for its integration using the Vivado IP integrator. This allows the creation of a `.bit` file, the bitstream, that can be load in the FPGA. Taking advantage of the CPU present in the Red Pitaya and its Jupyter-based interface a Python driver has been created to interface to the custom QCNN accelerator from the CPU.
 

## Description of the archive

This is the repository tree:
```
MIBCI-QCNNs
├── csim-launcher.tcl
├── csim-launcher-template.txt
├── directives.tcl
├── img
│   ├── EEGNet.jpg
│   ├── EEGNet.png
│   └── EEGNet.svg
├── implementation.ipynb
├── LICENSE
├── MIBCI-QCNN.cpp
├── MIBCI-QCNN.h
├── MIBCI-QCNN-h-template.txt
├── MIBCI-QCNN-tb.cpp
├── MIBCI-QCNN-tb-template.txt
├── MIBCI-QCNN-template.txt
├── README.md
├── requirements.txt
├── synth-launcher.tcl
├── training.ipynb
├── usage.ipynb
└── utils
    ├── accuracy_test.py
    ├── createnpys.py
    ├── get_data.py
    ├── hlsparser.py
    ├── hls_tools.py
    └── train_tools.py
```

The text you are reading is in the [`README.md`](README.md) file and the license that protects the code in the repository is available in [`LICENSE`](LICENSE). Additionally, the header picture is included under the [`img/`](img) folder.

The core code is under the [`utils/`](utils) folder, where there are six files:
1. [`get_data.py`](utils/get_data.py). Adapted from [this code](https://github.com/MHersche/eegnet-based-embedded-bci/blob/master/get_data.py), allows the user to download the data and apply the data reduction methods.
1. [`train_tools.py`](utils/train_tools.py). It embeds the training process: normalizing the data, splitting the validation from the training set and training the global model.
1. [`createnpys.py`](utils/createnpys.py). Writes in a folder containing the a global-trained model the validation dataset and the model's parameters for each fold
1. [`hls_tools.py`](utils/hls_tools.py). Contains the Vivado HLS launchers for simulation and synthesis. If used, the functions must be called from a `vivado_hls`-enabled bash. The main version of the simulation launcher function, `launch_csim` splits the global model simulation per folds, so at least 5 CPU kernels must be free in that launch and the `screen` linux command must be available. Their functions depend on:
    1. The [`MIBCI-QCNN-tb-template.txt`](MIBCI-QCNN-tb-template.txt), [`MIBCI-QCNN-template.txt`](MIBCI-QCNN-template.txt) and [`MIBCI-QCNN-h-template.txt`](MIBCI-QCNN-h-template.txt), i.e. the source and testbench formatted files, enabling their control from Python.
    1. The `vivado_hls` simulation launcher, [`csim_lancher.tcl`](csim_lancher.tcl) and its template [`csim_lancher-template.txt`](csim_lancher-template.txt).
    1. The `vivado_hls` simulation launcher, [`synth-launcher.tcl`](synth-launcher.tcl) and its directives [`directives.tcl`](directives.tcl).
1. [`hlsparser.py`](utils/hlsparser.py). This code is fully authored by Tiago Lascasas dos Santos and is also available [here](https://github.com/tiagolascasas/Vivado-HLS-Report-Parser).
1. [`accuracy_test.py`](utils/accuracy_test.py). Here is a function that computes the validation accuracy of the HLS-simulated version of the model, saving a the validation accuracy-per-fold plot.

All of the functions contained in the six [`utils`](utils) files depend on some popular Python libraries, available in [`requirements.txt`](requirements.txt)

Most of the steps taken to develop the project ara available in the three Jupyter notebooks. All the global model training process is contained in the [`training.ipynb`](training.ipynb) notebook, relying in the [`get_data.py`](utils/get_data.py) and [`train_tools.py`](utils/train_tools.py) files. The simulation and synthetization steps of the HLS design are included in the [`implementation.ipynb`](implementation.ipynb). Which uses the remaining four `utils`, [`createnpys.py`](utils/createnpys.py), [`hls_tools.py`](utils/hls_tools.py), [`hlsparser.py`](utils/hlsparser.py) and [`accuracy_test.py`](utils/accuracy_test.py). Inside the [`usage.ipynb`](usage.ipynb) there is the code to run test the FPGA, this is the only code that must be run on the Zynq SoC. Its first cell explains its dependencies.

Finally, the source files ([`MIBCI-QCNN.cpp`](MIBCI-QCNN.cpp) and [`MIBCI-QCNN.h`](MIBCI-QCNN.h)) and the testbench ([`MIBCI-QCNN-tb.cpp`](MIBCI-QCNN-tb.cpp)) built for the `T=3`, `ds=2`, `Nchan=64` and `Nclasses=4` has been also added.

## Instructions to build and test project

> All the steps previous to the Red Pitaya execution has been tested in Ubuntu 20.04.2 LTS.

The notebooks are self-explained, so use the [`training.ipynb`](training.ipynb) to download the data and preprocess is according to the desired data reduction methods. At the end of the notebook you will get a folder called `global_model` with five subfolders containing the folds' parameters and their training details.

Then in [`implementation.ipynb`](implementation.ipynb) you will find the details to check the results when the model is implemented and synthesizing its design. At the synthetization process finalization, you will get a folder `MIBCI-QCNN-synth` containing an HLS project. 

Open the `MIBCI-QCNN-synth` project from the Vivado HLS GUI and export the design as an IP. To run the design on the FPGA you must integrate this IP in the Zynq Processing system with the Vivado IP integrator and then generate the bitstream. The description of this process is perfectly detailed in the fist minutes of this [FPGA Developer](https://youtu.be/Dupyek4NUoI) video.

Once the bitstream is generated upload it to the Red Pitaya using a SFTP service, as [Filezilla](https://filezilla-project.org/). You will like to save them in a Jupyter-accessible folder. In our case we created the `/home/jupyter/MIBCI-QCNNs/` folder and uploaded the bitstream and the [`usage.ipynb`](usage.ipynb) notebook to there. To test the FPGA performance, [`usage.ipynb`](usage.ipynb) is created to load each model's fold parameters from the foldes called `global_model/fold_i/npyparams` and the validation dataset from `global_model/fold_i/validationDS`, so you can just get the `global_model` directory of your training computer and upload it inside a folder called `global_model` at the same root as the [`usage.ipynb`](usage.ipynb) notebook. Inside the notebooks all the steps are explained.

If there is any issue, post them in the Github's Issues tab or [send us an email](mailto:eneriz@unizar.es).