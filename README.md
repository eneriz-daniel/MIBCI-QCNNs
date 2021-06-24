# MIBCI-QCNNs

![EEGNet-based model architecture](img/EEGNet.jpg)

This repo contains the source code of the project "FPGA implementation of BCIs using QCNNs" submitted to the [Xilinx Open Hardware Design Competition 2021](http://www.openhw.eu/).


- Team number: **xohw21-129**
- Project name: **FPGA implementation of BCIs using QCNNs**
- [Link to YouTube Video]()
- [Link to project repository](https://github.com/eneriz-daniel/MIBCI-QCNNs)

- University name: [University of Zaragoza](unizar.es)
- Participants:
    - Daniel Enériz Orta (eneriz@unizar.es)
    - Ana Caren Hernández Ruiz (anaacaren@unizar.es)

- Supervisor: Nicolás Medrano Marqués (nmedrano@unizar.es)


- Board used: Red Pitaya STEMLab 125-10 (XC7Z010)
- Software Version: Red Pitaya OS 1.04-7 

## Brief description of project

In this project a Brain Computer Interface (BCI) is explored using Quantized Convolutional Neural Networks (QCNNs) that process Electroencephalograph (EEG) data in order to recognize a Motor Imagery (MI) task. Concretely, the model is based on the [EEGNet](https://arxiv.org/abs/1611.08024) and trained over the [Physionet Motor Movement/Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/). This is a publicly available dataset composed of 64 EEG recordings of 109 subjects that performed the following motor imagery tasks:
1. Opening and closing the right fist (R).
1. Opening and closing the left fist (L).
1. Opening and closing both fists (B).
1. Opening and closing both feet (F).

Additionally a baseline data was acquired for the subjects when they were resting, forming a resting class (0). As other works that used this dataset, 2-, 3- and 4-classes classification task are explored using the tasks subsets L/R, L/R/0 and L/R/0/F, respectively.

The [Red Pitaya STEMLab 125-10](https://www.redpitaya.com/f130/STEMlab-board) board is the targeted platform to run the hardware design of the model. It mounts a [Xilinx Zynq 7010](https://www.xilinx.com/support/documentation/data_sheets/ds190-Zynq-7000-Overview.pdf) System on Chip (SoC) with the low-spec XC7Z010 FPGA and a dual-core ARM Cortex-A9 CPU. Two strategies are followed to reduce the FPGA's resources consumption:

1. The reduction of the model's input data size using the data-reduction methods presented [in this work](https://arxiv.org/abs/2004.00077) that consist on the reduction of the input's time window, the downsampling of the EEG recordings and the reduction of the number of EEG channels.
1. The use of fixed-point datatypes to represent the inputs, parameters, feature maps and outputs in the FPGA design. Due to it gets the best accuracy-resources footprint, a 16 bits fixed-point datatype with 8 bits for the integer part is selected. The Vivado HLS `ap_fixed.h` library was used for this purpose.

The hardware has been developed in Vivado HLS using an own-developed of the model in C++. Once the design is synthesized, an IP is exported for its integration using the Vivado IP integrator. This allows the creation of a `.bit` file, the bitstream, that can be load in the FPGA. Taking advantage of the CPU present in the Red Pitaya and its Jupyter-based interface a Python driver has been created to interface to the custom QCNN accelerator from the CPU.
 

## Description of archive
