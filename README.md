# MIBCI-QCNNs

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

In this project a Brain Computer Interface (BCI) for Motor Imagery (MI) tasks is explored using Quantized Convolutional Neural Networks (QCNNs). Concretely, the model is based on the [EEGNet](https://arxiv.org/abs/1611.08024) and trained over the [Physionet Motor Movement/Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/). #TODO: Ampliar un poco mas

The [Red Pitaya STEMLab 125-10](https://www.redpitaya.com/f130/STEMlab-board) board is the targeted platform to run the hardware design of the model. It mounts a [Xilinx Zynq 7010](https://www.xilinx.com/support/documentation/data_sheets/ds190-Zynq-7000-Overview.pdf) System on Chip (SoC) with the low-spec XC7Z010 FPGA and a dual-core ARM Cortex-A9 CPU. #TODO: Poner las formas de reducir el tamaño del modelo: datareduction atrategies y la cuanitzacion

The hardware has been developed in Vivado HLS using an own-developed of the model in C++. Once the design is synthesized, an IP is exported for its integration using the Vivado IP integrator. This allows the creation of a `.bit` file, the bitstream, that can be load in the FPGA. Taking advantage of the CPU present in the Red Pitaya and its Jupyter-based interface a Python driver has been created to interface to the custom QCNN accelerator from the CPU.
 

## Description of archive
