// Copyright (C) 2021 Daniel Eneriz Orta
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
// 
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


/*------------------------------------------------------------------------------
-------------------------------DEFINING DATA TYPE-------------------------------
------------------------------------------------------------------------------*/

#define FIXED // FLOAT or FIXED

/*------------------------------------------------------------------------------
-------------------------------DEPENDENCIES-------------------------------------
------------------------------------------------------------------------------*/

#include <algorithm> //To use max() and min() (for 'same' padding method)
#ifdef FIXED
    #include "ap_fixed.h" //Fixed type support
    #include "hls_math.h" //If fixed type is used
#endif
#ifdef FLOAT
    #include <cmath> //If float type is used
#endif


/*------------------------------------------------------------------------------
-------------------------------TESTING METHOD PARAMETERS------------------------
------------------------------------------------------------------------------*/

#define N_FOLDS 5

/*------------------------------------------------------------------------------
---------------------NEURAL NETOWRK ARCHITECTURE PARAMETERS---------------------
------------------------------------------------------------------------------*/

// Input data parameters
#define N_CLASSES 4 //Number of classes to identify
#define FS 160.0 //Sampling frequency in Hz
#define DS 2.0 //Downsampling factor
#define T 3 //Input data time window in seconds
#define N_SUBS 105 //Number of subjects
#define CHANS 64 //Number of EEG channels

// Conv2D parameters
#define CONV2D_1_K_0 1 //Kernel's first dimension
#define CONV2D_1_K_1 int(FS/(2*DS)) // Kernel's second dimension
#define CONV2D_1_NF 4 //Number of filters

// Conv Leaky ReLU
#define CONV_LRELU_ALPHA 0.6

// DepthConv2D parameters
#define DEPTHCONV2D_1_K_0 CHANS //Kernel's first dimension
#define DEPTHCONV2D_1_K_1 1 // Kernel's second dimension
#define DEPTHCONV2D_1_D 2 //Depth multiplier

// Depth Leaky ReLU
#define DEPTH_LRELU_ALPHA 0.5

// First AvgPool2D parameters
#define AVGPOOL_1_K_0 1 //Kernel's first dimension
#define AVGPOOL_1_K_1 int(6/DS) //Kernel's second dimension

// SepConv2D parameters
#define SEPCONV2D_1_K_0 1 //Kernel's first dimension
#define SEPCONV2D_1_K_1 16 // Kernel's second dimension
#define SEPCONV2D_1_NF 8 //Number of filters

// Sep Leaky ReLU
#define SEP_LRELU_ALPHA 0.4

// Second AvgPool2D parameters
#define AVGPOOL_2_K_0 1 //Kernel's first dimension
#define AVGPOOL_2_K_1 8 //Kernel's second dimension



/*------------------------------------------------------------------------------
-------------------------------READ FROM NPY PARAMETERS-------------------------
------------------------------------------------------------------------------*/

//Max array sizes
#define MAX_NDARRAY_SIZE 100000 //Maximum number of elements to read in a NDARRAY
#define MAX_NDARRAY_DIM 4 //Maximum dimensions in a NDARRAY



/*------------------------------------------------------------------------------
-----------------------------NEURAL NETWORK DATATYPES---------------------------
------------------------------------------------------------------------------*/
#ifdef FIXED
    typedef ap_fixed<16,8,AP_RND,AP_SAT> apfixed;
    typedef int apint;
    typedef ap_uint<18> apuint;
#endif
#ifdef FLOAT
    typedef float apfixed;
    typedef int apint;
    typedef int apuint;
#endif


using namespace std;

void MIBCI_QCNN(apfixed X[CHANS][int(FS/DS)*T],
                apfixed conv_weights[CONV2D_1_K_0][CONV2D_1_K_1][CONV2D_1_NF],
                apfixed conv_lrelu_alpha,
                apfixed depth_weights[DEPTHCONV2D_1_K_0][DEPTHCONV2D_1_K_1][CONV2D_1_NF][DEPTHCONV2D_1_D],
                apfixed depth_lrelu_alpha,
                apfixed sepdepth_weights[SEPCONV2D_1_K_0][SEPCONV2D_1_K_1][CONV2D_1_NF*DEPTHCONV2D_1_D],
                apfixed seppointwise_weights[CONV2D_1_NF*DEPTHCONV2D_1_D][SEPCONV2D_1_NF],
                apfixed sep_lrelu_alpha,
                apfixed dense_weights[SEPCONV2D_1_NF*int(FS/DS)*T/AVGPOOL_1_K_1/AVGPOOL_2_K_1][N_CLASSES],
                apfixed dense_bias[N_CLASSES],
                apfixed Y[N_CLASSES]);
