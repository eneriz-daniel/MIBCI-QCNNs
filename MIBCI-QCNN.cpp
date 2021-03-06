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

#include "MIBCI-QCNN.h"

apfixed LReLU(apfixed x, apfixed alpha){

    /* Leaky Rectified Linear Unit implementation.

    Args:
      x - Input value
      alpha - Leaky ReLU alpha parameter
    Returns: The LReLU output of x
    */

    if(x<0) return alpha*x;
    else return x;
}

void Softmax(apfixed x[N_CLASSES], apfixed y[N_CLASSES]){

  /* Softmax implementation

  Args:
    x - Input array to perform softmax
    y - Array to save the softmax resultant values
  */
  
  apfixed expx[N_CLASSES];

  apfixed expsum = 0;

  SoftmaxAccLoop: for(apint i=0; i<N_CLASSES; i++){
    #ifdef FLOAT
      expx[i] = exp(x[i]);
    #endif
    #ifdef FIXED
      expx[i] = hls::expf(x[i]);
    #endif
    expsum += expx[i];
  }

  SoftmaxDivLoop: for(apint i=0; i<N_CLASSES; i++){
    y[i] = expx[i]/expsum;
  }
}

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
                apfixed Y[N_CLASSES]){
#pragma HLS INTERFACE s_axilite port=sep_lrelu_alpha
#pragma HLS INTERFACE s_axilite port=depth_lrelu_alpha
#pragma HLS INTERFACE s_axilite port=sepdepth_weights
#pragma HLS INTERFACE s_axilite port=X
#pragma HLS INTERFACE s_axilite port=dense_bias
#pragma HLS INTERFACE s_axilite port=Y
#pragma HLS INTERFACE s_axilite port=depth_weights
#pragma HLS INTERFACE s_axilite port=dense_weights
#pragma HLS INTERFACE s_axilite port=seppointwise_weights
#pragma HLS INTERFACE s_axilite port=conv_lrelu_alpha
#pragma HLS INTERFACE s_axilite port=conv_weights
  /* Performs the union of the Conv2d, LReLU, DepthConv2d, LReLU, AvgPool2d,
  SepConv2d, LReLU, AvgPool2d, Flatten, Dense and Softmax layers of Tensorflow
  using the 'same' padding method in the Conv2d and SepConv2d and 'valid'
  padding method in the others.

  Args:
    X - Matrix containing the input
    conv_weights - 3D array containing the weights of the Conv2D layer
    depth_weights - 4D array containing the weights of the DepthConv2D layer
    sepdepth_weights - 3D array containing the weights of the SepConv2D
                       depthwise layer
    seppointwise_weights - 2D array containing the weights of the SepConv2d 
                           pointwise layer
    dense_weights - 2D array containing the weights of the Dense layer
    dense_bias - 1D array containing the bias parameter of the Dense layer
    Y - Matrix to store the output
  */

  apfixed acc; // The accumulator

  apint l_min; // Two auxiliary variables for the fiter's positions
  apint l_max;

  // Feature maps initialization
  apfixed A[CHANS][int(FS/DS)*T][CONV2D_1_NF];
  apfixed B[int(FS/DS)*T][DEPTHCONV2D_1_D*CONV2D_1_NF];
  apfixed C[int(FS/DS)*T/AVGPOOL_1_K_1][DEPTHCONV2D_1_D*CONV2D_1_NF];
  apfixed D[int(FS/DS)*T/AVGPOOL_1_K_1][DEPTHCONV2D_1_D*CONV2D_1_NF];
  apfixed E[int(FS/DS)*T/AVGPOOL_1_K_1][SEPCONV2D_1_NF];
  apfixed F[int(FS/DS)*T/AVGPOOL_1_K_1/AVGPOOL_2_K_1][SEPCONV2D_1_NF];
  apfixed G[int(FS/DS)*T/AVGPOOL_1_K_1/AVGPOOL_2_K_1*SEPCONV2D_1_NF];
  apfixed H[N_CLASSES];

  /*----------------------------------------------------------------------------
  -----------------------------CONV2D-------------------------------------------
  ----------------------------------------------------------------------------*/

  // Iterate over the number of filters
  ConvFiltersLoop: for(apint k=0; k<CONV2D_1_NF; k++){

    // Iterate over the input matrix
    ConvChansLoop: for(apint j=0; j<CHANS; j++){
      ConvFramesLoop: for(apint i=0; i<int(FS/DS)*T; i++){
        // Calculate the auxiliary positions
        l_min = max(0, CONV2D_1_K_1/2-1-i);
        l_max = min(CONV2D_1_K_1, int(FS/DS)*T - i + CONV2D_1_K_1/2 - 1);
        acc = 0; // Reset the accumulator
        ConvWeightsLoop: for(apint l=l_min; l<l_max; l++){
			#pragma HLS LOOP_TRIPCOUNT min=20 max=40
          // Multiply the input and the weight
          acc += X[j][i-(CONV2D_1_K_1/2-1)+l]*conv_weights[0][l][k];
        }
        A[j][i][k] = LReLU(acc, conv_lrelu_alpha); // Save the accumulator value
      }
    }
  }

  /*----------------------------------------------------------------------------
  -----------------------------DEPTHCONV2D--------------------------------------
  ----------------------------------------------------------------------------*/

  DepthMulLoop: for(apint l=0; l<DEPTHCONV2D_1_D; l++) {
    // Iterate over the number of filters
    DepthFiltersLoop: for(apint k=0; k<CONV2D_1_NF; k++){

      // Iterate over the input matrix
      DepthFramesLoop: for(apint j=0; j<int(FS/DS)*T; j++){
        acc = 0;
        DepthChansLoop: for(apint i=0; i<CHANS; i++){
          acc += A[i][j][k] * depth_weights[i][0][k][l];
        }
        B[j][l+k*DEPTHCONV2D_1_D] = LReLU(acc, depth_lrelu_alpha); // Save the accumulator value
      }
    }
  }

  /*----------------------------------------------------------------------------
  -----------------------------AVGPOOL1-----------------------------------------
  ----------------------------------------------------------------------------*/

  AvgPool1FiltersLoop: for(apint k=0; k<CONV2D_1_NF*DEPTHCONV2D_1_D; k++){
    AvgPool1FramesLoop: for(apint j=0; j<int(FS/DS)*T; j+=AVGPOOL_1_K_1){
      acc = 0;
      AvgPool1PoolingLoop: for(apint i=0; i<AVGPOOL_1_K_1; i++){
        acc += B[j+i][k];
      }
      C[j/AVGPOOL_1_K_1][k] = acc/AVGPOOL_1_K_1;
    }
  }

  /*----------------------------------------------------------------------------
  -----------------------------SEPCONV2D----------------------------------------
  ----------------------------------------------------------------------------*/

  SepDepthFiltersLoop: for(apint k=0; k<CONV2D_1_NF*DEPTHCONV2D_1_D; k++){
    SepDepthFramesLoop: for(apint j=0; j<int(FS/DS)*T/AVGPOOL_1_K_1; j++){
      l_min = max(0,SEPCONV2D_1_K_1/2-1-j);
      l_max = min(SEPCONV2D_1_K_1,int(FS/DS)*T/AVGPOOL_1_K_1-j+SEPCONV2D_1_K_1/2-1);
      acc = 0;
      SepDepthKernelLoop: for(apint l=l_min; l<l_max; l++){
      #pragma HLS LOOP_TRIPCOUNT min=8 max=16
        acc += C[j-SEPCONV2D_1_K_1/2+1+l][k]*sepdepth_weights[0][l][k];
      }
      D[j][k] = acc; // Save the accumulator value
    }
  }

  SepPointwiseFiltersLoop: for(apint k=0; k<SEPCONV2D_1_NF; k++){
    SepPointwiseFramesLoop: for(apint j=0; j<int(FS/DS)*T/AVGPOOL_1_K_1; j++){
      acc = 0;
      SepPointwiseKernelLoop: for(apint l=0; l<CONV2D_1_NF*DEPTHCONV2D_1_D; l++){
        acc += D[j][l]*seppointwise_weights[l][k];
      }
      E[j][k] = LReLU(acc, sep_lrelu_alpha); // Save the accumulator value
    }
  }


  /*----------------------------------------------------------------------------
  -----------------------------AVGPOOL2-----------------------------------------
  ----------------------------------------------------------------------------*/


  AvgPool2FiltersLoop: for(apint k=0; k<SEPCONV2D_1_NF; k++){
    AvgPool2FramesLoop: for(apint j=0; j<int(FS/DS)*T/AVGPOOL_1_K_1; j+=AVGPOOL_2_K_1){
      acc = 0;
      AvgPool2PoolingLoop: for(apint i=0; i<AVGPOOL_2_K_1; i++){
        acc += E[j+i][k];
      }
      F[j/AVGPOOL_2_K_1][k] = acc/AVGPOOL_2_K_1;
    }
  }

  /*----------------------------------------------------------------------------
  -----------------------------FLATTEN------------------------------------------
  ----------------------------------------------------------------------------*/
  //This can be avoided flattening the second AvgPool2D outputs

  FlattenFiltersLoop: for(apint k=0; k<SEPCONV2D_1_NF; k++){
    FlattenFramesLoop: for(apint j=0; j<int(FS/DS)*T/AVGPOOL_1_K_1/AVGPOOL_2_K_1; j++){
      G[k+j*SEPCONV2D_1_NF] = F[j][k];
    }
  }

  /*----------------------------------------------------------------------------
  -----------------------------DENSE--------------------------------------------
  ----------------------------------------------------------------------------*/

  DenseOutLoop: for(apint k=0; k<N_CLASSES; k++){
    acc = dense_bias[k];
    DenseInLoop: for(apint j=0; j<int(FS/DS)*T/AVGPOOL_1_K_1/AVGPOOL_2_K_1*SEPCONV2D_1_NF; j++){
      acc += G[j]*dense_weights[j][k];
    }
    H[k] = acc;
  }

  Softmax(H, Y);

}
