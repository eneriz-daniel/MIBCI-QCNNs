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

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <bitset>
#include <stdexcept>
#include <typeinfo>
#include "MIBCI-QCNN.h"

/*----------------------------------------------------------------------------
--------------------------HELPFUL FUNCTIONS-----------------------------------
----------------------------------------------------------------------------*/

float TextStringToDouble(string words, int nbytes) {
  /* Takes a string containing binary data and returns the value
  associated to its representation, that could be either Float32
  (float) or Float64 (double).

  Args:
    words - String containing binary data to use as the
    representation of the floating point number

    nbytes - Number of bytes of the representation. Must be either
    4 (Float32) or 8 (Float64). Otherwise, an exception is thrown.

  Returns:
    The floating point number represented in the nbytes data.
  */

  reverse(words.begin(), words.end());
  string binaryString = "";

  union {
      double f;  // assuming 32-bit IEEE 754 single-precision
      uint64_t  i;    // assuming 32-bit 2's complement int
  } u64;

  union {
      float f;  // assuming 32-bit IEEE 754 single-precision
      int  i;    // assuming 32-bit 2's complement int
  } u32;
      
  for (char& _char : words) {
      binaryString +=bitset<8>(_char).to_string();
  }

  switch(nbytes){
    case 8:
      u64.i  = bitset<64>(binaryString).to_ullong();
      return u64.f;
    case 4:
      u32.i  = bitset<32>(binaryString).to_ulong();
      return u32.f;
    default:
      throw invalid_argument( "nbytes argument must be 4 or 8" );
  }
}

void GetFlatArrFromNpy(string npypath, float ndarray[MAX_NDARRAY_SIZE], int shape[MAX_NDARRAY_DIM]){
  /* Takes the path to a npy file containing a Numpy's n dimensional
  array of with np.single or np.double datatypes and fills ndarray with
  the flattened version the array and shape with the original shape.

  Args:
    npypath - String containing the .npy file path

    ndarray - Array to save the flattened version of the n dimensional
    array saved in the .npy. Its maximum size is MAX_NDARRAY_SIZE.

    shape - Array to save the shape of the n dimensional array. Its
    maximum size is MAX_NDARRAY_DIM.npy
  */   
  size_t loc1, loc2, loc3; //Three helpful variables when parsing the .npy header
  
  ifstream infile(npypath, ifstream::binary); //Opening the .npy file
  
  // Read the size
  int size;        
  infile.read(reinterpret_cast<char *>(&size), sizeof(size));
  
  // Allocate a string, make it large enough to hold the input
  string buffer;
  buffer.resize(size);
  
  // Read the text into the string
  infile.read(&buffer[0],  buffer.size() );
  infile.close();

  //PARSING THE HEADER
  //First, the data format is determined. Now only float is supported.
  loc1 = buffer.find("descr");
  loc2 = buffer.find(",", loc1+1);

  string descr = buffer.substr(loc1+1+6, loc2-loc1-1-6);

  int nbytes = (int)(descr[descr.find("<f")+2] - '0');
  
  //Second, the ndarray shape
  loc1 = buffer.find("shape");

  loc1 = buffer.find("(", loc1+1);
  loc2 = buffer.find(")", loc1+1);
  loc3 = buffer.find(",", loc1+1);

  string shape_str = buffer.substr(loc1+1, loc2-loc1-1);

  int ndim;
  if(loc2-loc3 == 1) ndim = 1; // One dimension array is shaped as (N,)
  else ndim = count(shape_str.begin(), shape_str.end(), ',') + 1;

  loc1 = -1;
  int nelements = 1;
  for(int i=0; i<ndim; i++){
    loc2 = shape_str.find(",", loc1+1);
    shape[i] = stoi(shape_str.substr(loc1+1, loc2-loc1-1), nullptr);
    nelements *= shape[i];
    loc1 = loc2;
  }
  
  //READING THE NDARRAY DATA
  string element_str;
  int elt_idx;
  int data_loc = buffer.find('\n', loc1+1)+1;

  for(elt_idx=0; elt_idx<nelements; elt_idx++){
    element_str = buffer.substr(data_loc+elt_idx*nbytes, nbytes);
    ndarray[elt_idx] = TextStringToDouble(element_str, nbytes);
  }
}

int argmax(apfixed x[N_CLASSES]){
  /* Argmax function. Takes an array and returns the index of
  the maximum element.

  Args:
    X - The array to find the index of the maximum element.
  Returns:
    The integer index of the maximum element in the input array.
  */
  
  apfixed comp = 0;
  int max_idx = 0;
  for(int i=0;i<N_CLASSES;i++){
    if(x[i]>comp){
      comp = x[i];
      max_idx = i;
    }
  }

  return max_idx;
}

int main() {

  //Root path of the model
  string model_path = "/home/daniel/BCI/XOH21/global_model/";
  char subdirectory[256];

  // Initialize model parameters variables
  float conv2d_1_w_tmp[CONV2D_1_K_0*CONV2D_1_K_1*CONV2D_1_NF];
  int conv2d_1_w_shape[4];
  apfixed conv2d_1_w[CONV2D_1_K_0][CONV2D_1_K_1][CONV2D_1_NF];

  float depthconv2d_1_w_tmp[DEPTHCONV2D_1_K_0*DEPTHCONV2D_1_K_1*DEPTHCONV2D_1_D*CONV2D_1_NF];
  int depthconv2d_1_w_shape[4];
  apfixed depthconv2d_1_w[DEPTHCONV2D_1_K_0][DEPTHCONV2D_1_K_1][CONV2D_1_NF][DEPTHCONV2D_1_D];

  float sepdepth_1_w_tmp[SEPCONV2D_1_K_0*SEPCONV2D_1_K_1*CONV2D_1_NF*DEPTHCONV2D_1_D];
  int sepdepth_1_w_shape[4];
  apfixed sepdepth_1_w[SEPCONV2D_1_K_0][SEPCONV2D_1_K_1][CONV2D_1_NF*DEPTHCONV2D_1_D];

  float seppointwise_1_w_tmp[CONV2D_1_NF*DEPTHCONV2D_1_D*SEPCONV2D_1_NF];
  int seppointwise_1_w_shape[4];
  apfixed seppointwise_1_w[CONV2D_1_NF*DEPTHCONV2D_1_D][SEPCONV2D_1_NF];

  float dense_1_w_tmp[SEPCONV2D_1_NF*int(FS/DS)*T/AVGPOOL_1_K_1/AVGPOOL_2_K_1*N_CLASSES];
  int dense_1_w_shape[4];
  apfixed dense_1_w[SEPCONV2D_1_NF*int(FS/DS)*T/AVGPOOL_1_K_1/AVGPOOL_2_K_1][N_CLASSES];

  float dense_1_b_tmp[N_CLASSES];
  int dense_1_b_shape[4];
  apfixed dense_1_b[N_CLASSES];
  
  // Initialize the array to read the input
  float X_tmp[CHANS*int(FS/DS)*T];
  int X_shape[2];
  apfixed X[CHANS][int(FS/DS)*T];

  // Initialize the array to save the output
  apfixed Y[N_CLASSES];


  #ifdef FIXED
  	  //Read fixed datatype details to include them in the output files names
	  int total_apfixed, int_apfixed;
	  sscanf(typeid(apfixed).name() , "8ap_fixedILi%dELi%dEL9ap_q_mode0EL9ap_o_mode0ELi0EE", &total_apfixed, &int_apfixed);
  #endif

  // Iterate over the folds
  for(int fold=4; fold<5; fold++){
  /*----------------------------------------------------------------------------
  --------------------------READING THE PARAMETERS------------------------------
  ----------------------------------------------------------------------------*/
  
    // Reading the weights of the conv2d_1 layer
    sprintf(subdirectory,"fold_%d/npyparams/conv2d_w.npy", fold);
    GetFlatArrFromNpy(model_path+subdirectory, conv2d_1_w_tmp, conv2d_1_w_shape);

    // Reshape it
    for(int k=0; k<CONV2D_1_K_0; k++){
      for(int j=0; j<CONV2D_1_K_1; j++){
        for(int i=0; i<CONV2D_1_NF; i++){
          conv2d_1_w[k][j][i] = conv2d_1_w_tmp[i+CONV2D_1_NF*j];
        }
      }
    }

    // Reading the weights of the depthconv2d_1 layer
    sprintf(subdirectory,"fold_%d/npyparams/depthconv2d_w.npy", fold);
    GetFlatArrFromNpy(model_path+subdirectory, depthconv2d_1_w_tmp, depthconv2d_1_w_shape);

    // Reshape it
    for(int l=0; l<DEPTHCONV2D_1_K_0; l++){
      for(int k=0; k<DEPTHCONV2D_1_K_1; k++){
        for(int j=0; j<CONV2D_1_NF; j++){
          for(int i=0; i<DEPTHCONV2D_1_D; i++){
            depthconv2d_1_w[l][k][j][i] = depthconv2d_1_w_tmp[i+DEPTHCONV2D_1_D*j+DEPTHCONV2D_1_D*CONV2D_1_NF*k+DEPTHCONV2D_1_D*CONV2D_1_NF*DEPTHCONV2D_1_K_1*l];
          }
        }
      }
    }

    // Reading the weights of the sepdepth sublayer
    sprintf(subdirectory,"fold_%d/npyparams/sepdepthconv2d_w.npy", fold);
    GetFlatArrFromNpy(model_path+subdirectory, sepdepth_1_w_tmp, sepdepth_1_w_shape);

    // Reshape it
    for(int l=0; l<SEPCONV2D_1_K_0; l++){
      for(int k=0; k<SEPCONV2D_1_K_1; k++){
        for(int j=0; j<CONV2D_1_NF*DEPTHCONV2D_1_D; j++){
          sepdepth_1_w[l][k][j] = sepdepth_1_w_tmp[j+DEPTHCONV2D_1_D*CONV2D_1_NF*k+DEPTHCONV2D_1_D*CONV2D_1_NF*SEPCONV2D_1_K_1*l];
        }
      }
    }

    // Reading the weights of the seppointwise sublayer
    sprintf(subdirectory,"fold_%d/npyparams/seppointconv2d_w.npy", fold);
    GetFlatArrFromNpy(model_path+subdirectory, seppointwise_1_w_tmp, seppointwise_1_w_shape);

    // Reshape it
    for(int l=0; l<CONV2D_1_NF*DEPTHCONV2D_1_D; l++){
      for(int k=0; k<SEPCONV2D_1_NF; k++){
        seppointwise_1_w[l][k] = seppointwise_1_w_tmp[k+SEPCONV2D_1_NF*l];
      }
    }

    // Reading the weights of the dense layer
    sprintf(subdirectory,"fold_%d/npyparams/dense_w.npy", fold);
    GetFlatArrFromNpy(model_path+subdirectory, dense_1_w_tmp, dense_1_w_shape);

    // Reshape it
    for(int l=0; l<SEPCONV2D_1_NF*int(FS/DS)*T/AVGPOOL_1_K_1/AVGPOOL_2_K_1; l++){
      for(int k=0; k<N_CLASSES; k++){
        dense_1_w[l][k] = dense_1_w_tmp[k+N_CLASSES*l];
      }
    }

    // Reading the bias params of the dense layer (No reshape needed, 1D array)
    sprintf(subdirectory,"fold_%d/npyparams/dense_b.npy", fold);
    GetFlatArrFromNpy(model_path+subdirectory, dense_1_b_tmp, dense_1_b_shape);

    for(int i=0; i<N_CLASSES; i++) dense_1_b[i] = dense_1_b_tmp[i];


  /*----------------------------------------------------------------------------
  ----------------------------TESTING THE NEURAL NETWORK -----------------------
  ----------------------------------------------------------------------------*/

    // File to save the model outputs
    #ifdef FIXED
      sprintf(subdirectory, "%sfold_%d/validationDS/y_hls_%d_%d.txt", model_path.c_str(), fold, total_apfixed, int_apfixed);
    #endif
    #ifdef FLOAT
      sprintf(subdirectory, "%sfold_%d/validationDS/y_hls_float.txt", model_path.c_str(), fold);
    #endif
    FILE *fout = fopen(subdirectory, "w");

    // Iterate over the validation samples of the current fold
    for(int j=0; j<21*int(DS*N_SUBS/N_FOLDS)*N_CLASSES; j++){
      cout << "Fold " << fold+1 << "/" << N_FOLDS << " | Sample " << j+1 << "/" << 21*int(DS*N_SUBS/N_FOLDS)*N_CLASSES << "...";

      // Reading an input element
      sprintf(subdirectory,"fold_%d/validationDS/X_samples/X_%d.npy", fold, j);
      GetFlatArrFromNpy(model_path+subdirectory, X_tmp, X_shape);

      // Reshape it
      for(int j=0; j<CHANS; j++){
        for(int i=0; i<int(FS/DS)*T; i++){
          X[j][i] = X_tmp[i+j*int(FS/DS)*T];
        }
      }

      //Calculate the neural network output
      MIBCI_QCNN(X, conv2d_1_w, CONV_LRELU_ALPHA, depthconv2d_1_w, DEPTH_LRELU_ALPHA, sepdepth_1_w, seppointwise_1_w, SEP_LRELU_ALPHA, dense_1_w, dense_1_b, Y);

      // Save it
      fprintf(fout, "%d %f %f %f %f\n", argmax(Y), float(Y[0]), float(Y[1]), float(Y[2]), float(Y[3]));

      cout << "OK" << endl;
    }
    fclose(fout);
  }
  return 0;
}
