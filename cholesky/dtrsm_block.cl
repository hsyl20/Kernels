#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * Update sub-diagonal blocks per line
 * 
 * Parameters: 
 *  - diagBlock : diagonal block
 *  - currBlock : current sub-diagonal block
 *
 * Call with:
 *  - global : n x n
 *  - local : n x 1     (n <= 512)
 * 
 */
__kernel void dtrsm_block(__global double * diagBlock, __global double * currBlock) {
   
   int X = get_global_id(0);
   int Y = get_global_id(1);

   int w = get_global_size(0);

   __local double diag[512];
   __local double curr[512];

   curr[X] = currBlock[X + Y*w];

   double my = 0.0;

   for (int i=0; i<w; i++) {

      if (X >= i) diag[X] = diagBlock[i + X*w];
   
      barrier(CLK_LOCAL_MEM_FENCE);

      if (X == i) {
         curr[i] = (curr[i] - my) / diag[i];
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      if (X > i) my += curr[i] * diag[X];

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   currBlock[X+Y*w] = curr[X]; 
   
}

