#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * Update other blocks per big block
 * 
 * Parameters: 
 *  - aBlock : sub-diagonal block for y
 *  - bBlock : sub-diagonal block for x
 *  - currBlock : current block
 *
 * Call with:
 *  - global : n x n
 !  - local : 16 x 16
 * 
 */
__kernel void dgemm_block(__global double * aBlock, __global double * bBlock, __global double * currBlock) {
   
   int x = get_local_id(0);
   int y = get_local_id(1);
   int w = get_global_size(0);
   int X = get_global_id(0);
   int Y = get_global_id(1);

   int off = y*16+x;
   int curr_off = Y*w+X;

   __local double a[16*16];
   __local double b[16*16];

   double my = 0.0;

   for (int k=0; k<w/16; k++) {

      a[off] = aBlock[k*16 + Y*w + x];
      b[off] = bBlock[k*16 + X*w + y];

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int r=0; r<16; r++) {
         my += a[r + y*16] * b[r + x*16];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   currBlock[curr_off] -= my;

}

