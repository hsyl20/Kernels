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

   double my[8];
   double res = 0.0;

   for (int k=0; k<w/16; k++) {

      a[off] = aBlock[k*16 + Y*w + x];
      b[off] = bBlock[k*16 + X*w + y];

      barrier(CLK_LOCAL_MEM_FENCE);

      #define red(u) a[u+y*16] * b[u+x*16]
      my[0] = red(0) + red(8);
      my[1] = red(1) + red(9);
      my[2] = red(2) + red(10);
      my[3] = red(3) + red(11);
      my[4] = red(4) + red(12);
      my[5] = red(5) + red(13);
      my[6] = red(6) + red(14);
      my[7] = red(7) + red(15);

      my[0] += my[4];
      my[1] += my[5];
      my[2] += my[6];
      my[3] += my[7];

      my[0] += my[2];
      my[1] += my[3];

      res += my[0] + my[1];

/*      for (int r=0; r<16; r++) {
         res += a[r + y*16] * b[r + x*16];
      }*/

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   currBlock[curr_off] -= res;

}

