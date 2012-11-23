#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * Update sub-diagonal blocks (version 1.0)
 * 
 * Parameters: 
 *  - m : matrix
 *  - n : matrix width
 *  - step : iteration (in step of 16 columns)
 *
 * Call with:
 *  - global : 16 x (n-step*16)
 !  - local : 16 x 16
 * 
 */
__kernel void dtrsm(__global double * m, long n, long step) {
   
   int x = get_local_id(0);
   int y = get_local_id(1);
   int Y = get_global_id(1);
   int k = get_group_id(1);

   int off = y*16+x;                // local offset
   int diag_off = step*16*(n+1);       // global diagonal block offset
   int curr_off = diag_off + Y*n;   // global current block offset

   // Load diagonal block and current block
   __local double diag[16*16];
   __local double curr[16*16];
   diag[off] = m[diag_off];
   curr[off] = m[curr_off];

   barrier(CLK_LOCAL_MEM_FENCE);

   for (int i=0; i<16; i++) {

      double d = diag[i*16+i];

      if (x == i && y > i) curr[off] /= d;

      barrier(CLK_LOCAL_MEM_FENCE);

      if (x > i && y > i && x <= y) curr[off] -= curr[x*16+i] * curr[y*16+i];

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   m[curr_off] = curr[off]; 
   
}

