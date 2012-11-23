#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * Update other blocks (version 1.0)
 * 
 * Parameters: 
 *  - m : matrix
 *  - n : matrix width
 *  - step : iteration (in step of 16 columns)
 *
 * Call with:
 *  - global : (n-step*16) x (n-step*16)
 !  - local : 16 x 16
 * 
 */
__kernel void dgemm(__global double * m, long n, long step) {
   
   int x = get_local_id(0);
   int y = get_local_id(1);
   int X = get_global_id(0);
   int Y = get_global_id(1);
   int k = get_group_id(1);

   int off = y*16+x;                // local offset
   int diag_off = step*16*(n+1);       // global diagonal block offset
   int a_off = diag_off + Y*n;       // sub-diagonal block 1 offset 
   int b_off = diag_off + X*n;       // sub-diagonal block 2 offset
   int curr_off = diag_off + Y*n + X;   // global current block offset

   __local double a[16*16];
   __local double b[16*16];
   __local double curr[16*16];
   curr[off] = m[curr_off];
   a[off] = m[a_off];
   b[off] = m[b_off];

   barrier(CLK_LOCAL_MEM_FENCE);

   for (int i=0; i<16; i++) {
      curr[off] -= a[i+y*n] * b[i+x*n];
   }

   m[curr_off] = curr[off]; 
   
}

