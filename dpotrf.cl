#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * Cholesky decomposition (version 1.0)
 * 
 * Call with:
 *    group size      = 16x16
 *    grid size       = 16x16
 *
 * Parameters:
 *  - m : matrix
 *  - n : matrix width
 *  - step : iteration (in block of 16 columns)
 * 
 */
__kernel void dpotrf(__global double * m, long n, long step) {
   
   __local double s[16*16];
   
   int x = get_local_id(0);
   int y = get_local_id(1);

   int off = y*16+x;                // local offset
   int diag_off = step*16*(n+1);       // global diagonal block offset

   // Load diagonal block
   __local double diag[16*16];
   diag[off] = m[diag_off];

   for (int i=0; i<16; i++) {

      if (x == i && y == i) diag[off] = sqrt(diag[off]);

      barrier(CLK_LOCAL_MEM_FENCE);

      if (x == i && y > i) diag[off] /= diag[i*16+i];

      barrier(CLK_LOCAL_MEM_FENCE);

      if (x > i && y > i && x <= y) diag[off] -= diag[x*16+i] * diag[y*16+i];

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   m[diag_off] = diag[off]; 
   
}
