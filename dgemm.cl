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
__kernel void dgemm(__global double * m, unsigned long n, unsigned long step) {
   
   int x = get_local_id(0);
   int y = get_local_id(1);
   int gx = get_group_id(0);
   int gy = get_group_id(1);

   int off = y*16+x;                // local offset
   int diag_off = step*16*(n+1) + y*n + x;       // global diagonal block offset
   int a_off = diag_off + (gy+1)*n*16;       // sub-diagonal block 1 offset 
   int b_off = diag_off + (gx+1)*n*16;       // sub-diagonal block 2 offset
   int curr_off = diag_off + (gy+1)*n*16 + (gx+1)*16;   // global current block offset

   __local double a[16*16];
   __local double b[16*16];
   __local double curr[16*16];

   if (gx <= gy) {
      curr[off] = m[curr_off];
      a[off] = m[a_off];
      b[off] = m[b_off];
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   if (gx <= gy) {
      double my[8];
      #define red(u) a[u+y*16] * b[u+x*16]
/*      my[0] = red(0) + red(8);
      my[1] = red(1) + red(9);
      my[2] = red(2) + red(10);
      my[3] = red(3) + red(11);
      my[4] = red(4) + red(12);
      my[5] = red(5) + red(13);
      my[6] = red(6) + red(14);
      my[7] = red(6) + red(15);

      my[0] += my[4];
      my[1] += my[5];
      my[2] += my[6];
      my[3] += my[7];

      my[0] += my[2];
      my[1] += my[3];

      my[0] += my[1];*/

      my[0] = 0.0;
      for (int i=0; i<16; i++) {
         my[0] += a[i+y*16] * b[i+x*16];
      }

      m[curr_off] = curr[off] - my[0]; 
   }
   
}

