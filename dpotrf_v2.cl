#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * Cholesky decomposition (version 2)
 * 
 * Call with:
 *    matrix size (m) = 32x32 
 *    group size      = 32x16
 *    grid size       = 32x32
 *
 * Compared to v1, we halve the number of work-items so that old NVidia cards can execute it.
 * Consequently, work-items do twice as much work-load as in v1
 *
 * Issues: 
 *   - does not work on old NVidia cards (32x32 = 1024 work-items > 512)
 * 
 */
__kernel void dpotrf(__global double * m) {
   
   __local double s[32*32];
   
   int x = get_local_id(0);
   int y = get_local_id(1);
   int w = 32;

   int off = y*w+x;
   int off2 = off + 16*w;

   s[off] = m[off];
   s[off2] = m[off2];

   int i=0;
   for (i=0; i<16; i++) {

      if (x == i && y == i) s[off] = sqrt(s[off]);

      barrier(CLK_LOCAL_MEM_FENCE);

      if (x == i && y > i) s[off] /= s[i*w+i];

      if (x == i) s[off2] /= s[i*w+i];

      barrier(CLK_LOCAL_MEM_FENCE);

      if (x > i && y > i && x <= y) s[off] -= s[x*w+i] * s[y*w+i];

      if (x > i && x <= (y+16)) s[off2] -= s[x*w+i] * s[(y+16)*w+i];

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   y += 16;
   for (i=16; i<32; i++) {

      if (x == i && y == i) s[y*w+x] = sqrt(s[y*w+x]);

      barrier(CLK_LOCAL_MEM_FENCE);

      if (x == i && y > i) s[y*w+x] /= s[i*w+i];

      barrier(CLK_LOCAL_MEM_FENCE);

      if (x > i && y > i && x <= y) s[y*w+x] -= s[x*w+i] * s[y*w+i];

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   m[off] = s[off]; 
   m[off2] = s[off2]; 
   
}
