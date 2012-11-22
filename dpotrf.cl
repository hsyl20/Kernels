/**
 * Cholesky decomposition (version 1.0)
 * 
 * Call with:
 *    matrix size (m) = 32x32 
 *    group size      = 32x32
 *    grid size       = 32x32
 * 
 */
__kernel void dpotrf(__global double * m) {
   
   __local double s[32*32];
   
   int x = get_local_id(0);
   int y = get_local_id(1);
   int w = get_local_size(0);

   int off = y*w+x;

   s[off] = m[off];

   int i=0;
   for (i=0; i<w; i++) {

      if (x == i && y == i) s[off] = sqrt(s[off]);

      barrier(CLK_LOCAL_MEM_FENCE);

      if (x == i && y > i) s[off] /= s[i*w+i];

      barrier(CLK_LOCAL_MEM_FENCE);

      if (x > i && y > i && x <= y) s[off] -= s[x*w+i] * s[y*w+i];

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   m[off] = s[off]; 
   
}
