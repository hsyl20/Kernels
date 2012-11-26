#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>

// Buffer size
#define N 32
// Buffer count (whole matrix size = N*BCOUNT ^ 2)
#define BCOUNT 4

#define min(a,b) ( a < b ? a : b)

int performCholesky(double * mat[BCOUNT][BCOUNT], cl_ulong n, cl_device_id dev, int * errCount, cl_ulong * duration, char ** log);

#pragma weak clGetExtensionFunctionAddressForPlatform
extern void * clGetExtensionFunctionAddressForPlatform(cl_platform_id, const char *);

#pragma weak clGetExtensionFunctionAddress
extern void * clGetExtensionFunctionAddress(const char *);

/* L is the reference matrix. We compute A = L*Lt to then perform
 * cholesky factorization on A (and we should find L back)*/

#define L(x,y) (100.0 / ((double)(x+y)+100.0))

int main() {

   int x, y, z, b, c;

   double * mat[BCOUNT][BCOUNT];

   for (b = 0; b<BCOUNT; b++) {
      for (c = 0; c<BCOUNT; c++) {
         mat[b][c] = malloc(N * N * sizeof(double));
      }
   }

   /* compute matN = L*Lt */
   printf("Computing input matrix (size = %d x %d, %d x %d blocks)...\n", N*BCOUNT, N*BCOUNT, BCOUNT, BCOUNT);
   for (y=0; y<N*BCOUNT; y++) {
      for (x=0; x<=y; x++) {
         int b = x/N;
         int c = y/N;
         int y2 = y % N;
         int x2 = x % N;
         mat[b][c][y2*N+x2] = 0.0;
         for (z=0; z <= min(x,y); z++) {
            mat[b][c][y2*N+x2] += L(z,y) * L(z,x);
         }
      }
   }

   cl_uint nb_platf;
   clGetPlatformIDs(0, NULL, &nb_platf);

   printf("%d OpenCL platform%s found\n", nb_platf, nb_platf > 1 ? "s" : "");

   cl_platform_id platfs[nb_platf];
   clGetPlatformIDs(nb_platf, platfs, NULL);

   cl_uint p;
   for (p=0; p<nb_platf; p++) {

      size_t plat_name_size;
      clGetPlatformInfo(platfs[p], CL_PLATFORM_NAME, 0, NULL, &plat_name_size);
      char plat_name[plat_name_size];
      clGetPlatformInfo(platfs[p], CL_PLATFORM_NAME, plat_name_size, &plat_name, NULL);

      size_t plat_vendor_size;
      clGetPlatformInfo(platfs[p], CL_PLATFORM_VENDOR, 0, NULL, &plat_vendor_size);
      char plat_vendor[plat_vendor_size];
      clGetPlatformInfo(platfs[p], CL_PLATFORM_VENDOR, plat_vendor_size, &plat_vendor, NULL);

      cl_uint nb_devs;
      clGetDeviceIDs(platfs[p], CL_DEVICE_TYPE_ALL, 0, NULL, &nb_devs);
      printf("\nBenchmarking platform: %s (%s) - %d device%s\n\n", plat_name, plat_vendor, nb_devs, nb_devs > 1 ? "s" : "");

      cl_device_id devs[nb_devs];
      clGetDeviceIDs(platfs[p], CL_DEVICE_TYPE_ALL, nb_devs, devs, NULL);

      cl_uint d;
      for (d=0; d<nb_devs; d++) {
         size_t dev_name_size;
         clGetDeviceInfo(devs[d], CL_DEVICE_NAME, 0, NULL, &dev_name_size);
         char dev_name[dev_name_size];
         clGetDeviceInfo(devs[d], CL_DEVICE_NAME, dev_name_size, dev_name, NULL);

         printf("  - Benchmarking device %s:\n", dev_name);

         int errCount;
         cl_ulong duration;
         char * log;

         int err = performCholesky(mat, N, devs[d], &errCount, &duration, &log);

         if (err != CL_SUCCESS) {
            printf("      - Error %d: %s\n", err, log);
         }
         else {
            printf("      - Execution time: %.f ms and %s (%d errors).\n", 
               duration/1000.0, (errCount == 0 ? "succeeded" : "failed"), errCount);
         }
         printf("\n");
      }

      
      if (strstr(plat_name, "SOCL") != NULL) {
         
         void (*clShutdown)(void) = (clGetExtensionFunctionAddressForPlatform != NULL ?
                                     clGetExtensionFunctionAddressForPlatform(platfs[p], "clShutdown") :
                                    (clGetExtensionFunctionAddress != NULL ?
                                     clGetExtensionFunctionAddress("clShutdown"):
                                     NULL));

         if (clShutdown != NULL) {
            clShutdown();
         }
      }
   }

   printf("\nDone.\n");



   return 0;
}

cl_int loadKernel(char * kernelFile, char * kernelName, cl_context ctx, cl_device_id dev, char **log, cl_kernel * kernel) {
   cl_int err;

   FILE * f = fopen(kernelFile, "r");
   if (f == NULL) return 1;

   fseek(f, 0, SEEK_END);
   size_t source_size = ftell(f);
   fseek(f, 0, SEEK_SET);

   char * source = malloc(source_size+1);
   fread(source, 1, source_size, f);
   source[source_size] = '\0';
   fclose(f);

   cl_program prg = clCreateProgramWithSource(ctx, 1, (const char**)&source, NULL, NULL);
   err = clBuildProgram(prg, 1, &dev, NULL, NULL, NULL);

   if (err != CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      *log = malloc(log_size);
      clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, log_size, *log, NULL);
      return err;
   }

   
   *kernel = clCreateKernel(prg, kernelName, &err);
   if (err != CL_SUCCESS) {
      char buffer[4096];
      sprintf(buffer, "Unable to create kernel with name \"%s\" from file %s", kernelName, kernelFile);
      *log = strdup(buffer);
      return err;
   }

   clReleaseProgram(prg);

   return CL_SUCCESS;
}

int performCholesky(double * mat[BCOUNT][BCOUNT], cl_ulong n, cl_device_id dev, int * errCount, cl_ulong * duration, char ** log) {

   int x, y, b, c, d;
   cl_int err;

   size_t size = n * n * sizeof(double);

   double * matR[BCOUNT][BCOUNT];
   for (b=0; b<BCOUNT; b++) {
      for(c=0; c<BCOUNT; c++) {
         matR[b][c] = malloc(size);
         memset(matR[b][c], 0, size);
      }
   }


   cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to create context");
      return err;
   }

   cl_command_queue cq = clCreateCommandQueue(ctx, dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to create command queue");
      return err;
   }

   cl_kernel dpotrf, dtrsm, dgemm, dtrsm_block, dgemm_block;
   err = loadKernel("dpotrf.cl", "dpotrf", ctx, dev, log, &dpotrf);
   if (err != CL_SUCCESS) {
      return err;
   }
   err = loadKernel("dtrsm.cl", "dtrsm", ctx, dev, log, &dtrsm);
   if (err != CL_SUCCESS) {
      return err;
   }
   err = loadKernel("dgemm.cl", "dgemm", ctx, dev, log, &dgemm);
   if (err != CL_SUCCESS) {
      return err;
   }
   err = loadKernel("dgemm_block.cl", "dgemm_block", ctx, dev, log, &dgemm_block);
   if (err != CL_SUCCESS) {
      return err;
   }
   err = loadKernel("dtrsm_block.cl", "dtrsm_block", ctx, dev, log, &dtrsm_block);
   if (err != CL_SUCCESS) {
      return err;
   }

   cl_mem buf[BCOUNT][BCOUNT];
   cl_event events[BCOUNT][BCOUNT];

   for (b=0; b<BCOUNT; b++) {
      for (c=0; c<BCOUNT; c++) {


         buf[b][c] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to allocate buffer");
            return err;
         }


         err = clEnqueueWriteBuffer(cq, buf[b][c], 0, 0, size, mat[b][c], 0, NULL, &events[b][c]);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to enqueue write buffer command");
            return err;
         }
      }
   }

   clFinish(cq);

   struct timespec start, end;
   clock_gettime(CLOCK_MONOTONIC, &start);

   cl_event ev;

   for (b=0; b<BCOUNT; b++) {

      /******************** Diagonal block ***********************/

      err = clSetKernelArg(dpotrf, 0, sizeof(cl_mem), &buf[b][b]);
      err |= clSetKernelArg(dpotrf, 1, sizeof(cl_ulong), &n);
      err |= clSetKernelArg(dtrsm, 0, sizeof(cl_mem), &buf[b][b]);
      err |= clSetKernelArg(dtrsm, 1, sizeof(cl_ulong), &n);
      err |= clSetKernelArg(dgemm, 0, sizeof(cl_mem), &buf[b][b]);
      err |= clSetKernelArg(dgemm, 1, sizeof(cl_ulong), &n);
      if (err != CL_SUCCESS) {
         *log = strdup("Unable to set kernel parameter");
         return err;
      }

      cl_long i;
      for (i=0; i<n/16; i++) {

         err = clSetKernelArg(dpotrf, 2, sizeof(cl_ulong), &i);
         err |= clSetKernelArg(dgemm, 2, sizeof(cl_ulong), &i);
         err |= clSetKernelArg(dtrsm, 2, sizeof(cl_ulong), &i);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to set kernel parameter");
            return err;
         }

         size_t dpotrf_global[] = {16,16,1};
         size_t dpotrf_local[] = {16,16,1};

         err = clEnqueueNDRangeKernel(cq, dpotrf, 2, NULL, dpotrf_global, dpotrf_local, 1, &events[b][b], &ev);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to enqueue kernel execution command");
            return err;
         }
         clReleaseEvent(events[b][b]);
         events[b][b] = ev;

         size_t r = n - (i+1)*16;

         if (r > 0) {

            size_t dtrsm_global[] = {16,r,1};
            size_t dtrsm_local[] = {16,16,1};
            err = clEnqueueNDRangeKernel(cq, dtrsm, 2, NULL, dtrsm_global, dtrsm_local, 1, &events[b][b], &ev);
            if (err != CL_SUCCESS) {
               *log = strdup("Unable to enqueue kernel execution command");
               return err;
            }
            clReleaseEvent(events[b][b]);
            events[b][b] = ev;

            size_t dgemm_global[] = {r, r,1};
            size_t dgemm_local[] = {16,16,1};
            err = clEnqueueNDRangeKernel(cq, dgemm, 2, NULL, dgemm_global, dgemm_local, 1, &events[b][b], &ev);
            if (err != CL_SUCCESS) {
               *log = strdup("Unable to enqueue kernel execution command");
               return err;
            }
            clReleaseEvent(events[b][b]);
            events[b][b] = ev;
         }
      }

      /*********** SUB-DIAGONAL BLOCKS *******************/
      err = clSetKernelArg(dtrsm_block, 0, sizeof(cl_mem), &buf[b][b]);
      if (err != CL_SUCCESS) {
         *log = strdup("Unable to set kernel parameter");
         return err;
      }

      for (c=b+1; c<BCOUNT; c++) {
         err = clSetKernelArg(dtrsm_block, 1, sizeof(cl_mem), &buf[b][c]);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to set kernel parameter");
            return err;
         }

         size_t dtrsm_block_global[] = {N,N,1};
         size_t dtrsm_block_local[] = {N,1,1};

         cl_event deps[] = {events[b][b], events[b][c]};
         err = clEnqueueNDRangeKernel(cq, dtrsm_block, 2, NULL, dtrsm_block_global, dtrsm_block_local, 2, deps, &ev);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to enqueue kernel execution command");
            return err;
         }
         clReleaseEvent(events[b][c]);
         events[b][c] = ev;
      }


      /*********** OTHER BLOCKS *******************/
      for (c=b+1; c<BCOUNT; c++) {
         for (d=b+1; d<=c; d++) {
            err = clSetKernelArg(dgemm_block, 0, sizeof(cl_mem), &buf[b][c]);
            err |= clSetKernelArg(dgemm_block, 1, sizeof(cl_mem), &buf[b][d]);
            err |= clSetKernelArg(dgemm_block, 2, sizeof(cl_mem), &buf[d][c]);
            if (err != CL_SUCCESS) {
               *log = strdup("Unable to set kernel parameter");
               return err;
            }

            size_t dgemm_block_global[] = {N,N,1};
            size_t dgemm_block_local[] = {16,16,1};

            cl_event deps[] = {events[c][c], events[b][d], events[c][d]};
            err = clEnqueueNDRangeKernel(cq, dgemm_block, 2, NULL, dgemm_block_global, dgemm_block_local, 3, deps, &ev);
            if (err != CL_SUCCESS) {
               *log = strdup("Unable to enqueue kernel execution command");
               return err;
            }
            clReleaseEvent(events[c][d]);
            events[c][d] = ev;
         }
      }

   }

   clFinish(cq);

   clock_gettime(CLOCK_MONOTONIC, &end);


   for (b=0; b<BCOUNT; b++) {
      for (c=0; c<BCOUNT; c++) {

         err = clEnqueueReadBuffer(cq, buf[b][c], 0, 0, size, matR[b][c], 1, &events[b][c], &ev);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to enqueue read buffer command");
            return err;
         }
         clReleaseEvent(events[b][c]);
         events[b][c] = ev;
      }
   }

   clFinish(cq);

   *duration = end.tv_nsec - start.tv_nsec + (end.tv_sec-start.tv_sec) * 10e9;

   for (b=0; b<BCOUNT; b++) {
      for (c=0; c<BCOUNT; c++) {
         clReleaseMemObject(buf[b][c]);
         clReleaseEvent(events[b][c]);
      }
   }
   clReleaseKernel(dpotrf);
   clReleaseKernel(dtrsm);
   clReleaseKernel(dgemm);
   clReleaseKernel(dtrsm_block);
   clReleaseKernel(dgemm_block);
   clReleaseCommandQueue(cq);
   clReleaseContext(ctx);


/*   for (y=0; y<n*BCOUNT; y++) {
      for (x=0; x<=y; x++) {
         printf("%.2f ", L(x,y));
      }
      printf("\n");
   }
   printf("\n");

   for (y=0; y<n*BCOUNT; y++) {
      for (x=0; x<=y; x++) {
         int b = x/N;
         int c = y/N;
         int y2 = y % N;
         int x2 = x % N;
         printf("%.2f ", matR[b][c][y2*n+x2]);
      }
      printf("\n");
   }*/

   // Check result
   *errCount = 0;
   for (y=0; y<n*BCOUNT; y++) {
      for (x=0; x<=y; x++) {
         int b = x/N;
         int c = y/N;
         int y2 = y % N;
         int x2 = x % N;
         if (fabs(matR[b][c][y2*n+x2]-L(x,y)) > 10e-9) {
            *errCount += 1;
         }
      }
   }

   return 0;
}
