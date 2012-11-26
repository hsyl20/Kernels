#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>

// Buffer size (max 512 because of dtrsm_block, must be divisible by 16)
#define N 64
// Buffer count (whole matrix size = N*BCOUNT ^ 2)
#define BCOUNT 5
double epsilon = 10e-8;

#define min(a,b) ( a < b ? a : b)

int performCholesky(double * mat[BCOUNT][BCOUNT], cl_ulong n, cl_int nb_dev, cl_device_id * devs, double epsilon, int * errCount, double * maxDiff, cl_ulong * duration, char ** log);
void benchDev(double * mat[BCOUNT][BCOUNT], cl_int nb_dev, cl_device_id * devs);

#pragma weak clGetExtensionFunctionAddressForPlatform
extern void * clGetExtensionFunctionAddressForPlatform(cl_platform_id, const char *);

#pragma weak clGetExtensionFunctionAddress
extern void * clGetExtensionFunctionAddress(const char *);

/* L is the reference matrix. We compute A = L*Lt to then perform
 * cholesky factorization on A (and we should find L back)*/

#define L(x,y) (100.0 / ((double)(x+y)+100.0))

int main() {

   int x, y, z, X, Y;

   double * mat[BCOUNT][BCOUNT];

   for (Y = 0; Y<BCOUNT; Y++) {
      for (X = 0; X<=Y; X++) {
         mat[Y][X] = malloc(N * N * sizeof(double));
      }
   }

   /* compute matN = L*Lt */
   printf("Computing input matrix (size = %d x %d, %d x %d blocks)...\n", N*BCOUNT, N*BCOUNT, BCOUNT, BCOUNT);
   for (y=0; y<N*BCOUNT; y++) {
      for (x=0; x<=y; x++) {
         int X = x/N;
         int Y = y/N;
         int y2 = y % N;
         int x2 = x % N;
         mat[Y][X][y2*N+x2] = 0.0;
         for (z=0; z <= min(x,y); z++) {
            mat[Y][X][y2*N+x2] += L(z,y) * L(z,x);
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
         benchDev(mat, 1, &devs[d]);
      }

      if (strstr(plat_name, "SOCL") != NULL) {

         benchDev(mat, nb_devs, devs);

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

void benchDev(double * mat[BCOUNT][BCOUNT], cl_int nb_dev, cl_device_id * devs) {

   if (nb_dev == 1) {
      cl_device_id dev = devs[0];
      size_t dev_name_size;
      clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &dev_name_size);
      char dev_name[dev_name_size];
      clGetDeviceInfo(dev, CL_DEVICE_NAME, dev_name_size, dev_name, NULL);

      printf("  - Benchmarking device %s:\n", dev_name);
   }
   else {
      printf("  - Benchmarking SOCL scheduler\n");
   }

   int errCount;
   cl_ulong duration;
   char * log;
   double maxDiff;

   int err = performCholesky(mat, N, nb_dev, devs, epsilon, &errCount, &maxDiff, &duration, &log);

   if (err != CL_SUCCESS) {
      printf("      - Error %d: %s\n", err, log);
   }
   else {
      printf("      - Execution time: %.f ms and %s",
            duration/1000.0, (errCount == 0 ? "succeeded" : "failed"));
      if (errCount > 0) {
         printf(" (%d errors, max diff %e, epsilon %e).\n", errCount, maxDiff, epsilon);
      }
      else printf(" (epsilon %e)\n", epsilon);
   }
   printf("\n");
}

cl_int loadKernel(char * kernelFile, char * kernelName, cl_context ctx, cl_int nb_dev, cl_device_id * devs, char **log, cl_kernel * kernel) {
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

   int d;
   for (d = 0; d<nb_dev; d++) {
      cl_device_id dev = devs[d];

      err = clBuildProgram(prg, 1, &dev, NULL, NULL, NULL);

      if (err != CL_SUCCESS) {
         size_t log_size;
         clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
         *log = malloc(log_size);
         clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, log_size, *log, NULL);
         return err;
      }
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

int performCholesky(double * mat[BCOUNT][BCOUNT], cl_ulong n, cl_int nb_dev, cl_device_id * devs, double epsilon, int * errCount, double * maxDiff, cl_ulong * duration, char ** log) {

   int x, y, X, Y;
   cl_int err;

   size_t size = n * n * sizeof(double);

   double * matR[BCOUNT][BCOUNT];
   for (Y=0; Y<BCOUNT; Y++) {
      for(X=0; X<=Y; X++) {
         matR[Y][X] = malloc(size);
         memset(matR[Y][X], 0, size);
      }
   }

   cl_context ctx = clCreateContext(NULL, nb_dev, devs, NULL, NULL, &err);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to create context");
      return err;
   }

   cl_device_id dev;

   // Compile for every device
   cl_kernel dpotrf, dtrsm, dgemm, dtrsm_block, dgemm_block;
   err = loadKernel("dpotrf.cl", "dpotrf", ctx, nb_dev, devs, log, &dpotrf);
   if (err != CL_SUCCESS) {
      return err;
   }
   err = loadKernel("dtrsm.cl", "dtrsm", ctx, nb_dev, devs, log, &dtrsm);
   if (err != CL_SUCCESS) {
      return err;
   }
   err = loadKernel("dgemm.cl", "dgemm", ctx, nb_dev, devs, log, &dgemm);
   if (err != CL_SUCCESS) {
      return err;
   }
   err = loadKernel("dgemm_block.cl", "dgemm_block", ctx, nb_dev, devs, log, &dgemm_block);
   if (err != CL_SUCCESS) {
      return err;
   }
   err = loadKernel("dtrsm_block.cl", "dtrsm_block", ctx, nb_dev, devs, log, &dtrsm_block);
   if (err != CL_SUCCESS) {
      return err;
   }

   // If more than one device, we are using SOCL to perform scheduling (=> dev = NULL)
   dev = (nb_dev == 1 ? devs[0] : NULL);

   cl_command_queue cq = clCreateCommandQueue(ctx, dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to create command queue");
      return err;
   }

   cl_mem buf[BCOUNT][BCOUNT];
   cl_event events[BCOUNT][BCOUNT];

   for (Y=0; Y<BCOUNT; Y++) {
      for (X=0; X<=Y; X++) {


         buf[Y][X] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to allocate buffer");
            return err;
         }


         err = clEnqueueWriteBuffer(cq, buf[Y][X], 0, 0, size, mat[Y][X], 0, NULL, &events[Y][X]);
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

   int step;

   for (step=0; step<BCOUNT; step++) {

      /******************** Diagonal block ***********************/

      err = clSetKernelArg(dpotrf, 0, sizeof(cl_mem), &buf[step][step]);
      err |= clSetKernelArg(dpotrf, 1, sizeof(cl_ulong), &n);
      err |= clSetKernelArg(dtrsm, 0, sizeof(cl_mem), &buf[step][step]);
      err |= clSetKernelArg(dtrsm, 1, sizeof(cl_ulong), &n);
      err |= clSetKernelArg(dgemm, 0, sizeof(cl_mem), &buf[step][step]);
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

         err = clEnqueueNDRangeKernel(cq, dpotrf, 2, NULL, dpotrf_global, dpotrf_local, 1, &events[step][step], &ev);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to enqueue kernel execution command");
            return err;
         }
         clReleaseEvent(events[step][step]);
         events[step][step] = ev;

         size_t r = n - (i+1)*16;

         if (r > 0) {

            size_t dtrsm_global[] = {16,r,1};
            size_t dtrsm_local[] = {16,16,1};
            err = clEnqueueNDRangeKernel(cq, dtrsm, 2, NULL, dtrsm_global, dtrsm_local, 1, &events[step][step], &ev);
            if (err != CL_SUCCESS) {
               *log = strdup("Unable to enqueue kernel execution command");
               return err;
            }
            clReleaseEvent(events[step][step]);
            events[step][step] = ev;

            size_t dgemm_global[] = {r, r,1};
            size_t dgemm_local[] = {16,16,1};
            err = clEnqueueNDRangeKernel(cq, dgemm, 2, NULL, dgemm_global, dgemm_local, 1, &events[step][step], &ev);
            if (err != CL_SUCCESS) {
               *log = strdup("Unable to enqueue kernel execution command");
               return err;
            }
            clReleaseEvent(events[step][step]);
            events[step][step] = ev;
         }
      }

      /*********** SUB-DIAGONAL BLOCKS *******************/
      err = clSetKernelArg(dtrsm_block, 0, sizeof(cl_mem), &buf[step][step]);
      if (err != CL_SUCCESS) {
         *log = strdup("Unable to set kernel parameter");
         return err;
      }

      X = step;
      for (Y=step+1; Y<BCOUNT; Y++) {
         err = clSetKernelArg(dtrsm_block, 1, sizeof(cl_mem), &buf[Y][X]);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to set kernel parameter");
            return err;
         }

         size_t dtrsm_block_global[] = {N,N,1};
         size_t dtrsm_block_local[] = {N,1,1};

         cl_event deps[] = {events[step][step], events[Y][X]};
         err = clEnqueueNDRangeKernel(cq, dtrsm_block, 2, NULL, dtrsm_block_global, dtrsm_block_local, 2, deps, &ev);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to enqueue kernel execution command");
            return err;
         }
         clReleaseEvent(events[Y][X]);
         events[Y][X] = ev;
      }


      /*********** OTHER BLOCKS *******************/
      for (Y=step+1; Y<BCOUNT; Y++) {
         for (X=step+1; X<=Y; X++) {
            err = clSetKernelArg(dgemm_block, 0, sizeof(cl_mem), &buf[Y][step]);
            err |= clSetKernelArg(dgemm_block, 1, sizeof(cl_mem), &buf[X][step]);
            err |= clSetKernelArg(dgemm_block, 2, sizeof(cl_mem), &buf[Y][X]);
            if (err != CL_SUCCESS) {
               *log = strdup("Unable to set kernel parameter");
               return err;
            }

            size_t dgemm_block_global[] = {N,N,1};
            size_t dgemm_block_local[] = {16,16,1};

            cl_event deps[] = {events[Y][step], events[X][step], events[Y][X]};
            err = clEnqueueNDRangeKernel(cq, dgemm_block, 2, NULL, dgemm_block_global, dgemm_block_local, 3, deps, &ev);
            if (err != CL_SUCCESS) {
               *log = strdup("Unable to enqueue kernel execution command");
               return err;
            }
            clReleaseEvent(events[Y][X]);
            events[Y][X] = ev;
         }
      }

   }

   clFinish(cq);

   clock_gettime(CLOCK_MONOTONIC, &end);


   for (Y=0; Y<BCOUNT; Y++) {
      for (X=0; X<=Y; X++) {

         err = clEnqueueReadBuffer(cq, buf[Y][X], 0, 0, size, matR[Y][X], 1, &events[Y][X], &ev);
         if (err != CL_SUCCESS) {
            *log = strdup("Unable to enqueue read buffer command");
            return err;
         }
         clReleaseEvent(events[Y][X]);
         events[Y][X] = ev;
      }
   }

   clFinish(cq);

   *duration = end.tv_nsec - start.tv_nsec + (end.tv_sec-start.tv_sec) * 10e9;

   for (Y=0; Y<BCOUNT; Y++) {
      for (X=0; X<=Y; X++) {
         clReleaseMemObject(buf[Y][X]);
         clReleaseEvent(events[Y][X]);
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
         printf("%.3f ", L(x,y));
      }
      printf("\n");
   }
   printf("\n");

   for (y=0; y<n*BCOUNT; y++) {
      for (x=0; x<=y; x++) {
         X = x/N;
         Y = y/N;
         int y2 = y % N;
         int x2 = x % N;
         printf("%.3f ", matR[Y][X][y2*n+x2]);
      }
      printf("\n");
   }*/

   // Check result
   *errCount = 0;
   *maxDiff = 0.0;

   for (y=0; y<n*BCOUNT; y++) {
      for (x=0; x<=y; x++) {
         X = x/N;
         Y = y/N;
         int y2 = y % N;
         int x2 = x % N;
         double diff = fabs(matR[Y][X][y2*n+x2]-L(x,y));
         if (diff > epsilon) {
            *errCount += 1;
            if (diff > *maxDiff) *maxDiff = diff;
         }
      }
   }

   return 0;
}
