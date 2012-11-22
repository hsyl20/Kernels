#include <stdio.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>

#define min(a,b) ( a < b ? a : b)
#define N 32

int performCholesky(double * matA, double * matRef, char * kernelFile, size_t * groupSize, cl_device_id dev, int * errCount, cl_ulong * duration, char ** log);

#pragma weak clGetExtensionFunctionAddressForPlatform
extern void * clGetExtensionFunctionAddressForPlatform(cl_platform_id, const char *);

#pragma weak clGetExtensionFunctionAddress
extern void * clGetExtensionFunctionAddress(const char *);

int main() {

   int x, y, z;

   double * matA = malloc(N * N * sizeof(double));

   /* L is the reference matrix. We compute A = L*Lt to then perform
    * cholesky factorization on A (and we should find L back)*/

   double * matL = malloc(N * N * sizeof(double));
   for (y=0; y<N; y++) {
      for (x=0; x<=y; x++) {
         matL[y*N+x] = 1.0 / ((double)(x+y)+5.0);
      }
   }

   /* compute A = L*Lt */
   for (y=0; y<N; y++) {
      for (x=0; x<=y; x++) {
         matA[y*N+x] = 0.0;
         for (z=0; z<=min(x,y); z++) {
            matA[y*N+x] += matL[y*N+z] * matL[x*N+z];
         }
      }
   }

   char * kernelFiles[] = {
      "dpotrf_v1.cl",
      "dpotrf_v2.cl"
   };

   size_t kernelGroupSize[][3] = {
      {32, 32, 1},
      {32, 16, 1}
   };

   int nb_ker = sizeof(kernelFiles) / sizeof(char*);

   printf("Benchmarking %d kernels:\n", nb_ker);
   int k;
   for (k=0; k<nb_ker; k++) {
      printf("  - %s: group size %dx%dx%d\n", kernelFiles[k], (int)kernelGroupSize[k][0], (int)kernelGroupSize[k][1], (int)kernelGroupSize[k][2]);
   }
   printf("\n");

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

         int k;
         for (k=0; k<nb_ker; k++) {

            int err = performCholesky(matA, matL, kernelFiles[k], (size_t*)&kernelGroupSize[k], devs[d], &errCount, &duration, &log);

            if (err != CL_SUCCESS) {
               printf("      - Error %d with kernel %s. Build log: %s\n", err, kernelFiles[k], log);
            }
            else {
               printf("      - kernel %s took %.f ms and %s (%d errors).\n", kernelFiles[k], duration/1000.0, (errCount == 0 ? "succeeded" : "failed"), errCount);
            }
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

int performCholesky(double * matA, double * matRef, char * kernelFile, size_t * kernelGroupSize, cl_device_id dev, int * errCount, cl_ulong * duration, char ** log) {

   cl_event ev_writeA, ev_ker, ev_readA;
   int x, y;
   cl_int err;

   double * matB = malloc(N * N * sizeof(double));
   memset(matB, 0, N * N * sizeof(double));

   FILE * f = fopen(kernelFile, "r");
   if (f == NULL) return 1;

   fseek(f, 0, SEEK_END);
   size_t source_size = ftell(f);
   fseek(f, 0, SEEK_SET);

   char * source = malloc(source_size+1);
   fread(source, 1, source_size, f);
   source[source_size] = '\0';
   fclose(f);

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

   cl_program prg = clCreateProgramWithSource(ctx, 1, (const char**)&source, NULL, NULL);
   err = clBuildProgram(prg, 1, &dev, NULL, NULL, NULL);

   if (err != CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      *log = malloc(log_size);
      clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, log_size, *log, NULL);
      return err;
   }

   
   cl_kernel ker = clCreateKernel(prg, "dpotrf", &err);
   if (err != CL_SUCCESS) {
      char buffer[4096];
      sprintf(buffer, "Unable to create kernel with name \"%s\" from file %s", "dpotrf", kernelFile);
      *log = strdup(buffer);
      return err;
   }

   cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * N * sizeof(double), NULL, &err);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to allocate buffer");
      return err;
   }

   err = clEnqueueWriteBuffer(cq, bufA, 0, 0, N * N * sizeof(double), matA, 0, NULL, &ev_writeA);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to enqueue write buffer command");
      return err;
   }

   err = clSetKernelArg(ker, 0, sizeof(bufA), &bufA);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to set kernel parameter");
      return err;
   }

   size_t globalDim[] = {N, N};
   cl_event deps[] = {ev_writeA};
   err = clEnqueueNDRangeKernel(cq, ker, 2, NULL, globalDim, kernelGroupSize, 1, deps, &ev_ker);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to enqueue kernel execution command");
      return err;
   }

   err = clEnqueueReadBuffer(cq, bufA, 0, 0, N*N*sizeof(double), matB, 1, &ev_ker, &ev_readA);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to enqueue read buffer command");
      return err;
   }

   clFinish(cq);

   cl_ulong start, end;
   err = clGetEventProfilingInfo(ev_ker, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to get event profiling info (start time)");
      return err;
   }

   err = clGetEventProfilingInfo(ev_ker, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
   if (err != CL_SUCCESS) {
      *log = strdup("Unable to get event profiling info (end time)");
      return err;
   }

   *duration = end-start;

   clReleaseEvent(ev_readA);
   clReleaseEvent(ev_writeA);
   clReleaseEvent(ev_ker);
   clReleaseMemObject(bufA);
   clReleaseKernel(ker);
   clReleaseProgram(prg);
   clReleaseCommandQueue(cq);
   clReleaseContext(ctx);

   // Check result
   *errCount = 0;
   for (y=0; y<N; y++) {
      for (x=0; x<=y; x++) {
         if (fabs(matB[y*N+x]-matRef[y*N+x]) > 10e-9) {
            *errCount += 1;
         }
      }
   }

   return 0;
}
