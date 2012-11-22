#include <stdio.h>
#include <math.h>
#include <CL/cl.h>

#define min(a,b) ( a < b ? a : b)
#define N 32

int performCholesky(double * matA, double * matRef, char * kernelFile, cl_device_id dev, int * valid, cl_ulong * duration, char ** log);

int main() {

   int x, y, z;

   double * matA = malloc(N * N * sizeof(double));

   // init L
   //intf("L:\n");
   double * matL = malloc(N * N * sizeof(double));
   for (y=0; y<N; y++) {
      for (x=0; x<=y; x++) {
         matL[y*N+x] = x+y+1.0;
         //printf("%.2f ", matL[y*N + x]);
      }
      //printf("\n");
   }

   // compute A = L*Lt
   //printf("A:\n");
   for (y=0; y<N; y++) {
      for (x=0; x<=y; x++) {
         matA[y*N+x] = 0.0;
         for (z=0; z<=min(x,y); z++) {
            matA[y*N+x] += matL[y*N+z] * matL[x*N+z];
         }
        // printf("%.2f ", matA[y*N + x]);
      }
      //printf("\n");
   }

   cl_uint nb_platf;
   clGetPlatformIDs(0, NULL, &nb_platf);

   printf("Found %d platform%s\n", nb_platf, nb_platf > 1 ? "s" : "");

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
      printf("\nBenchmark on platform: %s (%s)\n", plat_name, plat_vendor);

      cl_uint nb_devs;
      clGetDeviceIDs(platfs[p], CL_DEVICE_TYPE_ALL, 0, NULL, &nb_devs);
      printf("Found %d device%s\n", nb_devs, nb_devs > 1 ? "s" : "");

      cl_device_id devs[nb_devs];
      clGetDeviceIDs(platfs[p], CL_DEVICE_TYPE_ALL, nb_devs, devs, NULL);

      cl_uint d;
      for (d=0; d<nb_devs; d++) {
         size_t dev_name_size;
         clGetDeviceInfo(devs[d], CL_DEVICE_NAME, 0, NULL, &dev_name_size);
         char dev_name[dev_name_size];
         clGetDeviceInfo(devs[d], CL_DEVICE_NAME, dev_name_size, dev_name, NULL);

         int valid;
         cl_ulong duration;
         char * k = "./dpotrf.cl";
         char * log;

         int err = performCholesky(matA, matL, k, devs[d], &valid, &duration, &log);

         if (err != 0) {
            printf("Error %d on %s\n", err, dev_name);
            printf("Build log: %s\n", log);
         }
         else {
            printf("Execution of kernel %s on %s took %.f ms and %s.\n", k, dev_name, duration/1000.0, (valid ? "succeeded" : "failed"));
         }
      }
   }

   printf("\nDone.\n");

   return 0;
}

int performCholesky(double * matA, double * matRef, char * kernelFile, cl_device_id dev, int * valid, cl_ulong * duration, char ** log) {

   cl_event ev_writeA, ev_ker, ev_readA;
   int x, y;

   double * matB = malloc(N * N * sizeof(double));

   FILE * f = fopen(kernelFile, "r");
   if (f == NULL) return 1;

   fseek(f, 0, SEEK_END);
   size_t source_size = ftell(f);
   fseek(f, 0, SEEK_SET);

   char * source = malloc(source_size+1);
   fread(source, 1, source_size, f);
   source[source_size] = '\0';
   fclose(f);

   cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
   cl_command_queue cq = clCreateCommandQueue(ctx, dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, NULL);

   cl_program prg = clCreateProgramWithSource(ctx, 1, (const char**)&source, NULL, NULL);
   clBuildProgram(prg, 1, &dev, NULL, NULL, NULL);

   if (log != NULL) {
      size_t log_size;
      clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      *log = malloc(log_size);
      clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, log_size, *log, NULL);
   }

   cl_kernel ker = clCreateKernel(prg, "dpotrf", NULL);

   cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * N * sizeof(double), NULL, NULL);

   clEnqueueWriteBuffer(cq, bufA, 0, 0, N * N * sizeof(double), matA, 0, NULL, &ev_writeA);

   clSetKernelArg(ker, 0, sizeof(bufA), &bufA);

   size_t globalDim[] = {N, N};
   size_t localDim[] = {32, 32};
   cl_event deps[] = {ev_writeA};
   clEnqueueNDRangeKernel(cq, ker, 2, NULL, globalDim, localDim, 1, deps, &ev_ker);

   clEnqueueReadBuffer(cq, bufA, 0, 0, N*N*sizeof(double), matB, 1, &ev_ker, &ev_readA);

   clFinish(cq);

   cl_ulong start, end;
   clGetEventProfilingInfo(ev_ker, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
   clGetEventProfilingInfo(ev_ker, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

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
   *valid = 1;
   for (y=0; y<N; y++) {
      for (x=0; x<=y; x++) {
         if (fabs(matB[y*N+x]-matRef[y*N+x]) > 0.01) {
            *valid = 0;
            return 0;
         }
      }
   }

   return 0;
}
