#include <stdio.h>
#include <math.h>
#include <CL/cl.h>

#define min(a,b) ( a < b ? a : b)

int main() {

   #define N 32
   size_t size = N * N * sizeof(double);

   cl_event ev_writeA, ev_ker, ev_readA;
   int x, y, z;

   double * matA = malloc(N * N * sizeof(double));

   // init L
   printf("L:\n");
   double * matL = malloc(N * N * sizeof(double));
   for (y=0; y<N; y++) {
      for (x=0; x<=y; x++) {
         matL[y*N+x] = x+y+1.0;
         printf("%.2f ", matL[y*N + x]);
      }
      printf("\n");
   }

   // compute A = L*Lt
   printf("A:\n");
   for (y=0; y<N; y++) {
      for (x=0; x<=y; x++) {
         matA[y*N+x] = 0.0;
         for (z=0; z<=min(x,y); z++) {
            matA[y*N+x] += matL[y*N+z] * matL[x*N+z];
         }
         printf("%.2f ", matA[y*N + x]);
      }
      printf("\n");
   }

   cl_uint nb_platf;
   clGetPlatformIDs(0, NULL, &nb_platf);

   printf("Nombre de plateformes: %d\n", nb_platf);

   cl_platform_id platfs[nb_platf];
   clGetPlatformIDs(nb_platf, platfs, NULL);

   size_t plat_name_size;
   clGetPlatformInfo(platfs[0], CL_PLATFORM_NAME, 0, NULL, &plat_name_size);
   char plat_name[plat_name_size];
   clGetPlatformInfo(platfs[0], CL_PLATFORM_NAME, plat_name_size, &plat_name, NULL);
   printf("Nom de la plateforme: %s\n", plat_name);

   size_t plat_vendor_size;
   clGetPlatformInfo(platfs[0], CL_PLATFORM_VENDOR, 0, NULL, &plat_vendor_size);
   char plat_vendor[plat_vendor_size];
   clGetPlatformInfo(platfs[0], CL_PLATFORM_VENDOR, plat_vendor_size, &plat_vendor, NULL);
   printf("Nom de la plateforme: %s\n", plat_vendor);

   cl_uint nb_devs;
   clGetDeviceIDs(platfs[0], CL_DEVICE_TYPE_ALL, 0, NULL, &nb_devs);

   cl_device_id devs[nb_devs];
   clGetDeviceIDs(platfs[0], CL_DEVICE_TYPE_ALL, nb_devs, devs, NULL);

   cl_context ctx = clCreateContext(NULL, nb_devs, devs, NULL, NULL, NULL);
   cl_command_queue cq = clCreateCommandQueue(ctx, devs[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, NULL);

   FILE * f = fopen("./dpotrf.cl", "r");
   fseek(f, 0, SEEK_END);
   size_t source_size = ftell(f);
   fseek(f, 0, SEEK_SET);

   char * source = malloc(source_size+1);
   fread(source, 1, source_size, f);
   source[source_size] = '\0';
   fclose(f);

   printf("Code kernel: %s\n", source);

   cl_program prg = clCreateProgramWithSource(ctx, 1, (const char**)&source, NULL, NULL);
   clBuildProgram(prg, nb_devs, devs, NULL, NULL, NULL);

   size_t log_size;
   clGetProgramBuildInfo(prg, devs[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
   char log[log_size];
   clGetProgramBuildInfo(prg, devs[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

   printf("Build log: %s\n", log);

   cl_kernel ker = clCreateKernel(prg, "dpotrf", NULL);

   cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, NULL);

   clEnqueueWriteBuffer(cq, bufA, 0, 0, size, matA, 0, NULL, &ev_writeA);

   clSetKernelArg(ker, 0, sizeof(bufA), &bufA);

   size_t globalDim[] = {N, N};
   size_t localDim[] = {32, 32};
   cl_event deps[] = {ev_writeA};
   clEnqueueNDRangeKernel(cq, ker, 2, NULL, globalDim, localDim, 1, deps, &ev_ker);

   clEnqueueReadBuffer(cq, bufA, 0, 0, size, matA, 1, &ev_ker, &ev_readA);

   clFinish(cq);

   printf("Cholesky A: \n");
   for (y=0; y<N; y++) {
      for (x=0; x<=y; x++) {
         printf("%.2f ", matA[y*N + x]);
      }
      printf("\n");
   }

   printf("Fini.\n");

   return 0;
}

