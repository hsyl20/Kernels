all:
	gcc -Wall -lOpenCL -lm -pthread -g -o test test.c
	gcc -Wall -lrt -lOpenCL -lm -pthread -g -o multi_kernel multi_kernel.c
