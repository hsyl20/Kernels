all:
	mkdir -p build
	cp -f cholesky/*.cl build/
	gcc -Wall -lOpenCL -lm -pthread -g -o build/cholesky_single_kernel cholesky/single_kernel.c
	gcc -Wall -lrt -lOpenCL -lm -pthread -g -o build/cholesky_multi_kernel cholesky/multi_kernel.c
