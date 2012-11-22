all:
	gcc -Wall test.c -lOpenCL -lm -pthread -g -o test
