#include "vecAdd.h"

extern "C" __global__ void vecAdd(int *a, int *b, int* c, void* p){
	int* flag = (int*)p;
	
	int count=0;
/*	while(count!=100){
		count++;
		clock_t start = clock();
		clock_t now;
		for (;;){
			now = clock();
			clock_t cycles = now > start? now - start: now+(0xffffffff - start);
			if (cycles >= 100000000){
				break;
			}
		}
	}
*/
	int tid = blockIdx.x;
	if (tid < N){
		c[tid]=a[tid]+b[tid];
	}

	*flag = 0;
	printf("after computation flag = %d\n", *flag);
	

}
