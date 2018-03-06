#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "vecAdd.h"
#include <pthread.h>


// for pthread usage
struct threadParam {
	CUresult* resultAddr;
};

void* checkResult(void* param){
	struct threadParam* p = (struct threadParam*)param;
	CUresult* flagAddr = p->resultAddr;
	while (*flagAddr!=CUDA_SUCCESS);
	printf("successful!!\n");
	printf("CUDA_success code = %d\n", (int)(*flagAddr));
}



// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

CUdevice device;
CUcontext context;
CUmodule module;
CUfunction function;
size_t totalGlobalMem;

char* module_file = (char*) "vecAdd.ptx";
char* kernel_name = (char*) "vecAdd";

inline void __checkCudaErrors(CUresult err, const char* file, const int line){
	if (CUDA_SUCCESS !=err ){
		fprintf (stderr, "CUDA Driver API err = %04d from file <%s>, line %i.\n", err, file, line);
		exit(-1);
	}
}


void initCUDA(){
	int deviceCount = 0;
	CUresult err = cuInit(0);
	int major = 0, minor = 0;

	if (err == CUDA_SUCCESS) checkCudaErrors(cuDeviceGetCount(&deviceCount));

	if (deviceCount==0) {
		fprintf(stderr, "error: no devices supporting CUDA\n");
		exit(-1);
	}

	// get first CUDA device
	checkCudaErrors(cuDeviceGet(&device, 0));
	char name[100];
	cuDeviceGetName(name, 100, device);
	printf("> Using device 0: %s\n", name);

	// get compute capabilities and the device name
	checkCudaErrors (cuDeviceComputeCapability(&major, &minor, device));
	printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

	checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, device));
	printf(" Total amout of gloabal memory: %llu bytes\n", (unsigned long long)totalGlobalMem);
	printf(" 64-bit Memory Address: %s\n", (totalGlobalMem > (unsigned long long) 4*1024*1-24*1024L)?"YES":"NO");

	err = cuCtxCreate(&context, 0, device);
	if (err != CUDA_SUCCESS){
		fprintf(stderr, "* Error initializing the CUDA context.\n");
		cuCtxDetach(context);
		exit(-1);
	}

	err = cuModuleLoad(&module, module_file);
	if (err!=CUDA_SUCCESS){
		fprintf(stderr, "* Error loading the module %s\n", module_file);
		cuCtxDetach(context);
		exit(-1);
	}

	err = cuModuleGetFunction(&function, module, kernel_name);

	if (err!=CUDA_SUCCESS){
		fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
		cuCtxDetach(context);
		exit(-1);
	}
}

void finalizeCUDA(){
	cuCtxDetach(context);
}


void setupDeviceMemory(CUdeviceptr *d_a, CUdeviceptr *d_b, CUdeviceptr *d_c){
	checkCudaErrors (cuMemAlloc(d_a, sizeof(int)*N));
	checkCudaErrors (cuMemAlloc(d_b, sizeof(int)*N));
	checkCudaErrors (cuMemAlloc(d_c, sizeof(int)*N));
}

void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c){
	checkCudaErrors(cuMemFree(d_a));
	checkCudaErrors(cuMemFree(d_b));
	checkCudaErrors(cuMemFree(d_c));
}

void runKernel (CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c){
	void *args[3] = {&d_a, &d_b, &d_c};

	checkCudaErrors(cuLaunchKernel(function,N,1,1,
									1,1,1,
									0,0,args,0));
}

int main(int argc, char **argv){
	int a[N], b[N], c[N];
	CUdeviceptr d_a, d_b, d_c;


	// initialize another thread to check whether the result is correct
	pthread_t checkCUresult;
	struct threadParam param;
	CUresult result = CUDA_ERROR_NOT_INITIALIZED;
	param.resultAddr = &result;

	pthread_create(&checkCUresult, NULL, checkResult, &param);

	// initialize host arrays
	for (int i=0; i<N; i++){
		a[i]=N-i;
		b[i]=i*i;
	}

	initCUDA();
	setupDeviceMemory(&d_a, &d_b, &d_c);

	checkCudaErrors(cuMemcpyHtoD(d_a, a, sizeof(int)*N));
	checkCudaErrors(cuMemcpyHtoD(d_b, b, sizeof(int)*N));
	
//	runKernel(d_a, d_b, d_c);
	void *args[3] = {&d_a, &d_b, &d_c};

	checkCudaErrors(cuLaunchKernel(function,N,1,1,1,1,1,0,0,args,0));
	result = cuCtxSynchronize();

	checkCudaErrors(cuMemcpyDtoH(c, d_c, sizeof(int)*N));

	for (int i=0; i<N; i++){
		
	}

	pthread_join(checkCUresult, NULL);
	releaseDeviceMemory(d_a, d_b, d_c);
	finalizeCUDA();

	return 0;

}
