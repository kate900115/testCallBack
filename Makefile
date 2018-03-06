all: program

program: 
	nvcc -ptx vecAdd.cu
	g++ -I /usr/local/cuda-9.0/targets/x86_64-linux/include main.cpp -o main -lcuda -lpthread

clean:
	rm -rf main vecAdd.ptx
