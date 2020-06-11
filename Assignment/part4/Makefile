NVCC 	= /opt/cuda/4.1/bin/nvcc
CUDAPATH = /opt/cuda/4.1

NVCCFLAGS = -I$(CUDAPATH)/include #--ptxas-options=-v 
LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

ALL	= heatCUDA
all: $(ALL)

kernels.o: kernels.cu 
	$(NVCC) -c -g $(NVCCFLAGS) $+ $(LFLAGS) -o $@

heatCUDA: kernels.o heatCUDA.cu
	$(NVCC) -g -O2 $(NVCCFLAGS) $+ $(LFLAGS) -o $@

heatCUDAcpu:  kernels.o heatCUDA.cu
	$(NVCC) -g -O2 $(NVCCFLAGS) $+ $(LFLAGS) -D CPU_REDUCTION -o $@

	
clean:
	rm -fr $(ALL) *.o *.prv *.pcf *.row *.sym *.mpits set-0

