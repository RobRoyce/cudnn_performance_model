NVCC=nvcc
ARCH?=sm_70

CUDA_PATH?=/usr/local/cuda-11.3
CUDA_LIB64=$(CUDA_PATH)/lib64
CUDNN_PATH?=$(CUDA_PATH)/include
NCCL_PATH?=/usr/local/nccl
MPI_PATH?=/usr/local/openmpi
MPI_INCLUDE_PATH?=$(MPI_PATH)/include
BIN_DIR?=bin
MKDIR=mkdir -p
#BLAS
BLAS_LIBRARY?=cublas
BLAS_PATH?=$(CUDA_LIB64)
#CONV
CONV_LIBRARY?=cudnn
CONV_PATH?=$
KERNELS_DIR=../kernels/
PAD_KERNELS?=1
USE_TENSOR_CORES?=0
COMMA=,
NVCC_ARCH_ARGS=$(foreach a,$(subst $(COMMA), ,$(ARCH)),--generate-code arch=$(patsubst sm_%,compute_%,$(a)),code=$(a))

.PHONY=all gemm conv #rnn all_reduce nccl_single nccl_mpi clean

all: gemm conv

gemm:
	$(MKDIR) $(BIN_DIR) 
	$(CUDA_PATH)/bin/$(NVCC) gemm_bench.cu -DUSE_TENSOR_CORES=$(USE_TENSOR_CORES) -DPAD_KERNELS=$(PAD_KERNELS) -o $(BIN_DIR)/gemm_bench -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -L $(BLAS_PATH) -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand $(NVCC_ARCH_ARGS) -std=c++11

conv:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) conv_bench.cu -DUSE_TENSOR_CORES=$(USE_TENSOR_CORES) -DPAD_KERNELS=$(PAD_KERNELS) -o $(BIN_DIR)/conv_bench -I $(KERNELS_DIR) -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcurand -lcudnn $(NVCC_ARCH_ARGS) -std=c++11


clean:
	rm -rf $(BIN_DIR)

rebuild: clean all
