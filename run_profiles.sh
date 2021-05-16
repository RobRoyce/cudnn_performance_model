DEEPBENCH_BIN=/usr/eda/cs259/robroyce/DeepBench/code/nvidia/bin
CUDNN_CONV_PARAMS=/usr/eda/cs259/robroyce/cudnn_performance_model/config/cudnn_conv_params.csv
CUDNN_GEMM_PARAMS=/usr/eda/cs259/robroyce/cudnn_performance_model/config/cudnn_gemm_params.csv

cd $DEEPBENCH_BIN

echo "gemm_bench inference float"
./gemm_bench inference float $CUDNN_GEMM_PARAMS > gemm_inference_float

echo "gemm_bench inference half"
./gemm_bench inference half $CUDNN_GEMM_PARAMS > gemm_inference_half

echo "gemm_bench inference int8"
./gemm_bench inference int8 $CUDNN_GEMM_PARAMS > gemm_inference_int8

echo "conv_bench inference float"
./conv_bench inference float $CUDNN_CONV_PARAMS > conv_inference_float

echo "conv_bench inference half"
./conv_bench inference half $CUDNN_CONV_PARAMS > conv_inference_half

echo "conv_bench inference int8"
./conv_bench inference int8 $CUDNN_CONV_PARAMS > conv_inference_int8