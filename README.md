# cuDNN Performance Model

This project is based on an assignment for UCLA's graduate-level course -- Current Topics in Computer Science: System Design/Architecture: Learning Machines.

The goal is to build a performance model for cuDNN-accelerated kernels which, given a kernel configuration and fixed GPU parameters, can predict the inference time of new configurations. We use this to determine which features are most important, so as to better understand the performance of GPUs and their respective workloads.

We customize [Baidu's DeepBench](https://github.com/baidu-research/DeepBench) to generate 100k sample runs of cuDNN-accelerated convolution and general matrix-multiply (GEMM). The output of DeepBench is [1] the forward pass time (inference), and [2] the algorithm used by DeepBench (WINOGRAD, etc). We use the same set of configurations to get data on three different Nvidia GPUs:

- GTX 1070
- RTX 2070
- Titan V

Using inference time as a target feature, we train various models to predict on previously unseen cuDNN kernel configurations. After finding a suitable model, we perform statistical analysis to determine feature importance. As of the time of this commit, we settled on a random forest ensemble as the model of choice, and we use permutation importance to determine feature significance. See `analysis.ipynb` for methodology and results.
