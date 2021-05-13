import csv
import numpy as np
import pandas as pd

CUDNN_CONV_PARAMS = 'config/cudnn_conv_params.csv'
CUDNN_GEMM_PARAMS = 'config/cudnn_gemm_params.csv'
SAMPLE_SIZE = 8192


def generate_deepbench_convolution(
        width, height, channels, batch,
        filter_cubes, filter_width, padw, padh,
        stridew, strideh):
    with open(CUDNN_CONV_PARAMS, 'w') as file:
        for w, h in zip(width, height):
            for c in channels:
                for n in batch:
                    for k in filter_cubes:
                        for s in filter_width:
                            for pw in padw:
                                for ph in padh:
                                    for sw in stridew:
                                        for sh in strideh:
                                            file.write(f'{w},{h},{c},{n},{k},{s},{s},{pw},{ph},{sw},{sh}\n')


def generate_deepbench_gemm(inputs, batch, outputs, a_t, b_t):
    with open(CUDNN_GEMM_PARAMS, 'w') as file:
        for i in inputs:
            for b in batch:
                for o in outputs:
                    for a in a_t:
                        for aa in b_t:
                            file.write(f'{i},{b},{o},{a},{aa}\n')


def random_subsample():
    df = pd.read_csv(CUDNN_CONV_PARAMS)
    df = df.sample(SAMPLE_SIZE, random_state=42, replace=True)
    df.to_csv(CUDNN_CONV_PARAMS, header=False, index=False, mode='w')

    df = pd.read_csv(CUDNN_GEMM_PARAMS)
    df = df.sample(SAMPLE_SIZE, random_state=1, replace=True)
    df.to_csv(CUDNN_GEMM_PARAMS, header=False, index=False, mode='w')


if __name__ == '__main__':
    # Generate Conv Parameters
    input = pd.read_csv('config/input.csv')
    width = input[input.columns[0]]
    height = input[input.columns[1]]
    channels = [1 << i for i in range(7)]
    batch = [1 << i for i in range(7)]
    filter_cubes = [1 << i for i in range(7)]
    filter_width = [3, 5]
    padw = [0, 1]
    padh = [0, 1]
    stridew = [1, 2, 3, 5]
    strideh = [1, 2, 3, 5]

    generate_deepbench_convolution(width, height, channels, batch,
        filter_cubes, filter_width, padw, padh, stridew, strideh)

    # Generate GEMM Parameters
    inputs = [1 << i for i in range(19)]
    batch = [1 << i for i in range(7)]
    outputs = [1 << i for i in range(9)]
    a_t = [0, 1]

    generate_deepbench_gemm(inputs, batch, outputs, a_t, a_t)
    random_subsample()
