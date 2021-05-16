import numpy as np
import pandas as pd

def concat_conv():
    conv_1 = pd.read_csv('V100_conv_inference_1', delim_whitespace=True)
    conv_2 = pd.read_csv('V100_conv_inference_2', delim_whitespace=True)
    conv_3 = pd.read_csv('V100_conv_inference_3', delim_whitespace=True)
    conv_1['pad_kernels'] = np.zeros(conv_1.shape[0])
    conv_2['pad_kernels'] = np.zeros(conv_2.shape[0])
    df = pd.concat([conv_1, conv_2, conv_3])
    df.to_csv('./V100_conv.csv', index=False)


def concat_gemm():
    gemm_1 = pd.read_csv('V100_gemm_inference_1', delim_whitespace=True)
    gemm_2 = pd.read_csv('V100_gemm_inference_2', delim_whitespace=True)
    df = pd.concat([gemm_1, gemm_2])
    df.to_csv('./V100_gemm.csv', index=False)

def concat_archs():
    conv_1070 = pd.read_csv('1070_conv.csv')
    conv_2070 = pd.read_csv('2070_conv.csv')
    conv_V100 = pd.read_csv('V100_conv.csv')
    gemm_2070 = pd.read_csv('2070_gemm.csv')
    gemm_V100 = pd.read_csv('V100_gemm.csv')

    df = pd.concat([conv_1070, conv_2070, conv_V100])
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('conv.csv', index=False)

    df = pd.concat([gemm_2070, gemm_V100])
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('gemm.csv', index=False)


if __name__ == '__main__':
    # concat_conv()
    # concat_gemm()

    concat_archs()