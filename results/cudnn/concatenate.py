import numpy as np
import pandas as pd

def concat_conv():
    conv_1 = pd.read_csv('2070_conv_inference_1', delim_whitespace=True)
    conv_2 = pd.read_csv('2070_conv_inference_2', delim_whitespace=True)
    conv_3 = pd.read_csv('2070_conv_inference_3', delim_whitespace=True)
    conv_4 = pd.read_csv('2070_conv_inference_4', delim_whitespace=True)
    conv_1['pad_kernels'] = np.zeros(conv_1.shape[0])
    conv_2['pad_kernels'] = np.zeros(conv_2.shape[0])
    conv_3['pad_kernels'] = np.zeros(conv_3.shape[0])
    df = pd.concat([conv_1, conv_2, conv_3, conv_4])
    df.to_csv('./2070_conv.csv', index=False)


def concat_gemm():
    gemm_1 = pd.read_csv('2070_gemm_inference_1', delim_whitespace=True)
    gemm_2 = pd.read_csv('2070_gemm_inference_2', delim_whitespace=True)
    gemm_3 = pd.read_csv('2070_gemm_inference_3', delim_whitespace=True)
    df = pd.concat([gemm_1, gemm_2, gemm_3])
    df.to_csv('./2070_gemm.csv', index=False)

if __name__ == '__main__':
    concat_conv()
    concat_gemm()