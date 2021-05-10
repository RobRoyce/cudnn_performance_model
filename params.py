import csv


def generate_deepbench_convolution(
        width, height, channels, batch,
        filter_cubes, filter_width, padw, padh,
        stridew, strideh):
    with open('config/cudnn_conv_params.csv', 'w') as file:
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
    with open('config/cudnn_gemm_params.csv', 'w') as file:
        for i in inputs:
            for b in batch:
                for o in outputs:
                    for a in a_t:
                        for aa in b_t:
                            file.write(f'{i},{b},{o},{a},{aa}\n')


if __name__ == '__main__':
    # Generate Conv Parameters
    width = [16 << i for i in range(8)]
    height = [16 << i for i in range(8)]
    channels = [1 << i for i in range(7)]
    batch = [1 << i for i in range(6)]
    filter_cubes = [1 << i for i in range(7)]
    filter_width = [3, 4, 5]
    padw = [1, 2]
    padh = [1, 2]
    stridew = [1, 2]
    strideh = [1, 2]
    generate_deepbench_convolution(
        width, height, channels, batch,
        filter_cubes, filter_width, padw, padh,
        stridew, strideh)

    # Generate GEMM Parameters
    inputs = [16 << i for i in range(19)]
    batch = [1 << i for i in range(6)]
    outputs = [1 << i for i in range(19)]
    a_t = [0, 1]
    generate_deepbench_gemm(inputs, batch, outputs, a_t, a_t)
