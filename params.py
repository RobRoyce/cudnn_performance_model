#include csv
import csv


def generate_deepbench_convolution(
        width,
        height,
        channels,
        batch,
        filter_cubes,
        filter_width,
        filter_height,
        padw,
        padh,
        stridew,
        strideh
):
    # w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride

    with open('config/conv_problems.h', 'w') as file:
        file.write('''std::vector <std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,unsigned int, unsigned int, unsigned int,unsigned int, unsigned int, unsigned int, unsigned int>> training_set = {std::make_tuple(14, 14, 128, 8, 256, 3, 3, 1, 1, 1, 1),};\nstd::vector <std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,unsigned int, unsigned int, unsigned int,unsigned int, unsigned int, unsigned int, unsigned int>> inference_server_set = {\n''')
        for w, h in zip(width, height):
            for c in channels:
                for n in batch:
                    for k in filter_cubes:
                        for s in filter_width:
                            for r in filter_height:
                                for pw in padw:
                                    for ph in padh:
                                        for sw in stridew:
                                            for sh in strideh:
                                                file.write(f'std::make_tuple({w}, {h}, {c}, {n}, {k}, {s}, {r}, {pw}, {ph}, {sw}, {sh}),\n')
        file.write('std::make_tuple(1,1,1,1,1,1,1,1,1,1,1)};\n')
        file.write('''std::vector <std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,unsigned int, unsigned int, unsigned int,unsigned int, unsigned int, unsigned int, unsigned int>> inference_device_set = {std::make_tuple(224, 224, 64, 1, 64, 3, 3, 1, 1, 1, 1)};''')


def generate_deepbench_gemm():
    pass


if __name__ == '__main__':
    width = [16 << i for i in range(7)]
    height = [16 << i for i in range(7)]
    channels = [1 << i for i in range(9)]
    batch = [1 << i for i in range(5)]
    filter_cubes = [1 << i for i in range(7)]
    filter_width = [2, 3, 4, 5]
    filter_height = [2, 3, 4, 5]
    padw = [1]
    padh = [1]
    stridew = [1]
    strideh = [1]

    generate_deepbench_convolution(
        width,
        height,
        channels,
        batch,
        filter_cubes,
        filter_width,
        filter_height,
        padw,
        padh,
        stridew,
        strideh
    )
