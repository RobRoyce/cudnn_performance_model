#include csv
import csv


def generate_deepbench_convolution(
        w, h, c, n, k, s, r, padw, padh, stridew, strideh
):
    # w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
    print(f'std::make_tuple({w}, {h}, {c}, {n}, {k}, {s}, {r}, {padw}, {padh}, {stridew}, {strideh}),')


def generate_deepbench_gemm():
    pass


if __name__ == '__main__':
    width = []
    height = []

    with open('config/input.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            width.append(row[0])
            height.append(row[1])

    generate_deepbench_convolution()


