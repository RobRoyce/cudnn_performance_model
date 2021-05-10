// Vector saves w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
std::vector <std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int>> training_set;

// Vector saves w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
std::vector <std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int>> inference_server_set;

// Vector saves w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
std::vector <std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int>> inference_device_set;