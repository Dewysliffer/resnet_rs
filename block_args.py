# 将ResNet50的 3463调整为3393
BLOCK_ARGS = {
    50: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 3},
        {"input_filters": 256, "num_repeats": 9},
        {"input_filters": 512, "num_repeats": 3},
    ],
    101: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 4},
        {"input_filters": 256, "num_repeats": 23},
        {"input_filters": 512, "num_repeats": 3},
    ],
}
