# Placeholder for internal Python library.

# TensorFlow Recommenders-Addons, a TensorFlow library for building recommender systems.
licenses(["notice"])  # Apache 2.0

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

py_library(
    name = "tensorflow_recommenders_addons",
    srcs = glob(["**/*.py"]),
    deps = [
        "//tensorflow_recommenders_addons/dynamic_embedding/python/ops:dynamic_embedding",
        "//tensorflow_recommenders_addons/utils",
    ],
)

py_test(
    name = "tensorflow_recommenders_addons_test",
    size = "small",
    srcs = glob(["**/*test.py"]),
    main = "tests/run_all_test.py",
    deps = [
        ":tensorflow_recommenders_addons",
    ],
)
