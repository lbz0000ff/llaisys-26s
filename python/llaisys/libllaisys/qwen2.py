from ctypes import POINTER, Structure, c_char_p, c_float, c_int, c_int64, c_size_t, c_void_p
from enum import IntEnum

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t


llaisysQwen2Model_t = c_void_p


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class Qwen2WeightLoadStatus(IntEnum):
    SUCCESS = 0
    INVALID_ARGUMENT = 1
    UNKNOWN_NAME = 2
    DTYPE_MISMATCH = 3
    SHAPE_MISMATCH = 4
    DUPLICATE = 5


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelLoadWeight.argtypes = [
        llaisysQwen2Model_t,
        c_char_p,
        c_void_p,
        llaisysDataType_t,
        POINTER(c_size_t),
        c_size_t,
    ]
    lib.llaisysQwen2ModelLoadWeight.restype = c_int

    lib.llaisysQwen2ModelExpectedWeightCount.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelExpectedWeightCount.restype = c_size_t

    lib.llaisysQwen2ModelLoadedWeightCount.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelLoadedWeightCount.restype = c_size_t

    lib.llaisysQwen2ModelCachedTokenCount.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelCachedTokenCount.restype = c_size_t

    lib.llaisysQwen2ModelReset.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelReset.restype = None

    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64
