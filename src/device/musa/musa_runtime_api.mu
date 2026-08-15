#include "../runtime_api.hpp"

#include <musa_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::musa {

namespace runtime_api {
namespace {
void checkMusa(musaError_t status, const char *operation) {
    if (status != musaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + musaGetErrorString(status));
    }
}

musaMemcpyKind musaMemcpyKindOf(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return musaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return musaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return musaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return musaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("invalid LLAISYS memcpy kind");
    }
}
} // namespace

int getDeviceCount() {
    int count = 0;
    checkMusa(musaGetDeviceCount(&count), "musaGetDeviceCount");
    return count;
}

void setDevice(int device) {
    checkMusa(musaSetDevice(device), "musaSetDevice");
}

void deviceSynchronize() {
    checkMusa(musaDeviceSynchronize(), "musaDeviceSynchronize");
}

llaisysStream_t createStream() {
    musaStream_t stream = nullptr;
    checkMusa(musaStreamCreate(&stream), "musaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    checkMusa(
        musaStreamDestroy(reinterpret_cast<musaStream_t>(stream)),
        "musaStreamDestroy");
}
void streamSynchronize(llaisysStream_t stream) {
    checkMusa(
        musaStreamSynchronize(reinterpret_cast<musaStream_t>(stream)),
        "musaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkMusa(musaMalloc(&ptr, size), "musaMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    checkMusa(musaFree(ptr), "musaFree");
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    checkMusa(musaMallocHost(&ptr, size), "musaMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    checkMusa(musaFreeHost(ptr), "musaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    checkMusa(musaMemcpy(dst, src, size, musaMemcpyKindOf(kind)), "musaMemcpy");
}

void memcpyAsync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream) {
    checkMusa(
        musaMemcpyAsync(
            dst,
            src,
            size,
            musaMemcpyKindOf(kind),
            reinterpret_cast<musaStream_t>(stream)),
        "musaMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::musa
