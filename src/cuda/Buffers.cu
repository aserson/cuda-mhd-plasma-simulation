#include "cuda/Buffers.cuh"

#include "cuda/KernelCaller.cuh"

namespace mhd {

// CpuBuffer1D functions definitions
template <typename T>
CpuBuffer1D<T>::CpuBuffer1D()
    : _buffer(nullptr), _bufferLength(0), _bufferSize(0) {}

template <typename T>
CpuBuffer1D<T>::CpuBuffer1D(unsigned int bufferLength)
    : _bufferLength(bufferLength) {
    _bufferSize = _bufferLength * sizeof(T);

    CUDA_CALL(
        cudaHostAlloc((void**)&_buffer, _bufferSize, cudaHostAllocDefault));
}

template <typename T>
CpuBuffer1D<T>::~CpuBuffer1D() {
    CUDA_CALL(cudaFreeHost(_buffer));
}

template <typename T>
T* CpuBuffer1D<T>::data() {
    return _buffer;
}

template <typename T>
const T* CpuBuffer1D<T>::data() const {
    return _buffer;
}

template <typename T>
T& CpuBuffer1D<T>::operator[](unsigned int index) {
    return _buffer[index];
}

template <typename T>
const T& CpuBuffer1D<T>::operator[](unsigned int index) const {
    return _buffer[index];
}

template <typename T>
unsigned int CpuBuffer1D<T>::size() const {
    return _bufferSize;
}

template <typename T>
unsigned int CpuBuffer1D<T>::length() const {
    return _bufferLength;
}

template <typename T>
void CpuBuffer1D<T>::clear() {
    memset(_buffer, 0x0, _bufferSize);
}

template <typename T>
void CpuBuffer1D<T>::copyToDevice(T* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

template <typename T>
void CpuBuffer1D<T>::copyFromDevice(const T* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
}

// CpuBuffer2D functions definitions
template <typename T>
CpuBuffer2D<T>::CpuBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0) {}

template <typename T>
CpuBuffer2D<T>::CpuBuffer2D(unsigned int sideLength) : _sideLength(sideLength) {
    _bufferSize = _sideLength * _sideLength * sizeof(T);

    CUDA_CALL(
        cudaHostAlloc((void**)&_buffer, _bufferSize, cudaHostAllocDefault));
}

template <typename T>
CpuBuffer2D<T>::~CpuBuffer2D() {
    CUDA_CALL(cudaFreeHost(_buffer));
}

template <typename T>
T* CpuBuffer2D<T>::data() {
    return _buffer;
}

template <typename T>
const T* CpuBuffer2D<T>::data() const {
    return _buffer;
}

template <typename T>
T& CpuBuffer2D<T>::operator[](unsigned int index) {
    return _buffer[index];
}

template <typename T>
const T& CpuBuffer2D<T>::operator[](unsigned int index) const {
    return _buffer[index];
}

template <typename T>
unsigned int CpuBuffer2D<T>::size() const {
    return _bufferSize;
}

template <typename T>
unsigned int CpuBuffer2D<T>::length() const {
    return _sideLength;
}

template <typename T>
void CpuBuffer2D<T>::clear() {
    memset(_buffer, 0x0, _bufferSize);
}

template <typename T>
void CpuBuffer2D<T>::copyToDevice(T* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

template <typename T>
void CpuBuffer2D<T>::copyFromDevice(const T* src) {
    if (src != nullptr) {
        CUDA_CALL(
            cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
    } else {
        std::cerr << "Copying Error from device: Source buffer is nullptr!"
                  << std::endl;
    }
}

// GpuBuffer2D functions definitions
template <typename T>
GpuBuffer2D<T>::GpuBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0) {}

template <typename T>
GpuBuffer2D<T>::GpuBuffer2D(unsigned int sideLength) : _sideLength(sideLength) {
    _bufferSize = _sideLength * _sideLength * sizeof(T);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

template <typename T>
GpuBuffer2D<T>::~GpuBuffer2D() {
    CUDA_CALL(cudaFree(_buffer));
}

template <typename T>
T* GpuBuffer2D<T>::data() {
    return _buffer;
}

template <typename T>
const T* GpuBuffer2D<T>::data() const {
    return _buffer;
}

template <typename T>
unsigned int GpuBuffer2D<T>::size() const {
    return _bufferSize;
}

template <typename T>
unsigned int GpuBuffer2D<T>::length() const {
    return _sideLength;
}

template <typename T>
void GpuBuffer2D<T>::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

template <typename T>
void GpuBuffer2D<T>::copyToHost(T* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost));
}

template <typename T>
void GpuBuffer2D<T>::copyFromHost(const T* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyHostToDevice));
}

template <typename T>
void GpuBuffer2D<T>::copyToDevice(T* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToDevice));
}

template <typename T>
void GpuBuffer2D<T>::copyFromDevice(const T* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToDevice));
}

// GpuComplexBuffer2D functions definitions
template <typename T>
GpuComplexBuffer2D<T>::GpuComplexBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0) {}

template <typename T>
GpuComplexBuffer2D<T>::GpuComplexBuffer2D(unsigned int sideLength)
    : _sideLength(sideLength) {
    _bufferSize = (_sideLength / 2 + 1) * _sideLength * sizeof(Complex);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

template <typename T>
GpuComplexBuffer2D<T>::~GpuComplexBuffer2D() {
    CUDA_CALL(cudaFree(_buffer));
}

template <typename T>
typename GpuComplexBuffer2D<T>::Complex* GpuComplexBuffer2D<T>::data() {
    return _buffer;
}

template <typename T>
const typename GpuComplexBuffer2D<T>::Complex* GpuComplexBuffer2D<T>::data()
    const {
    return _buffer;
}

template <typename T>
unsigned int GpuComplexBuffer2D<T>::size() const {
    return _bufferSize;
}

template <typename T>
unsigned int GpuComplexBuffer2D<T>::length() const {
    return _sideLength;
}

template <typename T>
void GpuComplexBuffer2D<T>::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

template <typename T>
void GpuComplexBuffer2D<T>::copyToHost(Complex* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost));
}

template <typename T>
void GpuComplexBuffer2D<T>::copyFromHost(const Complex* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyHostToDevice));
}

template <typename T>
void GpuComplexBuffer2D<T>::copyToDevice(Complex* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToDevice));
}

template <typename T>
void GpuComplexBuffer2D<T>::copyFromDevice(const Complex* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToDevice));
}

// The solver runs in either double or float precision
template class CpuBuffer1D<double>;
template class CpuBuffer1D<float>;
template class CpuBuffer2D<double>;
template class CpuBuffer2D<float>;
template class GpuBuffer2D<double>;
template class GpuBuffer2D<float>;
// Single-element counter for the forcing phase RNG
template class GpuBuffer2D<unsigned int>;
template class GpuComplexBuffer2D<double>;
template class GpuComplexBuffer2D<float>;

// GpuStateBuffer2D functions definitions

GpuStateBuffer2D::GpuStateBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0) {}

GpuStateBuffer2D::GpuStateBuffer2D(unsigned int sideLength)
    : _sideLength(sideLength) {
    _bufferSize = (_sideLength / 2 + 1) * _sideLength * sizeof(curandState);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

GpuStateBuffer2D::~GpuStateBuffer2D() {
    CUDA_CALL(cudaFree(_buffer));
}

curandState* GpuStateBuffer2D::data() {
    return _buffer;
}
const curandState* GpuStateBuffer2D::data() const {
    return _buffer;
}
unsigned int GpuStateBuffer2D::size() const {
    return _bufferSize;
}
unsigned int GpuStateBuffer2D::length() const {
    return _sideLength;
}
}  // namespace mhd

namespace graphics {

// CpuFloatBuffer functions definitions

CpuFloatBuffer::CpuFloatBuffer()
    : _buffer(nullptr), _bufferLength(0), _bufferSize(0) {}

CpuFloatBuffer::CpuFloatBuffer(unsigned int bufferLength)
    : _bufferLength(bufferLength) {
    _bufferSize = _bufferLength * sizeof(float);

    CUDA_CALL(
        cudaHostAlloc((void**)&_buffer, _bufferSize, cudaHostAllocDefault));
}

CpuFloatBuffer::~CpuFloatBuffer() {
    CUDA_CALL(cudaFreeHost(_buffer));
}

float* CpuFloatBuffer::data() {
    return _buffer;
}

const float* CpuFloatBuffer::data() const {
    return _buffer;
}

float& CpuFloatBuffer::operator[](unsigned int index) {
    return _buffer[index];
}

const float& CpuFloatBuffer::operator[](unsigned int index) const {
    return _buffer[index];
}

unsigned int CpuFloatBuffer::size() const {
    return _bufferSize;
}

unsigned int CpuFloatBuffer::length() const {
    return _bufferLength;
}

void CpuFloatBuffer::clear() {
    memset(_buffer, 0x0, _bufferSize);
}

void CpuFloatBuffer::copyToDevice(float* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

void CpuFloatBuffer::copyFromDevice(const float* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
}

// GpuFloatBuffer functions definitions

GpuFloatBuffer::GpuFloatBuffer()
    : _buffer(nullptr), _bufferLength(0), _bufferSize(0) {}

GpuFloatBuffer::GpuFloatBuffer(unsigned int bufferLength)
    : _bufferLength(bufferLength) {
    _bufferSize = _bufferLength * sizeof(float);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

GpuFloatBuffer::~GpuFloatBuffer() {
    cudaFree(_buffer);
}

float* GpuFloatBuffer::data() {
    return _buffer;
}

const float* GpuFloatBuffer::data() const {
    return _buffer;
}

unsigned int GpuFloatBuffer::size() const {
    return _bufferSize;
}

unsigned int GpuFloatBuffer::length() const {
    return _bufferLength;
}

void GpuFloatBuffer::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void GpuFloatBuffer::copyToDevice(float* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

void GpuFloatBuffer::copyFromDevice(const float* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
}

// CpuPixelBuffer2D functions definitions

CpuPixelBuffer2D::CpuPixelBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0), _channels(0) {}

CpuPixelBuffer2D::CpuPixelBuffer2D(unsigned int sideLength,
                                   unsigned int channels)
    : _sideLength(sideLength), _channels(channels) {
    _bufferSize = _sideLength * _sideLength * _channels * sizeof(unsigned char);

    CUDA_CALL(
        cudaHostAlloc((void**)&_buffer, _bufferSize, cudaHostAllocDefault));
}

CpuPixelBuffer2D::~CpuPixelBuffer2D() {
    CUDA_CALL(cudaFreeHost(_buffer));
}

unsigned char* CpuPixelBuffer2D::data() {
    return _buffer;
}

const unsigned char* CpuPixelBuffer2D::data() const {
    return _buffer;
}

unsigned int CpuPixelBuffer2D::size() const {
    return _bufferSize;
}

unsigned int CpuPixelBuffer2D::length() const {
    return _sideLength;
}

void CpuPixelBuffer2D::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void CpuPixelBuffer2D::copyToDevice(unsigned char* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyHostToDevice));
}

void CpuPixelBuffer2D::copyFromDevice(const unsigned char* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyDeviceToHost));
}

// GPUPixelBuffer2D functions definitions

GpuPixelBuffer2D::GpuPixelBuffer2D()
    : _buffer(nullptr), _sideLength(0), _bufferSize(0), _channels(0) {}

GpuPixelBuffer2D::GpuPixelBuffer2D(unsigned int sideLength,
                                   unsigned int channels)
    : _sideLength(sideLength), _channels(channels) {
    _bufferSize = _sideLength * _sideLength * _channels * sizeof(unsigned char);

    CUDA_CALL(cudaMalloc((void**)&_buffer, _bufferSize));
}

GpuPixelBuffer2D::~GpuPixelBuffer2D() {
    cudaFree(_buffer);
}

unsigned char* GpuPixelBuffer2D::data() {
    return _buffer;
}

const unsigned char* GpuPixelBuffer2D::data() const {
    return _buffer;
}

unsigned int GpuPixelBuffer2D::size() const {
    return _bufferSize;
}

unsigned int GpuPixelBuffer2D::length() const {
    return _sideLength;
}

void GpuPixelBuffer2D::clear() {
    CUDA_CALL(cudaMemset(_buffer, 0x0, _bufferSize));
}

void GpuPixelBuffer2D::copyToHost(unsigned char* dst) const {
    CUDA_CALL(cudaMemcpy(dst, _buffer, _bufferSize, cudaMemcpyDeviceToHost));
}

void GpuPixelBuffer2D::copyFromHost(const unsigned char* src) {
    CUDA_CALL(cudaMemcpy(_buffer, src, _bufferSize, cudaMemcpyHostToDevice));
}

// CPUColorMapBuffer functions definitions

unsigned char* CpuColorMapBuffer::data() {
    return _buffer;
}

const unsigned char* CpuColorMapBuffer::data() const {
    return _buffer;
}

unsigned char& CpuColorMapBuffer::red(unsigned int index) {
    return _buffer[3 * index + 0];
}

unsigned char& CpuColorMapBuffer::green(unsigned int index) {
    return _buffer[3 * index + 1];
}

unsigned char& CpuColorMapBuffer::blue(unsigned int index) {
    return _buffer[3 * index + 2];
}

const unsigned char& CpuColorMapBuffer::red(unsigned int index) const {
    return _buffer[3 * index + 0];
}

const unsigned char& CpuColorMapBuffer::green(unsigned int index) const {
    return _buffer[3 * index + 1];
}

const unsigned char& CpuColorMapBuffer::blue(unsigned int index) const {
    return _buffer[3 * index + 2];
}

unsigned int CpuColorMapBuffer::size() const {
    return sizeof(_buffer);
}

unsigned int CpuColorMapBuffer::length() const {
    return _length;
}
}  // namespace graphics
