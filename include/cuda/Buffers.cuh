#pragma once

#include <cufft.h>
#include <curand_kernel.h>

namespace mhd {

// Maps the real type of the simulation to the matching cufft complex type
template <typename T>
struct FftTypes;

template <>
struct FftTypes<double> {
    using Complex = cufftDoubleComplex;
};

template <>
struct FftTypes<float> {
    using Complex = cufftComplex;
};

template <typename T>
using Complex_t = typename FftTypes<T>::Complex;

template <typename T>
class CpuBuffer1D {
private:
    T* _buffer;
    unsigned int _bufferLength;
    unsigned int _bufferSize;

public:
    CpuBuffer1D();
    CpuBuffer1D(unsigned int bufferLength);
    ~CpuBuffer1D();

    T* data();
    const T* data() const;

    T& operator[](unsigned int index);
    const T& operator[](unsigned int index) const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToDevice(T* dst) const;
    void copyFromDevice(const T* src);
};

template <typename T>
class CpuBuffer2D {
private:
    T* _buffer;
    unsigned int _sideLength;
    unsigned int _bufferSize;

public:
    CpuBuffer2D();
    CpuBuffer2D(unsigned int sideLength);
    ~CpuBuffer2D();

    T* data();
    const T* data() const;

    T& operator[](unsigned int index);
    const T& operator[](unsigned int index) const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToDevice(T* dst) const;
    void copyFromDevice(const T* src);
};

template <typename T>
class GpuBuffer2D {
private:
    T* _buffer;
    unsigned int _sideLength;
    unsigned int _bufferSize;

public:
    GpuBuffer2D();
    GpuBuffer2D(unsigned int sideLength);
    ~GpuBuffer2D();

    T* data();
    const T* data() const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToHost(T* dst) const;
    void copyFromHost(const T* src);
    void copyToDevice(T* dst) const;
    void copyFromDevice(const T* src);
};

template <typename T>
class GpuComplexBuffer2D {
public:
    using Complex = Complex_t<T>;

private:
    Complex* _buffer;
    unsigned int _sideLength;
    unsigned int _bufferSize;

public:
    GpuComplexBuffer2D();
    GpuComplexBuffer2D(unsigned int sideLength);
    ~GpuComplexBuffer2D();

    Complex* data();
    const Complex* data() const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToHost(Complex* dst) const;
    void copyFromHost(const Complex* src);
    void copyToDevice(Complex* dst) const;
    void copyFromDevice(const Complex* src);
};

// Aliases for the double-precision buffers used by the host-facing code
// (Writer, Painter), which stays in double regardless of the solver type
using CpuDoubleBuffer1D = CpuBuffer1D<double>;
using CpuDoubleBuffer2D = CpuBuffer2D<double>;
using GpuDoubleBuffer2D = GpuBuffer2D<double>;

class GpuStateBuffer2D {
private:
    curandState* _buffer;
    unsigned int _sideLength;
    unsigned int _bufferSize;

public:
    GpuStateBuffer2D();
    GpuStateBuffer2D(unsigned int sideLength);
    ~GpuStateBuffer2D();

    curandState* data();
    const curandState* data() const;

    unsigned int size() const;
    unsigned int length() const;
};
}  // namespace mhd

namespace graphics {
class CpuFloatBuffer {
private:
    float* _buffer;
    unsigned int _bufferLength;
    unsigned int _bufferSize;

public:
    CpuFloatBuffer();
    CpuFloatBuffer(unsigned int bufferLength);
    ~CpuFloatBuffer();

    float* data();
    const float* data() const;

    float& operator[](unsigned int index);
    const float& operator[](unsigned int index) const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToDevice(float* dst) const;
    void copyFromDevice(const float* src);
};

class GpuFloatBuffer {
private:
    float* _buffer;
    unsigned int _bufferLength;
    unsigned int _bufferSize;

public:
    GpuFloatBuffer();
    GpuFloatBuffer(unsigned int bufferLength);
    ~GpuFloatBuffer();

    float* data();
    const float* data() const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToDevice(float* dst) const;
    void copyFromDevice(const float* src);
};

class CpuPixelBuffer2D {
private:
    unsigned char* _buffer;
    unsigned int _sideLength;
    unsigned int _channels;
    unsigned int _bufferSize;

public:
    CpuPixelBuffer2D();
    CpuPixelBuffer2D(unsigned int sideLength, unsigned int channels = 3);
    ~CpuPixelBuffer2D();

    unsigned char* data();
    const unsigned char* data() const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToDevice(unsigned char* dst) const;
    void copyFromDevice(const unsigned char* src);
};

class GpuPixelBuffer2D {
private:
    unsigned char* _buffer;
    unsigned int _sideLength;
    unsigned int _channels;
    unsigned int _bufferSize;

public:
    GpuPixelBuffer2D();
    GpuPixelBuffer2D(unsigned int sideLength, unsigned int channels = 3);
    ~GpuPixelBuffer2D();

    unsigned char* data();
    const unsigned char* data() const;

    unsigned int size() const;
    unsigned int length() const;

    void clear();
    void copyToHost(unsigned char* dst) const;
    void copyFromHost(const unsigned char* src);
};

class CpuColorMapBuffer {
private:
    unsigned char _buffer[256 * 3];
    unsigned int _length = 256;
    unsigned int _channels = 3;

public:
    unsigned char* data();
    const unsigned char* data() const;

    unsigned char& red(unsigned int index);
    unsigned char& green(unsigned int index);
    unsigned char& blue(unsigned int index);

    const unsigned char& red(unsigned int index) const;
    const unsigned char& green(unsigned int index) const;
    const unsigned char& blue(unsigned int index) const;

    unsigned int size() const;
    unsigned int length() const;
};
}  // namespace graphics
