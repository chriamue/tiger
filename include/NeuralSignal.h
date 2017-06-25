/*
 * Copyright(C) 2016, Blake C. Lucas, Ph.D. (img.science@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef NEURALSIGNAL_H_
#define NEURALSIGNAL_H_
#include <AlloyMath.h>
#include <AlloyOptimizationMath.h>
#include <memory>
#include <map>
#include "tiny_dnn/util/util.h"
namespace tgr {
enum class ChannelType
	: int32_t {
		// 0x0001XXX : in/out data
	data = 0x0001000,  // input/output data, fed by other layer or input channel
	// 0x0002XXX : trainable parameters, updated for each back propagation
	weight = 0x0002000,
	bias = 0x0002001,
	label = 0x0004000,
	aux = 0x0010000  // layer-specific storage
};
enum class Padding {
	Valid = tiny_dnn::padding::valid, Same = tiny_dnn::padding::same
};

inline std::vector<ChannelType> ChannelOrder(bool has_bias) {
	if (has_bias) {
		return {ChannelType::data, ChannelType::weight, ChannelType::bias};
	} else {
		return {ChannelType::data, ChannelType::weight};
	}
}
enum class BackendType {
	internal = 0, nnpack = 1, libdnn = 2, avx = 3, opencl = 4
};
inline aly::dim3 Convert(const tiny_dnn::shape3d& s) {
	return aly::dim3(s.width_, s.height_, s.depth_);
}
inline tiny_dnn::shape3d Convert(const aly::dim3& d) {
	return tiny_dnn::shape3d(d.x, d.y, d.z);
}
inline std::vector<aly::dim3> Convert(
		const std::vector<tiny_dnn::shape3d>& shapes) {
	std::vector<aly::dim3> out(shapes.size());
	for (int i = 0; i < out.size(); i++) {
		tiny_dnn::shape3d s = shapes[i];
		out[i] = aly::dim3(s.width_, s.height_, s.depth_);
	}
	return out;
}
inline std::vector<tiny_dnn::shape3d> Convert(
		const std::vector<aly::dim3>& shapes) {
	std::vector<tiny_dnn::shape3d> out(shapes.size());
	for (int i = 0; i < out.size(); i++) {
		aly::dim3 s = shapes[i];
		out[i] = tiny_dnn::shape3d(s.x, s.y, s.z);
	}
	return out;
}

inline BackendType DefaultEngine() {
#ifdef CNN_USE_AVX
#if defined(__AVX__) || defined(__AVX2__)
	return BackendType::avx;
#else
#error "your compiler does not support AVX"
#endif
#else
	return BackendType::internal;
#endif
}
inline std::ostream &operator<<(std::ostream &os, BackendType type) {
	switch (type) {
	case BackendType::internal:
		os << "Internal";
		break;
	case BackendType::nnpack:
		os << "NNPACK";
		break;
	case BackendType::libdnn:
		os << "LibDNN";
		break;
	case BackendType::avx:
		os << "AVX";
		break;
	case BackendType::opencl:
		os << "OpenCL";
		break;
	default:
		throw std::runtime_error("Not supported ostream enum.");
		break;
	}
	return os;
}
bool isTrainableWeight(ChannelType vtype);
typedef std::vector<float, aly::aligned_allocator<float, 64>> Storage;
typedef std::vector<Storage> Tensor;
class NeuralLayer;
struct Terminal {
	int x;
	int y;
	NeuralLayer* layer;
	Terminal(int x = 0, int y = 0, NeuralLayer* l = nullptr) :
			x(x), y(y), layer(l) {

	}
	Terminal(int x = 0, int y = 0, const std::shared_ptr<NeuralLayer>& l =
			nullptr) :
			x(x), y(y), layer(l.get()) {

	}
	bool operator ==(const Terminal & r) const;
	bool operator !=(const Terminal & r) const;
	bool operator <(const Terminal & r) const;
	bool operator >(const Terminal & r) const;
};
size_t ShapeVolume(aly::int3 dims);
class NeuralSignal {
public:
	ChannelType type;
	aly::int3 dimensions;
	int64_t id;
	Tensor weight;
	Tensor change;
	NeuralLayer* input;
	std::vector<std::shared_ptr<NeuralLayer>> outputs;
	NeuralSignal(NeuralLayer* input, aly::int3 dimensions, ChannelType type);
	void clearGradients();
	void mergeGradients(Storage& dst);
	void addOutput(const std::shared_ptr<NeuralLayer>& output);
	NeuralSignal& operator=(const NeuralSignal& other);
};
typedef std::shared_ptr<NeuralSignal> SignalPtr;
}
#endif
