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
#include "NeuronLayer.h"
using namespace aly;
namespace tgr {
	NeuronLayer::NeuronLayer(int width, int height, int bins, int id):width(width),height(height),bins(bins),id(id) {
		neurons.resize(width*height*bins);
	}
	void NeuronLayer::resize(int w, int h, int b) {
		neurons.resize(w * h * b);
		neurons.shrink_to_fit();
		width = w;
		height = h;
		bins = b;
	}
	const Neuron& NeuronLayer::operator[](const size_t i) const {
		return neurons[i];
	}
	Neuron& NeuronLayer::operator[](const size_t i) {
		return neurons[i];
	}
	Neuron& NeuronLayer::operator()(const int i, const int j, const int k) {
		return neurons[aly::clamp(i, 0, width - 1) + aly::clamp(j, 0, height - 1) * width
			+ aly::clamp(k, 0, bins - 1) * width * height];
	}
	Neuron& NeuronLayer::operator()(const size_t i, const size_t j, const size_t k) {
		return neurons[aly::clamp((int)i, 0, width - 1) + aly::clamp((int)j, 0, height - 1) * width
			+ aly::clamp((int)k, 0, bins - 1) * width * height];
	}
	Neuron& NeuronLayer::operator()(const aly::int3 ijk) {
		return neurons[aly::clamp(ijk.x, 0, width - 1) + aly::clamp(ijk.y, 0, height - 1) * width
			+ aly::clamp(ijk.z, 0, bins - 1) * width * height];
	}
	Neuron& NeuronLayer::operator()(const NeuronIndex ijk) {
		return neurons[aly::clamp(ijk.x, 0, width - 1) + aly::clamp(ijk.y, 0, height - 1) * width
			+ aly::clamp(ijk.bin, 0, bins - 1) * width * height];
	}
	const Neuron& NeuronLayer::operator()(const int i, const int j, const int k) const {
		return neurons[aly::clamp(i, 0, width - 1) + aly::clamp(j, 0, height - 1) * width
			+ aly::clamp(k, 0, bins - 1) * width * height];
	}
	const Neuron& NeuronLayer::operator()(const size_t i, const size_t j, const size_t k) const {
		return neurons[aly::clamp((int)i, 0, width - 1) + aly::clamp((int)j, 0, height - 1) * width
			+ aly::clamp((int)k, 0, bins - 1) * width * height];
	}
	const Neuron& NeuronLayer::operator()(const aly::int3 ijk) const {
		return neurons[aly::clamp(ijk.x, 0, width - 1) + aly::clamp(ijk.y, 0, height - 1) * width
			+ aly::clamp(ijk.z, 0, bins - 1) * width * height];
	}
	const Neuron& NeuronLayer::operator()(const NeuronIndex ijk) const {
		return neurons[aly::clamp(ijk.x, 0, width - 1) + aly::clamp(ijk.y, 0, height - 1) * width
			+ aly::clamp(ijk.bin, 0, bins - 1) * width * height];
	}
}
