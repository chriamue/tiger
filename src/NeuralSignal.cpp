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
#include <NeuralSignal.h>
#include "NeuralLayer.h"
namespace tgr {
bool Terminal::operator ==(const Terminal & r) const {
	return (x == r.x && y == r.y && layer == r.layer);
}
bool Terminal::operator !=(const Terminal & r) const {
	return (x != r.x || y != r.y || layer != r.layer);
}
bool Terminal::operator <(const Terminal & r) const {
	return (std::make_tuple(x, y, (layer) ? layer->getId() : -1)
			< std::make_tuple(r.x, r.y, (layer) ? layer->getId() : -1));
}
bool Terminal::operator >(const Terminal & r) const {
	return (std::make_tuple(x, y, (layer) ? layer->getId() : -1)
			< std::make_tuple(r.x, r.y, (layer) ? layer->getId() : -1));
}
bool isTrainableWeight(ChannelType vtype) {
	return (static_cast<int>(vtype) & static_cast<int>(ChannelType::weight) == static_cast<int>(ChannelType::weight));
}
size_t ShapeVolume(aly::int3 dims) {
	return (size_t) dims.x * (size_t) dims.y * (size_t) dims.z;
}
float* NeuralSignal::getValuePtr(const aly::int3& pos) {
	if(pos.z>value.size())return nullptr;
	if(value[pos.z].size()==0)return nullptr;
	return &value[pos.z][aly::clamp(pos.x,0, dimensions.x - 1)
			+ dimensions.x * aly::clamp(pos.y, 0, dimensions.y - 1)];
}
float* NeuralSignal::getChangePtr(const aly::int3& pos) {
	if(pos.z>value.size())return nullptr;
	if(value[pos.z].size()==0)return nullptr;
	return &change[pos.z][aly::clamp(pos.x,
			0, dimensions.x - 1)
			+ dimensions.x * aly::clamp(pos.y, 0, dimensions.y - 1)];
}
float NeuralSignal::getValue(const aly::int3& pos) {
	return value[aly::clamp(pos.z, 0, dimensions.z - 1)][aly::clamp(pos.x,
			0, dimensions.x - 1)
			+ dimensions.x * aly::clamp(pos.y, 0, dimensions.y - 1)];
}
float NeuralSignal::getChange(const aly::int3& pos) {
	return change[aly::clamp(pos.z, 0, dimensions.z - 1)][aly::clamp(pos.x,
			0, dimensions.x - 1)
			+ dimensions.x * aly::clamp(pos.y, 0, dimensions.y - 1)];
}
NeuralSignal::NeuralSignal(NeuralLayer* input, aly::int3 dimensions, ChannelType type) :
		type(type), id(-1), value( { Storage(ShapeVolume(dimensions)) }), change(
				{ Storage(ShapeVolume(dimensions)) }), input(input) {
}
void NeuralSignal::clearGradients() {
	for (Storage& store : change) {
		store.assign(store.size(),0.0f);
	}
}
void NeuralSignal::mergeGradients(Storage& dst) {
	const auto &grad_head = change[0];
	size_t sz = grad_head.size();
	dst.resize(sz);
	std::copy(grad_head.begin(), grad_head.end(), &dst[0]);
#pragma omp parallel for
	for (int i = 0; i < sz; i++) {
		for (size_t sample = 1; sample < (int) change.size(); ++sample) {
			Storage& cur = change[sample];
			dst[i] += cur[i];
		}
	}
}
void NeuralSignal::addOutput(const NeuralLayerPtr& output) {
	outputs.push_back(output);
}
NeuralSignal& NeuralSignal::operator=(const NeuralSignal& other) {
	//Does not copy references to inputs and outputs.
	value = other.value;
	change = other.change;
	dimensions = other.dimensions;
	type = other.type;
	id = other.id;
	return *this;
}
}
