/*
 * NeuronVolume.cpp
 *
 *  Created on: Jul 1, 2017
 *      Author: blake
 */

#include "Neuron.h"
using namespace aly;
namespace tgr {

NeuronVolume::NeuronVolume(int rows, int cols, int slices) :
		rows(rows), cols(cols), slices(slices) {
	resize(rows, cols, slices);
}
const Neuron& NeuronVolume::operator ()(int i, int j, int k) const {
	return data[clamp(i, 0, rows - 1) + clamp(j, 0, cols - 1) * rows+ clamp(k, 0, slices - 1) * rows * cols];
}
Neuron& NeuronVolume::operator ()(int i, int j, int k) {
	return data[clamp(i, 0, rows - 1) + clamp(j, 0, cols - 1) * rows+ clamp(k, 0, slices - 1) * rows * cols];
}
void NeuronVolume::resize(int r, int c, int s) {
	data.resize(r * c * s);
	rows = r;
	cols = c;
	slices = s;
}
void NeuronVolume::clear() {
	data.clear();
}

}
