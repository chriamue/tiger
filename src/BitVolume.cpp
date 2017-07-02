/*
 * BitVolume.cpp
 *
 *  Created on: Jul 1, 2017
 *      Author: blake
 */

#include "BitVolume.h"
namespace aly {
BitVolume::BitVolume(int rows, int cols, int slices) :
		rows(rows), cols(cols), slices(slices) {
	resize(rows, cols, slices);
}
bool BitVolume::operator ()(int i, int j, int k) const {
	return data[clamp(i, 0, rows - 1) + clamp(j, 0, cols - 1) * rows
			+ clamp(k, 0, slices - 1) * rows * cols];
}
void BitVolume::set(int i, int j, int k, bool val) {
	data[clamp(i, 0, rows - 1) + clamp(j, 0, cols - 1) * rows
			+ clamp(k, 0, slices - 1) * rows * cols] = val;
}
void BitVolume::resize(int r, int c, int s) {
	data.resize(r * c * s);
	rows = r;
	cols = c;
	slices = s;
}
void BitVolume::resize(int r, int c, int s, bool val) {
	data.resize(r * c * s, val);
	rows = r;
	cols = c;
	slices = s;
}
void BitVolume::clear() {
	data.clear();
}
}
