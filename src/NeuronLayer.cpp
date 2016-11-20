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
#include "AlloyUnits.h"
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
	void NeuronLayer::draw(aly::AlloyContext* context, const aly::box2px& bounds) {

		float scale= bounds.dimensions.x/width;
		NVGcontext* nvg = context->nvgContext;
		float rOuter = 0.5f*scale;
		float rInner = 0.1f*scale;
		float lineWidth = scale*0.01f;
		nvgStrokeWidth(nvg, lineWidth);
		nvgStrokeColor(nvg, Color(64, 64, 64));
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				nvgFillColor(nvg, Color(64, 64, 64));
				float2 center = float2(bounds.position.x + (i + 0.5f)*scale, bounds.position.y + (j + 0.5f)*scale);
				nvgBeginPath(nvg);
				nvgCircle(nvg, center.x, center.y, scale*0.5f);
				nvgFill(nvg);
				for (int k = 0; k < bins; k++) {
					Neuron& n = operator()(i, j, k);
					float aeps = 0.5f / rOuter;
					float a0 = (float)(k-0.5f) / bins * NVG_PI * 2.0f - aeps;
					float a1 = (float)(k+0.5f) / bins * NVG_PI * 2.0f + aeps;
					nvgFillColor(nvg,Color( ColorMapToRGB(1.0f-n.value, ColorMap::RedToBlue)));
					nvgBeginPath(nvg);
					nvgArc(nvg, center.x, center.y, rInner, a0, a1, NVG_CW);
					nvgArc(nvg, center.x, center.y, mix(rInner,rOuter,n.value), a1, a0, NVG_CCW);
					nvgClosePath(nvg);
					nvgFill(nvg);
					if (lineWidth > 0.5f&&bins>1) {
						nvgStroke(nvg);
					}
				}
			}
		}
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
	Neuron& NeuronLayer::operator()(const Terminal ijk) {
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
	const Neuron& NeuronLayer::operator()(const Terminal ijk) const {
		return neurons[aly::clamp(ijk.x, 0, width - 1) + aly::clamp(ijk.y, 0, height - 1) * width
			+ aly::clamp(ijk.bin, 0, bins - 1) * width * height];
	}
}
