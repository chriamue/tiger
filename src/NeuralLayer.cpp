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
#include "NeuralLayer.h"
#include "AlloyUnits.h"
using namespace aly;
namespace tgr {
	std::string MakeID(int len) {
		std::stringstream ss;
		static const char lookUp[33] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345";
		for (int i = 0; i < len; i++) {
			ss << lookUp[RandomUniform(0, 31)];
		}
		return ss.str();
	}
	NeuralLayer::NeuralLayer(int width, int height, int bins, bool bias, const NeuronFunction& func) :width(width), height(height), bins(bins) {
		neurons.resize(width*height*bins, Neuron(func, bias));
		if (bias) {

		}
		id = MakeID();
	}
	NeuralLayer::NeuralLayer(const std::string& name,int width, int height, int bins,bool bias, const NeuronFunction& func) :name(name), width(width), height(height), bins(bins) {
		neurons.resize(width*height*bins,Neuron(func,bias));
		id = MakeID();
	}
	void NeuralLayer::resize(int w, int h, int b) {
		neurons.resize(w * h * b);
		neurons.shrink_to_fit();
		width = w;
		height = h;
		bins = b;
	}
	std::vector<SignalPtr> NeuralLayer::getBiasSignals() const {
		std::vector<SignalPtr> signals;
		for (const Neuron& n : neurons) {
			SignalPtr sig = n.getBiasSignal();
			if (sig.get()!=nullptr) {
				signals.push_back(sig);
			}
		}
		return signals;
	}
	void NeuralLayer::addChild(const std::shared_ptr<NeuralLayer>& layer) {
		children.push_back(layer);
		layer->dependencies.push_back(layer);
	}
	void NeuralLayer::setFunction(const NeuronFunction& func) {
		for (Neuron& n : neurons) {
			n.setFunction(func);
		}
	}
	int NeuralLayer::getBin(size_t index) const {
		return clamp((int)std::floor(neurons[index].value*bins), 0, bins-1);
	}
	int NeuralLayer::getBin(const Neuron& n) const {
		return clamp((int)std::floor(n.value*bins), 0, bins-1);
	}
	void NeuralLayer::draw(aly::AlloyContext* context, const aly::box2px& bounds) {

		float scale = bounds.dimensions.x / width;
		NVGcontext* nvg = context->nvgContext;
		float rOuter = 0.5f*scale;
		float rInner = 0.25f*scale;
		float lineWidth = scale*0.01f;
		nvgStrokeWidth(nvg, lineWidth);
		
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				float2 center = float2(bounds.position.x + (i + 0.5f)*scale, bounds.position.y + (j + 0.5f)*scale);
				nvgFillColor(nvg, Color(92, 92, 92));
				nvgBeginPath(nvg);
				nvgCircle(nvg, center.x, center.y, scale*0.5f);
				nvgFill(nvg);
				Neuron& n = operator()(i, j);
				if (bins > 1) {
					int b = getBin(n);
					float aeps = 0.5f / rOuter;
					float a0 = (float)(b - 0.5f) / bins * NVG_PI * 2.0f - NVG_PI*0.5f;
					float a1 = (float)(b + 0.5f) / bins * NVG_PI * 2.0f - NVG_PI*0.5f;

					nvgFillColor(nvg, Color(160, 160, 160));
					nvgBeginPath(nvg);
					nvgArc(nvg, center.x, center.y, rInner, a0, a1, NVG_CW);
					nvgArc(nvg, center.x, center.y, rOuter, a1, a0, NVG_CCW);
					nvgClosePath(nvg);
					nvgFill(nvg);
				}
				if (lineWidth > 0.5f) {
					int N = (int)n.input.size();
					nvgLineCap(nvg,NVG_SQUARE);
					for (int i = 0; i < N; i++) {
						SignalPtr& sig = n.input[i];
						float a = 2.0f*i*NVG_PI / N-0.5f*NVG_PI;
						float cosa = std::cos(a);
						float sina = std::sin(a);
						float sx = center.x + rInner*cosa;
						float sy = center.y + rInner*sina;
						float tw = mix(rInner, rOuter-lineWidth, sig->value);
						float wx = center.x + tw*cosa;
						float wy = center.y + tw*sina;
						float ex = center.x + (rOuter - lineWidth)*cosa;
						float ey = center.y + (rOuter - lineWidth)*sina;
						nvgStrokeColor(nvg, Color(128,128,128));
						nvgStrokeWidth(nvg, lineWidth);
						nvgBeginPath(nvg);
						nvgMoveTo(nvg, sx, sy);
						nvgLineTo(nvg, ex, ey);
						nvgStroke(nvg);

						nvgStrokeColor(nvg, Color(220,220,220));
						nvgBeginPath(nvg);
						nvgMoveTo(nvg, sx, sy);
						nvgLineTo(nvg, wx, wy);
						nvgStrokeWidth(nvg, 3*lineWidth);
						nvgStroke(nvg);

					}
				}
				nvgStrokeColor(nvg, Color(220, 220, 220));
				nvgStrokeWidth(nvg, lineWidth);
				nvgFillColor(nvg, Color(ColorMapToRGB(n.value, ColorMap::RedToBlue)));
				nvgBeginPath(nvg);
				nvgCircle(nvg, center.x, center.y, rInner);
				nvgFill(nvg);
				nvgStroke(nvg);

				nvgStrokeColor(nvg, Color(92, 92, 92));
				nvgBeginPath(nvg);
				nvgCircle(nvg, center.x, center.y, scale*0.5f);
				nvgStroke(nvg);
			}
		}
	}
	const Neuron& NeuralLayer::operator[](const size_t i) const {
		return neurons[i];
	}
	Neuron& NeuralLayer::operator[](const size_t i) {
		return neurons[i];
	}
	Neuron& NeuralLayer::get(const int i, const int j) {
		return neurons[aly::clamp(i, 0, width - 1) + aly::clamp(j, 0, height - 1) * width];
	}
	const Neuron& NeuralLayer::get(const int i, const int j) const {
		return neurons[aly::clamp(i, 0, width - 1) + aly::clamp(j, 0, height - 1) * width];
	}
	Neuron& NeuralLayer::operator()(const int i, const int j) {
		return neurons[aly::clamp(i, 0, width - 1) + aly::clamp(j, 0, height - 1) * width];
	}
	Neuron& NeuralLayer::operator()(const size_t i, const size_t j) {
		return neurons[aly::clamp((int)i, 0, width - 1) + aly::clamp((int)j, 0, height - 1) * width];
	}
	Neuron& NeuralLayer::operator()(const aly::int2 ij) {
		return neurons[aly::clamp(ij.x, 0, width - 1) + aly::clamp(ij.y, 0, height - 1) * width];
	}
	Neuron& NeuralLayer::operator()(const Terminal ij) {
		return neurons[aly::clamp(ij.x, 0, width - 1) + aly::clamp(ij.y, 0, height - 1) * width];
	}
	const Neuron& NeuralLayer::operator()(const int i, const int j) const {
		return neurons[aly::clamp(i, 0, width - 1) + aly::clamp(j, 0, height - 1) * width];
	}
	const Neuron& NeuralLayer::operator()(const size_t i, const size_t j) const {
		return neurons[aly::clamp((int)i, 0, width - 1) + aly::clamp((int)j, 0, height - 1) * width];
	}
	const Neuron& NeuralLayer::operator()(const aly::int2 ij) const {
		return neurons[aly::clamp(ij.x, 0, width - 1) + aly::clamp(ij.y, 0, height - 1) * width];
	}
	const Neuron& NeuralLayer::operator()(const Terminal ij) const {
		return neurons[aly::clamp(ij.x, 0, width - 1) + aly::clamp(ij.y, 0, height - 1) * width];
	}
}
