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
#include "NeuralFilter.h"
namespace tgr {
	class ConvolutionFilter :public NeuralFilter {
	protected:
		int kernelSize;
		bool bias;
		std::vector<std::pair<int,int>> connectionMap;
	public:
		ConvolutionFilter( int width, int height, int kernelSize,int features, bool bias);
		ConvolutionFilter(const NeuralLayerPtr& inputLayer, int kernelSize,int features, bool bias);
		ConvolutionFilter(const std::vector<NeuralLayerPtr>& inputLayers, int kernelSize, int features, bool bias);
		void setConnectionMap(const std::vector<std::pair<int, int>>& mapping) {
			connectionMap = mapping;
		}
		virtual void initialize(NeuralSystem& sys, const NeuronFunction& func) override;
	};
	typedef std::shared_ptr<ConvolutionFilter> ConvolutionFilterPtr;
}