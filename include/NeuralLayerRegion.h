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
#ifndef _NEURAL_LAYER_REGION_H_
#define _NEURAL_LAYER_REGION_H_

#include "AlloyWidget.h"
#include "AvoidanceRouting.h"
namespace tgr {
	class NeuralLayer;
	class Neuron;
}
namespace aly {
	
	class NeuralLayerRegion : public Composite, public dataflow::AvoidanceNode{
	protected:
		static const float fontSize;
		tgr::NeuralLayer* layer;
		TextLabelPtr textLabel;
		int selectionRadius;
		pixel2 cursorPosition;
		int2 lastSelected;
		std::shared_ptr<IconButton> cancelButton;
		std::shared_ptr<IconButton> expandButton;
		std::list<tgr::Neuron*> activeList;
	public:
		pixel2 cursorOffset;
		float scale;
		tgr::NeuralLayer* getLayer() const {
			return layer;
		}
		std::function<void()> onExpand;
		std::function<void()> onHide;
		bool isFocused(bool recurse=true) const;
		void setScale(float s,pixel2 cursor);
		float setSize(float w);
		void setScale(float s) {
			scale = s;
		};
		void setExpandable(bool t) {
			expandButton->setVisible(t);
		}
		void reset() {
			cursorPosition=float2(0.0f,0.0f);
			lastSelected=int2(-1);
			activeList.clear();
			scale = 1;
			cursorOffset = float2(0.0f, 0.0f);
		}
		static float2 getPadding() {
			return float2(0.0f, 14.0f +30.0f+ fontSize);
		}
		void setSelectionRadius(int radius) {
			selectionRadius = radius;
		}
		void setExtents(const box2px& ext) {
			extents = ext;
		}
		aly::box2px getObstacleBounds() const override;
		NeuralLayerRegion(const std::string& name, tgr::NeuralLayer* layer, const AUnit2D& pos,
			const AUnit2D& dims, bool resizeable = true);
		virtual bool onEventHandler(AlloyContext* context, const InputEvent& event)
			override;
		virtual void draw(AlloyContext* context) override;
	};
	typedef std::shared_ptr<NeuralLayerRegion> NeuralLayerRegionPtr;
}
#endif