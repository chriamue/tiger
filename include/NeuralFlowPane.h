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
#ifndef _NEURALFLOWPANE_H_
#define _NEURALFLOWPANE_H_
#include "NeuralLayer.h"
#include "NeuralLayerRegion.h"

#include "AlloyUI.h"
#include "AvoidanceRouting.h"
namespace aly {
	class NeuralFlowPane;
	class NeuralConnection: public dataflow::AvoidanceConnection {
	public:
		NeuralLayerRegionPtr source;
		NeuralLayerRegionPtr destination;
		bool selected = false;
		void setSelected(bool b) {
			selected = b;
		}
		bool isSelected() const {
			return selected;
		}
		dataflow::Direction getDirection() const {
			return dataflow::Direction::South;
		}
		float2 getSourceLocation() const {
			box2px bounds = source->getBounds(false);
			return bounds.position + float2(bounds.dimensions.x*0.5f, bounds.dimensions.y-30.0f);
		}
		float2 getDestinationLocation() const {
			box2px bounds = destination->getBounds(false);
			return bounds.position + float2(bounds.dimensions.x*0.5f,26.0f);
		}
		NeuralConnection(const NeuralLayerRegionPtr& source=nullptr, const NeuralLayerRegionPtr& destination=nullptr):source(source),destination(destination) {
		}
		bool operator ==(const std::shared_ptr<NeuralConnection> & r) const;
		bool operator !=(const std::shared_ptr<NeuralConnection> & r) const;
		bool operator <(const std::shared_ptr<NeuralConnection> & r) const;
		bool operator >(const std::shared_ptr<NeuralConnection> & r) const;
		~NeuralConnection() {}
		float distance(const float2& pt);
		void draw(AlloyContext* context, NeuralFlowPane* flow);
	};
	typedef std::shared_ptr<NeuralConnection> NeuralConnectionPtr;
	class NeuralFlowPane :public Composite {
	protected:
		aly::dataflow::AvoidanceRouting router;
		std::vector<NeuralLayerRegionPtr> layerRegions;
		ImageGlyphPtr backgroundImage;
		bool dragEnabled;
		tgr::NeuralLayer* selectedLayer;
	public:
		std::function<void(tgr::NeuralLayer*, const InputEvent& e)> onSelect;
		void setSelected(tgr::NeuralLayer* layer, const InputEvent& e) {
			selectedLayer = layer;
			if (onSelect)onSelect(layer,e);
		}
		void setSelected(tgr::NeuralLayer* layer) {
			selectedLayer = layer;
		}
		std::set<NeuralConnectionPtr> connections;
		virtual void pack(const pixel2& pos, const pixel2& dims, const double2& dpmm, double pixelRatio, bool clamp) override;
		virtual bool NeuralFlowPane::onEventHandler(AlloyContext* context, const InputEvent& e) override;
		NeuralFlowPane(const std::string& name, const AUnit2D& pos, const AUnit2D& dims);
		void add(tgr::NeuralLayer* layer, const pixel2& cursor);
		virtual void draw(AlloyContext* context) override;
		void update();
	};
	typedef std::shared_ptr<NeuralFlowPane> NeuralFlowPanePtr;
}
#endif