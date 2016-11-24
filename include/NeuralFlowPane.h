#ifndef _NEURALFLOWPANE_H_
#define _NEURALFLOWPANE_H_
#include "NeuralLayer.h"
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
			return bounds.position + float2(bounds.dimensions.x*0.5f, bounds.dimensions.y);
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

		bool dragEnabled;
	public:
		std::set<NeuralConnectionPtr> connections;
		virtual void pack(const pixel2& pos, const pixel2& dims, const double2& dpmm, double pixelRatio, bool clamp) override;
		virtual bool NeuralFlowPane::onEventHandler(AlloyContext* context, const InputEvent& e) override;
		NeuralFlowPane(const std::string& name, const AUnit2D& pos, const AUnit2D& dims);
		void add(tgr::NeuralLayer* layer, const pixel2& cursor);
		virtual void draw(AlloyContext* context) override;
	};
	typedef std::shared_ptr<NeuralFlowPane> NeuralFlowPanePtr;
}
#endif