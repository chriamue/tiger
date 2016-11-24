#ifndef _NEURALFLOWPANE_H_
#define _NEURALFLOWPANE_H_
#include "NeuralLayer.h"
#include "AlloyUI.h"
#include "AvoidanceRouting.h"
namespace aly {
	class NeuralFlowPane;
	class NeuralConnection {
	public:
		tgr::NeuralLayer* source;
		tgr::NeuralLayer* target;
		bool selected = false;
		std::vector<float2> path;
		void setSelected(bool b) {
			selected = b;
		}
		bool isSelected() const {
			return selected;
		}
		NeuralConnection(tgr::NeuralLayer* source=nullptr, tgr::NeuralLayer* target=nullptr):source(source),target(target) {
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
		virtual bool NeuralFlowPane::onEventHandler(AlloyContext* context, const InputEvent& e) override;
		NeuralFlowPane(const std::string& name, const AUnit2D& pos, const AUnit2D& dims);
		void add(tgr::NeuralLayer* layer, const pixel2& cursor);
		virtual void draw(AlloyContext* context) override;
	};
	typedef std::shared_ptr<NeuralFlowPane> NeuralFlowPanePtr;
}
#endif