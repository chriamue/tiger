#ifndef _NEURALFLOWPANE_H_
#define _NEURALFLOWPANE_H_
#include "NeuralLayer.h"
#include "AlloyUI.h"
namespace aly {
	class NeuralFlowPane :public Composite {
	protected:
		std::vector<NeuralLayerRegionPtr> layerRegions;
		bool dragEnabled;
	public:
		virtual bool NeuralFlowPane::onEventHandler(AlloyContext* context, const InputEvent& e) override;
		NeuralFlowPane(const std::string& name, const AUnit2D& pos, const AUnit2D& dims);
		void add(tgr::NeuralLayer* layer, const pixel2& cursor);
	};
	typedef std::shared_ptr<NeuralFlowPane> NeuralFlowPanePtr;
}
#endif