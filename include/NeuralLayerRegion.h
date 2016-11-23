#include "AlloyWidget.h"
#ifndef _NEURAL_LAYER_REGION_H_
#define _NEURAL_LAYER_REGION_H_
namespace tgr {
	class NeuralLayer;
}
namespace aly {
	
	class NeuralLayerRegion : public Composite {
	protected:
		pixel2 cursorDownPosition;
		box2px windowInitialBounds;
		bool resizing;
		WindowPosition winPos;
		bool resizeable;
		tgr::NeuralLayer* layer;
		void drawNeurons(AlloyContext* context);
	public:
		bool isResizing() const {
			return resizing;
		}
		bool isResizeable() const {
			return resizeable;
		}
		virtual bool isDragEnabled() const override {
			if (resizeable) {
				return ((dragButton != -1) && winPos == WindowPosition::Center);
			}
			else {
				return (dragButton != -1);
			}
		}
		std::function<void(NeuralLayerRegion* composite, const box2px& bounds)> onResize;
		NeuralLayerRegion(const std::string& name, tgr::NeuralLayer* layer, const AUnit2D& pos,
			const AUnit2D& dims, bool resizeable = true);
		virtual bool onEventHandler(AlloyContext* context, const InputEvent& event)
			override;
		virtual void draw(AlloyContext* context) override;
	};
	typedef std::shared_ptr<NeuralLayerRegion> NeuralLayerRegionPtr;
}
#endif