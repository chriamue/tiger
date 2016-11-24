#include "AlloyWidget.h"
#ifndef _NEURAL_LAYER_REGION_H_
#define _NEURAL_LAYER_REGION_H_
namespace tgr {
	class NeuralLayer;
	class Neuron;
}
namespace aly {
	
	class NeuralLayerRegion : public Composite {
	protected:
		static const float fontSize;
		tgr::NeuralLayer* layer;
		TextLabelPtr textLabel;
		int selectionRadius;
		pixel2 cursorPosition;
		int2 lastSelected;
		
		std::list<tgr::Neuron*> activeList;
	public:
		pixel2 cursorOffset;
		float scale;
		void setScale(float s,pixel2 cursor);
		static float2 getPadding() {
			return float2(0.0f, 4.0f + fontSize);
		}
		void setSelectionRadius(int radius) {
			selectionRadius = radius;
		}
		void setExtents(const box2px& ext) {
			extents = ext;
		}
		NeuralLayerRegion(const std::string& name, tgr::NeuralLayer* layer, const AUnit2D& pos,
			const AUnit2D& dims, bool resizeable = true);
		virtual bool onEventHandler(AlloyContext* context, const InputEvent& event)
			override;
		virtual void draw(AlloyContext* context) override;
	};
	typedef std::shared_ptr<NeuralLayerRegion> NeuralLayerRegionPtr;
}
#endif