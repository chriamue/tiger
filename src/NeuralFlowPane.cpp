#include "NeuralFlowPane.h"
#include "AlloyApplication.h"
namespace aly{
	NeuralFlowPane::NeuralFlowPane(const std::string& name, const AUnit2D& pos, const AUnit2D& dims) :
		Composite(name, pos, dims),dragEnabled(false) {

		Application::addListener(this);
	}

	bool NeuralFlowPane::onEventHandler(AlloyContext* context, const InputEvent& e) {
		if (e.type == InputType::MouseButton&&e.isUp() && e.button == GLFW_MOUSE_BUTTON_RIGHT) {
			dragEnabled = false;
		}

		if (!isVisible()||!context->isMouseOver(this,true)) {
			return false;
		}

		if (e.type == InputType::Scroll&&(e.isControlDown() ||context->isMouseOver(this,false))) {
			for (NeuralLayerRegionPtr r : layerRegions) {
				r->setScale(e.scroll.y, e.cursor);
			}
			context->requestPack();
		}
		if (e.type == InputType::Cursor&&dragEnabled) {
			for (NeuralLayerRegionPtr r : layerRegions) {
				r->setDragOffset(e.cursor, r->cursorOffset);
			}
			context->requestPack();
			return true;
		}
		if (e.type == InputType::MouseButton&&e.isDown()&&e.button == GLFW_MOUSE_BUTTON_RIGHT) {
			for (NeuralLayerRegionPtr r : layerRegions) {
				r->cursorOffset = e.cursor - r->getBoundsPosition();
				r->setDragOffset(e.cursor, r->cursorOffset);
			}
			dragEnabled = true;
			context->requestPack();
			return true;
		}
		return false;
	}
	void NeuralFlowPane::add(tgr::NeuralLayer* layer,const pixel2& cursor) {
		
		AlloyContext* context = AlloyApplicationContext().get();
		if (!layer->hasRegion()) {
			pixel2 offset = getDrawOffset() + getBoundsPosition();
			NeuralLayerRegionPtr layerRegion = layer->getRegion();
			float2 dims = layerRegion->dimensions.toPixels(float2(context->screenDimensions()), context->dpmm, context->pixelRatio);
			layerRegion->position = CoordPX(cursor - offset - 0.5f*dims);
			Composite::add(layerRegion);
			layerRegions.push_back(layerRegion);
		}
		else {
			NeuralLayerRegionPtr layerRegion = layer->getRegion();
			AlloyContext* context = AlloyApplicationContext().get();
			pixel2 offset = getDrawOffset() + getBoundsPosition();
			float2 dims = layerRegion->dimensions.toPixels(float2(context->screenDimensions()), context->dpmm, context->pixelRatio);
			layerRegion->position = CoordPX(cursor - offset - 0.5f*dims);
		}

		
	}
}