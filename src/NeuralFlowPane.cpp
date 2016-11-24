#include "NeuralFlowPane.h"
#include "AlloyApplication.h"
namespace aly{

	bool NeuralConnection::operator ==(const std::shared_ptr<NeuralConnection> & r) const {
		return (source == r->source && target == r->target);
	}
	bool NeuralConnection::operator !=(const std::shared_ptr<NeuralConnection> & r) const{
		return (source != r->source || target != r->target);
	}
	bool NeuralConnection::operator <(const std::shared_ptr<NeuralConnection> & r) const{
		return (std::make_tuple(source, target) < std::make_tuple(r->source, r->target));
	}
	bool NeuralConnection::operator >(const std::shared_ptr<NeuralConnection> & r) const{
		return (std::make_tuple(source, target) > std::make_tuple(r->source, r->target));
	}
	float NeuralConnection::distance(const float2& pt) {
		float minD = 1E30f;
		for (int i = 0; i < (int)path.size() - 1; i++) {
			lineseg2f ls(path[i], path[i + 1]);
			minD = std::min(ls.distance(pt), minD);
		}
		return minD;
	}
	void NeuralConnection::draw(AlloyContext* context, NeuralFlowPane* flow) {
		if (path.size() == 0)
			return;
		const float scale = 1.0f;// flow->getScale();
		NVGcontext* nvg = context->nvgContext;
		if (selected) {
			nvgStrokeWidth(nvg, std::max(scale * 6.0f, 1.0f));
			nvgStrokeColor(nvg, context->theme.LIGHTEST);
		}
		else {
			nvgStrokeWidth(nvg, std::max(scale * 4.0f, 1.0f));
			nvgStrokeColor(nvg, context->theme.LIGHTEST.toDarker(0.8f));
		}

		float2 offset = flow->getDrawOffset();
		nvgLineCap(nvg, NVG_ROUND);
		nvgLineJoin(nvg, NVG_BEVEL);
		nvgBeginPath(nvg);
		float2 pt0 = offset + path.front();
		nvgMoveTo(nvg, pt0.x, pt0.y);
		for (int i = 1; i < (int)path.size() - 1; i++) {
			float2 pt1 = offset + path[i];
			float2 pt2 = offset + path[i + 1];
			float diff = 0.5f
				* std::min(std::max(std::abs(pt1.x - pt2.x), std::abs(pt1.y - pt2.y)), std::max(std::abs(pt1.x - pt0.x), std::abs(pt1.y - pt0.y)));
			if (diff < scale * context->theme.CORNER_RADIUS) {
				nvgLineTo(nvg, pt1.x, pt1.y);
			}
			else {
				nvgArcTo(nvg, pt1.x, pt1.y, pt2.x, pt2.y, scale * context->theme.CORNER_RADIUS);
			}
			pt0 = pt1;
		}
		pt0 = offset + path.back();
		nvgLineTo(nvg, pt0.x, pt0.y);
		nvgStroke(nvg);
	}

	NeuralFlowPane::NeuralFlowPane(const std::string& name, const AUnit2D& pos, const AUnit2D& dims) :
		Composite(name, pos, dims),dragEnabled(false) {

		Application::addListener(this);
	}
	void NeuralFlowPane::pack(const pixel2& pos, const pixel2& dims, const double2& dpmm, double pixelRatio, bool clamp) {
		Composite::pack(pos, dims, dpmm, pixelRatio);
		router.update();
		for (NeuralConnectionPtr connect : connections) {
			router.evaluate(connect);
		}
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
	void NeuralFlowPane::draw(AlloyContext* context) {


		Composite::draw(context);
		for (NeuralConnectionPtr con : connections) {
			if (con->source->isVisible() && con->target->isVisible()) {
				con->draw(context, this);
			}
		}
	}
	void NeuralFlowPane::add(tgr::NeuralLayer* layer,const pixel2& cursor) {
		AlloyContext* context = AlloyApplicationContext().get();
		if (!layer->hasRegion()) {
			pixel2 offset = getDrawOffset() + getBoundsPosition();
			NeuralLayerRegionPtr layerRegion = layer->getRegion();
			float2 dims = layerRegion->dimensions.toPixels(float2(context->screenDimensions()), context->dpmm, context->pixelRatio);
			layerRegion->position = CoordPX(cursor - offset - 0.5f*dims);
			Composite::add(layerRegion);
			for(auto child:layer->getChildren()){
				NeuralConnectionPtr con = NeuralConnectionPtr(new NeuralConnection(layer,child.get()));
				connections.insert(con);
			}
			router.add(std::dynamic_pointer_cast<dataflow::AvoidanceNode>(layerRegion));
			layerRegions.push_back(layerRegion);
		} else {
			NeuralLayerRegionPtr layerRegion = layer->getRegion();
			AlloyContext* context = AlloyApplicationContext().get();
			pixel2 offset = getDrawOffset() + getBoundsPosition();
			float2 dims = layerRegion->dimensions.toPixels(float2(context->screenDimensions()), context->dpmm, context->pixelRatio);
			layerRegion->position = CoordPX(cursor - offset - 0.5f*dims);
		}

		
	}
}