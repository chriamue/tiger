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
#include "NeuralFlowPane.h"
#include "AlloyApplication.h"
#include "AlloyDrawUtil.h"
using namespace tgr;
namespace aly{

	bool NeuralConnection::operator ==(const std::shared_ptr<NeuralConnection> & r) const {
		return (source == r->source && destination == r->destination);
	}
	bool NeuralConnection::operator !=(const std::shared_ptr<NeuralConnection> & r) const{
		return (source != r->source || destination != r->destination);
	}
	bool NeuralConnection::operator <(const std::shared_ptr<NeuralConnection> & r) const{
		return (std::make_tuple(source, destination) < std::make_tuple(r->source, r->destination));
	}
	bool NeuralConnection::operator >(const std::shared_ptr<NeuralConnection> & r) const{
		return (std::make_tuple(source, destination) > std::make_tuple(r->source, r->destination));
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
		float lineWidth = 6.0f*scale;
		if (source->isFocused()) {
			nvgStrokeWidth(nvg, std::max(scale * 6.0f, 1.0f));
			nvgStrokeColor(nvg, context->theme.LIGHTEST);
			nvgFillColor(nvg, context->theme.LIGHTEST);
		}
		else {
			nvgStrokeWidth(nvg, std::max(scale * 4.0f, 1.0f));
			nvgStrokeColor(nvg, context->theme.LIGHTEST.toDarker(0.8f));
			nvgFillColor(nvg, context->theme.LIGHTEST.toDarker(0.8f));
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

		nvgLineCap(nvg, NVG_ROUND);
		nvgLineJoin(nvg, NVG_BEVEL);
		nvgBeginPath(nvg);

		float2 pos = aly::round(offset + path.front());
		float2 dims(15.0f,10.0f);
		nvgMoveTo(nvg, pos.x , pos.y + dims.y);
		nvgLineTo(nvg, pos.x-dims.x*0.5f, pos.y);
		nvgLineTo(nvg, pos.x + dims.x*0.5f, pos.y);
		nvgClosePath(nvg);
		
		nvgFill(nvg);
	}

	NeuralFlowPane::NeuralFlowPane(const std::string& name, const AUnit2D& pos, const AUnit2D& dims) :
		Composite(name, pos, dims),dragEnabled(false) {
		router.setBorderSpacing(30.0f);
		ImageRGBA img;
		ReadImageFromFile(AlloyApplicationContext()->getFullPath("images/tiger.png"), img);
		/*
		float2 center(0.5f*img.width, 0.5f*img.height);
		float t = 0.4f;
		for (int j = 0; j < img.height; j++) {
			for (int i = 0; i < img.width; i++) {
				RGBA& c = img(i, j);
				float r = distance(float2(i, j), center);

				if (r < t*img.height) {
					c.w = 192;
				}
				else {
					float s=1.0f-(r - t*img.height)/((1.0f-t)*img.height);
					c.w =  uint8_t(std::floor(192 *s*s));
				}
			}
		}
		WriteImageToFile(GetDesktopDirectory()+ALY_PATH_SEPARATOR+"tiger.png",img);
		*/
		backgroundImage = ImageGlyphPtr(new ImageGlyph(img,AlloyApplicationContext().get(),false));
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

		NVGcontext* nvg = context->nvgContext;
		box2px bounds = getBounds();
		float w = bounds.dimensions.x;
		float h = bounds.dimensions.y;
		pixel lineWidth = borderWidth.toPixels(bounds.dimensions.y, context->dpmm.y,
			context->pixelRatio);
		if (isScrollEnabled()) {
			pushScissor(nvg, getCursorBounds());
		}
		if (backgroundColor->a > 0) {
			nvgBeginPath(nvg);
			if (roundCorners) {
				nvgRoundedRect(nvg, bounds.position.x, bounds.position.y,
					bounds.dimensions.x, bounds.dimensions.y,
					context->theme.CORNER_RADIUS);
			}
			else {
				nvgRect(nvg, bounds.position.x, bounds.position.y,
					bounds.dimensions.x, bounds.dimensions.y);
			}
			nvgFillColor(nvg, *backgroundColor);
			nvgFill(nvg);
		}
		pushScissor(nvg, getCursorBounds());

		box2px imgb = bounds;
		float imga=backgroundImage->width / (float)backgroundImage->height;
		float ba = imgb.dimensions.x / imgb.dimensions.y;
		if (imga < ba) {
			imgb.dimensions = pixel2(imgb.dimensions.x, imgb.dimensions.x / imga);
		}
		else {
			imgb.dimensions = pixel2(imgb.dimensions.y*imga, imgb.dimensions.y);
		}
		imgb.position = bounds.center() -0.5f*imgb.dimensions;
		backgroundImage->draw(imgb, context->theme.DARK.toSemiTransparent(0.5f), COLOR_NONE, context);
		
		for (NeuralConnectionPtr con : connections) {
			if (con->source->isVisible() && con->destination->isVisible()) {
				con->draw(context, this);
			}
		}
		popScissor(nvg);
		for (std::shared_ptr<Region>& region : children) {
			if (region->isVisible()) {
				region->draw(context);
			}
		}

		if (verticalScrollTrack.get() != nullptr) {
			if (isScrollEnabled()) {
				if (extents.dimensions.y > h) {
					verticalScrollTrack->draw(context);
					verticalScrollHandle->draw(context);
				}
				else {
					verticalScrollTrack->draw(context);
				}
				if (extents.dimensions.x > w) {
					horizontalScrollTrack->draw(context);
					horizontalScrollHandle->draw(context);
				}
				else {
					horizontalScrollTrack->draw(context);
				}
			}
		}

		if (isScrollEnabled()) {
			popScissor(nvg);
		}
		if (borderColor->a > 0) {

			nvgLineJoin(nvg, NVG_ROUND);
			nvgBeginPath(nvg);
			if (roundCorners) {
				nvgRoundedRect(nvg, bounds.position.x + lineWidth * 0.5f,
					bounds.position.y + lineWidth * 0.5f,
					bounds.dimensions.x - lineWidth,
					bounds.dimensions.y - lineWidth,
					context->theme.CORNER_RADIUS);
			}
			else {
				nvgRect(nvg, bounds.position.x + lineWidth * 0.5f,
					bounds.position.y + lineWidth * 0.5f,
					bounds.dimensions.x - lineWidth,
					bounds.dimensions.y - lineWidth);
			}
			nvgStrokeColor(nvg, *borderColor);
			nvgStrokeWidth(nvg, lineWidth);
			nvgStroke(nvg);
			nvgLineJoin(nvg, NVG_MITER);
		}
	}
	void NeuralFlowPane::update() {
		for (NeuralLayerRegionPtr layerRegion : layerRegions) {
			NeuralLayer* layer = layerRegion->getLayer();
			bool visible = false;
			for (auto child : layer->getChildren()) {
				if (child->isVisible()) {
					visible = true;
				}
			}
			layerRegion->setExpandable(!visible&&layer->getChildren().size()>0);
		}
	}
	void NeuralFlowPane::add(tgr::NeuralLayer* layer,const pixel2& cursor) {
		AlloyContext* context = AlloyApplicationContext().get();
		if (!layer->hasRegion()) {
			pixel2 offset = getDrawOffset() + getBoundsPosition();
			NeuralLayerRegionPtr layerRegion = layer->getRegion();
			float2 dims = layerRegion->dimensions.toPixels(float2(context->screenDimensions()), context->dpmm, context->pixelRatio);
			layerRegion->position = CoordPX(aly::round(cursor - offset - 0.5f*dims));
			Composite::add(layerRegion);
			for(auto child:layer->getChildren()){
				if (child->hasRegion()) {
					NeuralConnectionPtr con = NeuralConnectionPtr(new NeuralConnection(layerRegion, child->getRegion()));
					connections.insert(con);
				}
			}
			for (auto dep : layer->getDependencies()) {
				if (dep->hasRegion()) {
					NeuralConnectionPtr con = NeuralConnectionPtr(new NeuralConnection(dep->getRegion(), layerRegion));
					connections.insert(con);
				}
			}
			router.add(std::dynamic_pointer_cast<dataflow::AvoidanceNode>(layerRegion));
			layerRegion->setVisible(true);
			layerRegions.push_back(layerRegion);
		} else {
			NeuralLayerRegionPtr layerRegion = layer->getRegion();
			layerRegion->setDragOffset(float2(0.0f, 0.0f));
			layerRegion->reset();
			layerRegion->setVisible(true);
			AlloyContext* context = AlloyApplicationContext().get();
			pixel2 offset = getDrawOffset() + getBoundsPosition();
			pixel2 dims = layerRegion->dimensions.toPixels(float2(context->screenDimensions()), context->dpmm, context->pixelRatio);
			layerRegion->position = CoordPX(aly::round(cursor - offset - 0.5f*dims));
		}
	}
}