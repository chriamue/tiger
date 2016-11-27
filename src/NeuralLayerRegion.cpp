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
#include "NeuralLayerRegion.h"
#include "AlloyDrawUtil.h"
#include "AlloyApplication.h"
#include "NeuralLayer.h"
using namespace tgr;
namespace aly {
	const float NeuralLayerRegion::fontSize = 24.0f;
	box2px NeuralLayerRegion::getObstacleBounds() const {
		box2px box = getBounds(false);
		box.position.y += 26.0f;
		box.dimensions.y -= 26.0f;
		return box;
	}
	NeuralLayerRegion::NeuralLayerRegion(const std::string& name, tgr::NeuralLayer* layer,
		const AUnit2D& pos, const AUnit2D& dims, bool resizeable) :
		Composite(name, pos, dims), layer(layer), scale(1.0f){
		selectionRadius = 2;
		cellPadding = pixel2(10, 10);
		if (resizeable) {
			Application::addListener(this);
		}
		setDragEnabled(true);
		setClampDragToParentBounds(false);
		AlloyContext* context = AlloyApplicationContext().get();
		std::shared_ptr<IconButton> cancelButton = std::shared_ptr<IconButton>(new IconButton(0xf070, CoordPX(0.0f, 2.0f), CoordPX(24, 24), IconType::CIRCLE));
		cancelButton->setOrigin(Origin::TopLeft);
		cancelButton->setRescaleOnHover(false);
		cancelButton->borderColor = MakeColor(AlloyApplicationContext()->theme.LIGHTEST);
		cancelButton ->iconColor = MakeColor(AlloyApplicationContext()->theme.LIGHT);
		cancelButton->foregroundColor = MakeColor(COLOR_NONE);
		cancelButton->backgroundColor = MakeColor(COLOR_NONE);
		cancelButton->borderWidth = UnitPX(0.0f);
		cancelButton->onMouseDown = [this](AlloyContext* context, const InputEvent& event) {
			this->setVisible(false);
			return true;
		};
		Composite::add(cancelButton);
	}

	void NeuralLayerRegion::draw(AlloyContext* context) {

		NVGcontext* nvg = context->nvgContext;
		aly::box2px bounds = getBounds();

		pushScissor(nvg, parent->getCursorBounds());
		nvgBeginPath(nvg);
		nvgRoundedRect(nvg, bounds.position.x, bounds.position.y, 26.0f, fontSize + 8.0f, 3.0f);
		nvgFillColor(nvg, context->theme.DARK.toSemiTransparent(0.5f));
		nvgFill(nvg);
		popScissor(nvg);
		Composite::draw(context);


		pushScissor(nvg, parent->getCursorBounds());
		nvgFontSize(nvg, fontSize);
		nvgFontFaceId(nvg, context->getFontHandle(FontType::Bold));
		nvgTextAlign(nvg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
		drawText(nvg, bounds.position + float2(28.0f, 0.0f), name, FontStyle::Outline, context->theme.LIGHTER, context->theme.DARK);
		popScissor(nvg);
		pushScissor(nvg, getCursorBounds());


		nvgBeginPath(nvg);
		nvgRoundedRect(nvg, bounds.position.x, bounds.position.y+fontSize+2.0f, bounds.dimensions.x, bounds.dimensions.y-fontSize-2.0f,3.0f);
		if (isFocused()) {
			nvgFillColor(nvg, context->theme.LIGHTEST);
		}
		else {
			nvgFillColor(nvg, context->theme.LIGHTEST.toDarker(0.8f));
		}
		nvgFill(nvg);

		pixel2 origin = bounds.position;
		pixel2 padding = getPadding();
		bounds.position += float2(0.0f,fontSize+8.0f);
		bounds.dimensions -= padding;
		int width = layer->width;
		int height = layer->height;
		float scale = bounds.dimensions.x / width;

		nvgBeginPath(nvg);
		nvgRect(nvg, bounds.position.x, bounds.position.y, bounds.dimensions.x, bounds.dimensions.y);
		nvgFillColor(nvg, context->theme.DARKER);
		nvgFill(nvg);

		float rOuter = 0.5f*scale;
		float rInner = 0.25f*scale;
		float lineWidth = scale*0.01f;
		nvgStrokeWidth(nvg, lineWidth);
		int bins = layer->bins;
		pixel2 pos = pixel2(-1, -1);
		int2 selected = int2(-1, -1);

		if (bounds.contains(cursorPosition)) {
			pos = (cursorPosition - bounds.position) / bounds.dimensions;
			selected = int2((int)std::floor(width*pos.x), (int)std::floor(height*pos.y));
		}
		if (lastSelected != selected) {
			for (Neuron* neuron : activeList) {
				neuron->active = false;
			}
			activeList.clear();
			if (selected.x != -1) {
				std::vector<Neuron*> out;
				(*layer)(selected.x, selected.y).getInputNeurons(out);
				for (Neuron* neuron : out) {
					neuron->active = true;
					activeList.push_back(neuron);
				}
			}
			lastSelected = selected;
		}
		std::list<int2> highlight;
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				float2 center = float2(bounds.position.x + (i + 0.5f)*scale, bounds.position.y + (j + 0.5f)*scale);
				nvgFillColor(nvg, Color(92, 92, 92));
				nvgBeginPath(nvg);
				nvgCircle(nvg, center.x, center.y, scale*0.5f);
				nvgFill(nvg);

				Neuron& n = (*layer)(i, j);
				if (n.active) {
					highlight.push_back(int2(i, j));
				}
				if (bins > 1) {
					int b = layer->getBin(n);
					float aeps = 0.5f / rOuter;
					float a0 = (float)(b - 0.5f) / bins * NVG_PI * 2.0f - NVG_PI*0.5f;
					float a1 = (float)(b + 0.5f) / bins * NVG_PI * 2.0f - NVG_PI*0.5f;

					nvgFillColor(nvg, Color(160, 160, 160));
					nvgBeginPath(nvg);
					nvgArc(nvg, center.x, center.y, rInner, a0, a1, NVG_CW);
					nvgArc(nvg, center.x, center.y, rOuter, a1, a0, NVG_CCW);
					nvgClosePath(nvg);
					nvgFill(nvg);
				}
				if (lineWidth > 0.1f&&selected.x != -1 && std::abs(i - selected.x) <= selectionRadius&&std::abs(j - selected.y) <= selectionRadius) {
					int N = (int)n.getInputWeightSize();
					nvgLineCap(nvg, NVG_SQUARE);
					for (int i = 0; i < N; i++) {
						SignalPtr& sig = n.getInput(i);
						float a = 2.0f*i*NVG_PI / N - 0.5f*NVG_PI;
						float cosa = std::cos(a);
						float sina = std::sin(a);
						float sx = center.x + rInner*cosa;
						float sy = center.y + rInner*sina;
						float tw = mix(rInner, rOuter - lineWidth, sig->value);
						float wx = center.x + tw*cosa;
						float wy = center.y + tw*sina;
						float ex = center.x + (rOuter - lineWidth)*cosa;
						float ey = center.y + (rOuter - lineWidth)*sina;
						nvgStrokeColor(nvg, Color(128, 128, 128));
						nvgStrokeWidth(nvg, lineWidth);
						nvgBeginPath(nvg);
						nvgMoveTo(nvg, sx, sy);
						nvgLineTo(nvg, ex, ey);
						nvgStroke(nvg);

						nvgStrokeColor(nvg, Color(220, 220, 220));
						nvgBeginPath(nvg);
						nvgMoveTo(nvg, sx, sy);
						nvgLineTo(nvg, wx, wy);
						nvgStrokeWidth(nvg, 3 * lineWidth);
						nvgStroke(nvg);

					}
				}
				nvgStrokeColor(nvg, Color(220, 220, 220));
				nvgStrokeWidth(nvg, lineWidth);
				nvgFillColor(nvg, Color(ColorMapToRGB(n.value, ColorMap::RedToBlue)));
				nvgBeginPath(nvg);
				nvgCircle(nvg, center.x, center.y, rInner);
				nvgFill(nvg);
				nvgStroke(nvg);

				nvgStrokeColor(nvg, Color(92, 92, 92));
				nvgBeginPath(nvg);
				nvgCircle(nvg, center.x, center.y, scale*0.5f);
				nvgStroke(nvg);
			}
		}
		if (selected.x != -1 && selected.y != -1) {
			nvgStrokeWidth(nvg, 2.0f);
			float2 center = float2(bounds.position.x + (selected.x + 0.5f)*scale, bounds.position.y + (selected.y + 0.5f)*scale);
			nvgStrokeColor(nvg, Color(200, 200, 200));
			nvgBeginPath(nvg);
			nvgCircle(nvg, center.x, center.y, scale*0.5f);
			nvgStroke(nvg);
		}
		for (int2 pos : highlight) {
			nvgStrokeWidth(nvg, 2.0f);
			float2 center = float2(bounds.position.x + (pos.x + 0.5f)*scale, bounds.position.y + (pos.y + 0.5f)*scale);
			nvgStrokeColor(nvg, Color(200, 200, 200));
			nvgBeginPath(nvg);
			nvgCircle(nvg, center.x, center.y, scale*0.5f);
			nvgStroke(nvg);
		}
		popScissor(context->nvgContext);

		if (selected.x != -1 && selected.y != -1) {
			Neuron* neuron = layer->get(selected.x, selected.y);

			context->setCursor(&Cursor::CrossHairs);
			nvgFontFaceId(nvg, context->getFontHandle(FontType::Bold));
			nvgFontSize(nvg, 16.0f);
			nvgTextAlign(nvg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
			drawText(nvg, cursorPosition + pixel2(0.0f, 10.0f), MakeString() << selected, FontStyle::Outline, context->theme.LIGHTER, context->theme.DARKER);
			drawText(nvg, cursorPosition + pixel2(0.0f, 10.0f + 16.0f), MakeString() << "in: " << neuron->getInputNeuronSize() << " / " << neuron->getInputWeightSize(), FontStyle::Outline, context->theme.LIGHTER, context->theme.DARKER);
			drawText(nvg, cursorPosition + pixel2(0.0f, 10.0f + 32.0f), MakeString() << "out: " << neuron->getOutputNeuronSize(), FontStyle::Outline, context->theme.LIGHTER, context->theme.DARKER);

		}
	}
	bool NeuralLayerRegion::onEventHandler(AlloyContext* context, const InputEvent& e) {
		if (Composite::onEventHandler(context, e))
			return true;

		bool over = context->isMouseOver(this, true);
		if (e.type == InputType::MouseButton
			&& e.button == GLFW_MOUSE_BUTTON_LEFT && e.isDown() && over) {
			dynamic_cast<Composite*>(this->parent)->putLast(this);
		}
		if (over) {
			cursorPosition = e.cursor;
		}
		else {
			cursorPosition = float2(-1, -1);
		}
		if (e.type == InputType::Scroll&&over) {
			setScale(e.scroll.y, e.cursor);
			context->requestPack();
			return true;
		}


		return false;
	}
	bool NeuralLayerRegion::isFocused(bool recurse) const {
		bool ret=(activeList.size() > 0 || lastSelected.x != -1);
		if (ret)return true;
		if (recurse) {
			for (auto child : layer->getChildren()) {
				if (child->hasRegion() && child->getRegion()->isFocused(false)) {
					return true;
				}
			}
		}
		return false;
	}
	void  NeuralLayerRegion::setScale(float s, pixel2 cursor) {
		AlloyContext* context = AlloyApplicationContext().get();
		box2px bounds = getBounds(false);
		float lastScale = scale;
		scale = clamp(scale * (1.0f + s * 0.1f), 0.1f, 10.0f);
		float scaling = scale / lastScale;
		pixel2 padding = getPadding();
		pixel2 newBounds = (bounds.dimensions - padding)*scaling + padding;

		pixel2 relPos = (cursor - bounds.position) / bounds.dimensions;
		pixel2 newPos = cursor - relPos*newBounds;
		bounds.position = aly::round(newPos);
		bounds.dimensions = aly::round(newBounds);
		setDragOffset(pixel2(0, 0));
		position = CoordPX(bounds.position - parent->getBoundsPosition());
		dimensions = CoordPX(bounds.dimensions);
	}
}