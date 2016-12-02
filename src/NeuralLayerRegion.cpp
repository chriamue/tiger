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
#include "NeuralFlowPane.h"
#include "AlloyDrawUtil.h"
#include "AlloyApplication.h"
#include "NeuralLayer.h"
using namespace tgr;
namespace aly {
	const float NeuralLayerRegion::fontSize = 24.0f;
	box2px NeuralLayerRegion::getObstacleBounds() const {
		box2px box = getBounds(false);
		box.position.y += 26.0f;
		box.dimensions.y -= 26.0 + 30.0f;
		return box;
	}
	void NeuralLayerRegion::drawCache(AlloyContext* context) {
		if (!cacheDirty)return;
		box2px bounds = box2px(pixel2(0, 0), pixel2(float(cacheImage.width), float(cacheImage.height)));
		NVGcontext* nvg = context->nvgContext;
		renderBuffer.begin(RGBAf(0.0f));
		nvgBeginFrame(nvg, cacheImage.width, cacheImage.height, 1.0f);
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
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				float2 center = float2(bounds.position.x + (i + 0.5f)*scale, bounds.position.y + (j + 0.5f)*scale);
				nvgFillColor(nvg, Color(92, 92, 92));
				nvgBeginPath(nvg);
				nvgCircle(nvg, center.x, center.y, scale*0.5f);
				nvgFill(nvg);
				Neuron& n = (*layer)(i, j);

				nvgStrokeColor(nvg, Color(220, 220, 220));
				nvgStrokeWidth(nvg, lineWidth);
				nvgFillColor(nvg, Color(ColorMapToRGB(n.normalizedValue(), ColorMap::RedToBlue)));
				nvgBeginPath(nvg);
				nvgCircle(nvg, center.x, center.y, rInner);
				nvgFill(nvg);
				nvgStroke(nvg);
			}
		}
		nvgEndFrame(nvg);
		renderBuffer.end();
		ImageRGBAf img = renderBuffer.getTexture().read();
		FlipVertical(img);
		ConvertImage(img, cacheImage);
		cacheGlyph.reset(new ImageGlyph(cacheImage, AlloyApplicationContext().get(), true));
		cacheDirty = false;
	}
	NeuralLayerRegion::NeuralLayerRegion(const std::string& name, tgr::NeuralLayer* layer,
		const AUnit2D& pos, const AUnit2D& dims, bool resizeable) :
		Composite(name, pos, dims), layer(layer), scale(1.0f) {
		selectionRadius = 2;
		cacheDirty = true;
		const int GLYPH_SIZE = 32;
		cacheImage.resize(layer->width * GLYPH_SIZE, layer->height * GLYPH_SIZE);
		cacheImage.set(RGBA(0,0,0,0));
		renderBuffer.initialize(cacheImage.width, cacheImage.height);
		cacheGlyph = ImageGlyphPtr(new ImageGlyph(cacheImage, AlloyApplicationContext().get(), false));
		cellPadding = pixel2(10, 10);
		if (resizeable) {
			Application::addListener(this);
		}
		setDragEnabled(true);
		setClampDragToParentBounds(false);
		AlloyContext* context = AlloyApplicationContext().get();
		cancelButton = std::shared_ptr<IconButton>(new IconButton(0xf070, CoordPX(2.0f, 2.0f), CoordPX(24, 24), IconType::CIRCLE));
		cancelButton->setOrigin(Origin::TopLeft);
		cancelButton->setRescaleOnHover(false);
		cancelButton->borderColor = MakeColor(AlloyApplicationContext()->theme.LIGHTEST);
		cancelButton->iconColor = MakeColor(AlloyApplicationContext()->theme.LIGHT);
		cancelButton->foregroundColor = MakeColor(COLOR_NONE);
		cancelButton->backgroundColor = MakeColor(COLOR_NONE);
		cancelButton->borderWidth = UnitPX(0.0f);
		cancelButton->onMouseDown = [this](AlloyContext* context, const InputEvent& event) {
			this->setVisible(false);
			if (onHide) {
				onHide();
			}
			return true;
		};
		Composite::add(cancelButton);

		expandButton = std::shared_ptr<IconButton>(new IconButton(0xf0e8, CoordPerPX(0.5f, 1.0f, 0.0f, -22.0f), CoordPX(26.0f, 26.0f), IconType::SQUARE));
		expandButton->setRoundCorners(true);
		expandButton->setVisible(false);
		expandButton->setOrigin(Origin::MiddleCenter);
		expandButton->borderColor = MakeColor(AlloyApplicationContext()->theme.DARK.toDarker(0.75f));
		expandButton->iconColor = MakeColor(AlloyApplicationContext()->theme.DARK);
		expandButton->foregroundColor = MakeColor(COLOR_NONE);
		expandButton->backgroundColor = MakeColor(COLOR_NONE);
		expandButton->borderWidth = UnitPX(0.0f);
		expandButton->onMouseDown = [this](AlloyContext* context, const InputEvent& event) {
			expandButton->setVisible(false);
			if (onExpand) {
				onExpand();
			}
			return true;
		};
		Composite::add(expandButton);
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
		pushScissor(nvg, parent->getCursorBounds());
		float tscale = aly::clamp(std::sqrt(scale),0.25f,1.25f);
		nvgFontSize(nvg, fontSize*tscale);
		nvgFontFaceId(nvg, context->getFontHandle(FontType::Bold));
		nvgTextAlign(nvg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
		drawText(nvg, bounds.position + float2(28.0f, 0.5f*fontSize*(1.0f- tscale)), name, FontStyle::Outline, context->theme.LIGHTER, context->theme.DARK);
		popScissor(nvg);

		pushScissor(nvg, getCursorBounds());
		nvgBeginPath(nvg);
		nvgRoundedRect(nvg, bounds.position.x, bounds.position.y + fontSize + 2.0f, bounds.dimensions.x, bounds.dimensions.y - fontSize - 2.0f - 30.0f, 3.0f);
		if (isFocused()) {
			nvgFillColor(nvg, context->theme.LIGHTEST);
		}
		else {
			nvgFillColor(nvg, context->theme.LIGHTEST.toDarker(0.8f));
		}
		nvgFill(nvg);
		if (expandButton->isVisible()) {
			nvgBeginPath(nvg);
			nvgRoundedRect(nvg, bounds.position.x + bounds.dimensions.x*0.5f - 15.0f, bounds.position.y + bounds.dimensions.y - 40.0f, 30.0f, 30.0f, 5.0f);
			nvgFill(nvg);
		}
		int width = layer->width;
		int height = layer->height;
		float scale = bounds.dimensions.x / width;
		pixel2 origin = bounds.position;
		pixel2 padding = getPadding();
		bounds.position += float2(0.0f, fontSize + 8.0f);
		bounds.dimensions -= padding;
		int bins = layer->bins;
		pixel2 pos = pixel2(-1, -1);
		int2 selected = int2(-1, -1);
		if (bounds.contains(cursorPosition)) {
			pos = (cursorPosition - bounds.position) / bounds.dimensions;
			selected = int2((int)std::floor(width*pos.x), (int)std::floor(height*pos.y));
			
		}
		if (cacheDirty) {
			context->addDeferredTask([this]() {
				drawCache(AlloyApplicationContext().get());
			});
		}
		cacheGlyph->draw(bounds, COLOR_NONE, COLOR_NONE, context);
		float rOuter = 0.5f*scale;
		float rInner = 0.25f*scale;
		float lineWidth = scale*0.01f;
		nvgStrokeWidth(nvg, lineWidth);


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
			if (selected.x != -1&& layer->getFlow()!=nullptr) {
				layer->getFlow()->setSelected(layer);
			}
		}
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				float2 center = float2(bounds.position.x + (i + 0.5f)*scale, bounds.position.y + (j + 0.5f)*scale);
				Neuron& n = (*layer)(i, j);
				if (lineWidth > 0.1f&&selected.x != -1 && std::abs(i - selected.x) <= selectionRadius&&std::abs(j - selected.y) <= selectionRadius) {
					int N = (int)n.getInputWeightSize();
					nvgLineCap(nvg, NVG_ROUND);

					for (int i = 0; i < N; i++) {
						SignalPtr& sig = n.getInput(i);
						float a = 2.0f*i*NVG_PI / N - 0.5f*NVG_PI;
						float cosa = std::cos(a);
						float sina = std::sin(a);
						float sx = center.x + rInner*cosa;
						float sy = center.y + rInner*sina;
						float tw =(rOuter-lineWidth) * clamp(0.5f+0.5f*sig->value,0.0f,1.0f);
						float wx = center.x + tw*cosa;
						float wy = center.y + tw*sina;
						float ex; 
						float ey;
						if (sig->value >= 0) {
							ex = center.x + (rOuter - lineWidth)*cosa;
							ey = center.y + (rOuter - lineWidth)*sina;
						}
						else {
							ex = center.x;
							ey = center.y;
						}
						nvgStrokeColor(nvg, Color(128, 128, 128));
						nvgStrokeWidth(nvg, lineWidth);
						nvgBeginPath(nvg);
						nvgMoveTo(nvg, sx, sy);
						nvgLineTo(nvg, ex, ey);
						nvgStroke(nvg);

						float valSum = 0.0f;
						int sz=(int)sig->mapping.size();
						if (sz > 0) {
							for (Neuron* ne : sig->get(&n)) {
								valSum += ne->normalizedValue();
							}
							nvgStrokeColor(nvg, Color(ColorMapToRGB(clamp(valSum/sz,0.0f,1.0f), ColorMap::RedToBlue)));
						} else {
							nvgStrokeColor(nvg, Color(220, 220, 220));
						}
						nvgBeginPath(nvg);
						if (sig->value >= 0) {
							nvgMoveTo(nvg, sx + lineWidth*cosa, sy + lineWidth*sina);
						}
						else {
							nvgMoveTo(nvg, sx - lineWidth*cosa, sy - lineWidth*sina);

						}
						nvgLineTo(nvg, wx, wy);
						nvgStrokeWidth(nvg, 3 * lineWidth);
						nvgStroke(nvg);

					}

					nvgStrokeWidth(nvg, 2.0f* lineWidth);
					nvgStrokeColor(nvg, Color(200, 200, 200));
					nvgBeginPath(nvg);
					nvgCircle(nvg, center.x, center.y, rOuter);
					nvgStroke(nvg);
					nvgBeginPath(nvg);
					nvgCircle(nvg, center.x, center.y, rInner);
					nvgStroke(nvg);
				}
				else if (n.active) {
					nvgStrokeWidth(nvg, 2.0f* lineWidth);
					nvgStrokeColor(nvg, Color(200, 200, 200));
					nvgBeginPath(nvg);
					nvgCircle(nvg, center.x, center.y, rInner);
					nvgStroke(nvg);
				}

			}
		}

		popScissor(context->nvgContext);
		
		Composite::draw(context);

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
		bool ret = (activeList.size() > 0 || lastSelected.x != -1);
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
	float  NeuralLayerRegion::setSize(float w) {
		AlloyContext* context = AlloyApplicationContext().get();
		pixel2 padding = getPadding();
		float aspectRatio = layer->width / (float)layer->height;
		pixel2 newBounds = aly::round(pixel2(w, w / aspectRatio) + padding);
		scale = 1.0f;
		bounds.dimensions = newBounds;
		dimensions = CoordPX(bounds.dimensions);
		setDragOffset(pixel2(0, 0));
		return newBounds.y;
	}
	void  NeuralLayerRegion::setScale(float s, pixel2 cursor) {
		AlloyContext* context = AlloyApplicationContext().get();
		box2px bounds = getBounds(false);
		float lastScale = scale;
		scale = clamp(scale * (1.0f + s * 0.1f), 0.1f, 10.0f);
		float scaling = scale / lastScale;
		pixel2 padding = getPadding();
		float aspectRatio = layer->height / (float)layer->width;
		pixel2 newBounds = aly::round(pixel2(bounds.dimensions.x, bounds.dimensions.x*aspectRatio)*scaling + padding);
		pixel2 relPos = (cursor - bounds.position) / bounds.dimensions;
		pixel2 newPos = cursor - relPos*newBounds;
		bounds.position = aly::round(newPos);
		bounds.dimensions = newBounds;
		setDragOffset(pixel2(0, 0));
		position = CoordPX(bounds.position - parent->getBoundsPosition());
		dimensions = CoordPX(bounds.dimensions);
	}
}