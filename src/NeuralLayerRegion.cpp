#include "NeuralLayerRegion.h"
#include "AlloyDrawUtil.h"
#include "AlloyApplication.h"
#include "NeuralLayer.h"
using namespace tgr;
namespace aly {
	NeuralLayerRegion::NeuralLayerRegion(const std::string& name, tgr::NeuralLayer* layer,
		const AUnit2D& pos, const AUnit2D& dims, bool resizeable) :
		Composite(name, pos, dims),layer(layer), resizing(false), winPos(
			WindowPosition::Outside), resizeable(resizeable) {

		windowInitialBounds.dimensions = float2(-1, -1);
		cellPadding = pixel2(10, 10);
		if (resizeable) {
			Application::addListener(this);
		}
		setAspectRatio(layer->getAspect());
		setAspectRule(AspectRule::FixedHeight);
		setDragEnabled(true);
		setClampDragToParentBounds(false);
		borderWidth = UnitPX(2.0f);
		borderColor = MakeColor(AlloyDefaultContext()->theme.LIGHTER);
	}
	void NeuralLayerRegion::drawNeurons(AlloyContext* context) {
		const aly::box2px bounds = getBounds();
		int width = layer->width;
		int height = layer->height;
		float scale = bounds.dimensions.x / width;
		NVGcontext* nvg = context->nvgContext;
		float rOuter = 0.5f*scale;
		float rInner = 0.25f*scale;
		float lineWidth = scale*0.01f;
		nvgStrokeWidth(nvg, lineWidth);
		int bins = layer->bins;
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				float2 center = float2(bounds.position.x + (i + 0.5f)*scale, bounds.position.y + (j + 0.5f)*scale);
				nvgFillColor(nvg, Color(92, 92, 92));
				nvgBeginPath(nvg);
				nvgCircle(nvg, center.x, center.y, scale*0.5f);
				nvgFill(nvg);
				Neuron& n = (*layer)(i, j);
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
				if (lineWidth > 0.5f) {
					int N = (int)n.getInputSize();
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
	}
	void NeuralLayerRegion::draw(AlloyContext* context) {
		pushScissor(context->nvgContext, getCursorBounds());
		drawNeurons(context);
		Composite::draw(context);
		popScissor(context->nvgContext);
		if (windowInitialBounds.dimensions.x < 0
			|| windowInitialBounds.dimensions.y < 0) {
			windowInitialBounds = getBounds(false);
			windowInitialBounds.position -= this->getDragOffset();
		}
		if (context->getCursor() == nullptr) {
			switch (winPos) {
			case WindowPosition::Center:
				if ((dragButton == GLFW_MOUSE_BUTTON_LEFT&&context->isLeftMouseButtonDown()) ||
					(dragButton == GLFW_MOUSE_BUTTON_RIGHT&&context->isRightMouseButtonDown())) {
					context->setCursor(&Cursor::Position);
				}
				break;
			case WindowPosition::Top:
			case WindowPosition::Bottom:
				context->setCursor(&Cursor::Vertical);
				break;
			case WindowPosition::Left:
			case WindowPosition::Right:
				context->setCursor(&Cursor::Horizontal);
				break;
			case WindowPosition::TopLeft:
			case WindowPosition::BottomRight:
				context->setCursor(&Cursor::SlantDown);
				break;
			case WindowPosition::BottomLeft:
			case WindowPosition::TopRight:
				context->setCursor(&Cursor::SlantUp);
				break;
			default:
				break;
			}

		}
	}
	bool NeuralLayerRegion::onEventHandler(AlloyContext* context,const InputEvent& e) {
		const float GLYPH_SCALE = 8.0f;
		if (Composite::onEventHandler(context, e))
			return true;
		if (resizeable) {
			bool over = context->isMouseOver(this, true);
			if (e.type == InputType::MouseButton
				&& e.button == GLFW_MOUSE_BUTTON_LEFT && e.isDown() && over) {
				dynamic_cast<Composite*>(this->parent)->putLast(this);
			}
			else if (!context->isLeftMouseButtonDown()) {
				resizing = false;
			}
			if (e.type == InputType::Scroll&&context->isMouseOver(this,true)) {
				box2px bounds = getBounds(false);
				pixel scaling = (pixel)(1 - 0.1f*e.scroll.y);
				pixel2 newBounds = bounds.dimensions*scaling;
				pixel2 cursor = context->cursorPosition;
				pixel2 relPos = (cursor - bounds.position) / bounds.dimensions;
				pixel2 newPos = cursor - relPos*newBounds;
				bounds.position = newPos;
				bounds.dimensions = newBounds;
				setDragOffset(pixel2(0, 0));
				position = CoordPX(bounds.position - parent->getBoundsPosition());
				dimensions = CoordPX(bounds.dimensions);
				float2 dims = GLYPH_SCALE*float2(layer->dimensions());
				cursor = aly::clamp(dims*(e.cursor - bounds.position) / bounds.dimensions, float2(0.0f), dims);
				context->requestPack();
				return true;
			}
			if (e.type == InputType::Cursor) {
				if (!resizing) {
					if (over) {
						box2px bounds = getBounds();
						winPos = WindowPosition::Center;
						if (e.cursor.x <= bounds.position.x + cellPadding.x) {
							if (e.cursor.y <= bounds.position.y + cellPadding.y) {
								winPos = WindowPosition::TopLeft;
							}
							else if (e.cursor.y
								>= bounds.position.y + bounds.dimensions.y
								- cellPadding.y) {
								winPos = WindowPosition::BottomLeft;
							}
							else {
								winPos = WindowPosition::Left;
							}
						}
						else if (e.cursor.x
							>= bounds.position.x + bounds.dimensions.x
							- cellPadding.x) {
							if (e.cursor.y <= bounds.position.y + cellPadding.y) {
								winPos = WindowPosition::TopRight;
							}
							else if (e.cursor.y
								>= bounds.position.y + bounds.dimensions.y
								- cellPadding.y) {
								winPos = WindowPosition::BottomRight;
							}
							else {
								winPos = WindowPosition::Right;
							}
						}
						else if (e.cursor.y
							<= bounds.position.y + cellPadding.y) {
							winPos = WindowPosition::Top;
						}
						else if (e.cursor.y
							>= bounds.position.y + bounds.dimensions.y
							- cellPadding.y) {
							winPos = WindowPosition::Bottom;
						}
					}
					else {
						winPos = WindowPosition::Outside;
					}
				}
			}
			if (over && e.type == InputType::MouseButton
				&& e.button == GLFW_MOUSE_BUTTON_LEFT && e.isDown()
				&& winPos != WindowPosition::Center
				&& winPos != WindowPosition::Outside) {
				if (!resizing) {
					cursorDownPosition = e.cursor;
					windowInitialBounds = getBounds(false);
					windowInitialBounds.position -= this->getDragOffset();
				}
				resizing = true;

			}
			if (resizing && e.type == InputType::Cursor) {
				float2 minPt = windowInitialBounds.min();
				float2 maxPt = windowInitialBounds.max();
				pixel2 cursor =
					box2px(pixel2(0.0f, 0.0f),
						pixel2(context->getScreenWidth() - 1.0f,
							context->getScreenHeight() - 1.0f)).clamp(e.cursor);
				switch (winPos) {
				case WindowPosition::Top:
					minPt.y += cursor.y - cursorDownPosition.y;
					break;
				case WindowPosition::Bottom:
					maxPt.y += cursor.y - cursorDownPosition.y;
					break;
				case WindowPosition::Left:
					minPt.x += cursor.x - cursorDownPosition.x;
					break;
				case WindowPosition::Right:
					maxPt.x += cursor.x - cursorDownPosition.x;
					break;
				case WindowPosition::TopLeft:
					minPt.x += cursor.x - cursorDownPosition.x;
					minPt.y += cursor.y - cursorDownPosition.y;
					break;
				case WindowPosition::BottomRight:
					maxPt.x += cursor.x - cursorDownPosition.x;
					maxPt.y += cursor.y - cursorDownPosition.y;
					break;
				case WindowPosition::BottomLeft:
					minPt.x += cursor.x - cursorDownPosition.x;
					maxPt.y += cursor.y - cursorDownPosition.y;
					break;
				case WindowPosition::TopRight:
					maxPt.x += cursor.x - cursorDownPosition.x;
					minPt.y += cursor.y - cursorDownPosition.y;

					break;
				default:
					break;
				}
				box2px newBounds(aly::min(minPt, maxPt),
					aly::max(maxPt - minPt, float2(50, 50)));
				pixel2 d = newBounds.dimensions;
				pixel2 d1, d2;
				if (aspectRule != AspectRule::Unspecified) {
					switch (winPos) {
					case WindowPosition::Left:
					case WindowPosition::Right:
						newBounds.dimensions = pixel2(d.x,
							d.x / (float)aspectRatio);
						break;
					case WindowPosition::Top:
					case WindowPosition::Bottom:
						newBounds.dimensions = pixel2(d.y * (float)aspectRatio,
							d.y);
						break;
					case WindowPosition::TopLeft:
					case WindowPosition::TopRight:
					case WindowPosition::BottomLeft:
					case WindowPosition::BottomRight:
						d1 = pixel2(d.x, d.x / (float)aspectRatio);
						d2 = pixel2(d.y * (float)aspectRatio, d.y);
						if (d1.x * d1.y > d2.x * d2.y) {
							newBounds.dimensions = d1;
						}
						else {
							newBounds.dimensions = d2;
						}
						break;
					default:
						break;
					}
				}

				if (clampToParentBounds) {
					pixel2 offset = getDragOffset();
					newBounds.position += offset;
					newBounds.clamp(parent->getBounds());
					newBounds.position -= offset;
				}
				this->position = CoordPX(newBounds.position - parent->getBoundsPosition());
				this->dimensions = CoordPX(newBounds.dimensions);
				if (onResize) {
					onResize(this, newBounds);
				}
				context->requestPack();
			}
			return false;
		}
		else {
			return false;
		}
	}
}