#include "TigerApp.h"
#include "AlloyParameterPane.h"
#include "AlloyExpandTree.h"
#include "NeuralFilter.h"
#include "ConvolutionFilter.h"
#include "AveragePoolFilter.h"
#include "AlloyDrawUtil.h"
using namespace aly;
using namespace tgr;
TigerApp::TigerApp() :
	Application(1800, 800, "Tiger Artifical Intelligence",true), selectedLayer(nullptr){
}
bool TigerApp::init(Composite& rootNode) {

	parametersDirty = true;
	frameBuffersDirty = true;

	BorderCompositePtr layout = BorderCompositePtr(new BorderComposite("UI Layout", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), true));
	ParameterPanePtr controls = ParameterPanePtr(new ParameterPane("Controls", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	expandTree = ExpandTreePtr(new ExpandTree("Expand Tree", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	BorderCompositePtr controlLayout = BorderCompositePtr(new BorderComposite("Control Layout", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), true));
	controls->onChange = [this](const std::string& label, const AnyInterface& value) {
		parametersDirty = true;
	};

	initialize();
	controls->setAlwaysShowVerticalScrollBar(false);
	controls->setScrollEnabled(false);
	controls->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controls->borderColor = MakeColor(getContext()->theme.DARK);
	controls->borderWidth = UnitPX(1.0f);

	controlLayout->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controlLayout->borderWidth = UnitPX(0.0f);
	renderRegion = CompositePtr(new Composite("View", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	layout->setWest(controlLayout, UnitPX(400.0f));
	layout->setEast(expandTree, UnitPX(400.0f));
	controlLayout->setCenter(controls);
	layout->setCenter(renderRegion);

	CompositePtr infoComposite = CompositePtr(new Composite("Info", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	infoComposite->backgroundColor = MakeColor(getContext()->theme.DARKER);
	infoComposite->borderColor = MakeColor(getContext()->theme.DARK);
	infoComposite->borderWidth = UnitPX(0.0f);
	playButton = IconButtonPtr(new IconButton(0xf144, CoordPerPX(0.5f, 0.5f, -35.0f, -35.0f), CoordPX(70.0f, 70.0f)));
	stopButton = IconButtonPtr(new IconButton(0xf28d, CoordPerPX(0.5f, 0.5f, -35.0f, -35.0f), CoordPX(70.0f, 70.0f)));
	playButton->borderWidth = UnitPX(0.0f);
	stopButton->borderWidth = UnitPX(0.0f);
	playButton->backgroundColor = MakeColor(getContext()->theme.DARKER);
	stopButton->backgroundColor = MakeColor(getContext()->theme.DARKER);
	playButton->foregroundColor = MakeColor(0, 0, 0, 0);
	stopButton->foregroundColor = MakeColor(0, 0, 0, 0);
	playButton->iconColor = MakeColor(getContext()->theme.LIGHTER);
	stopButton->iconColor = MakeColor(getContext()->theme.LIGHTER);
	playButton->borderColor = MakeColor(getContext()->theme.LIGHTEST);
	stopButton->borderColor = MakeColor(getContext()->theme.LIGHTEST);
	playButton->onMouseDown = [this](AlloyContext* context, const InputEvent& e) {
		if (e.button == GLFW_MOUSE_BUTTON_LEFT) {
			stopButton->setVisible(true);
			playButton->setVisible(false);
			
			return true;
		}
		return false;
	};
	stopButton->onMouseDown = [this](AlloyContext* context, const InputEvent& e) {
		if (e.button == GLFW_MOUSE_BUTTON_LEFT) {
			stopButton->setVisible(false);
			playButton->setVisible(true);
			running = false;
			return true;
		}
		return false;
	};
	stopButton->setVisible(false);
	infoComposite->add(playButton);
	infoComposite->add(stopButton);
	controlLayout->setSouth(infoComposite, UnitPX(80.0f));
	rootNode.add(layout);

	/*
	CompositePtr viewRegion = CompositePtr(new Composite("View", CoordPX(0.0f, 0.0f), CoordPerPX(1.0f, 1.0f, 0.0f, -80.0f)));

	float downScale = std::min((getContext()->getScreenWidth() - 350.0f) / img.width, (getContext()->getScreenHeight() - 80.0f) / img.height);

	
	ImageGlyphPtr imageGlyph = AlloyApplicationContext()->createImageGlyph(img, false);
	
	GlyphRegionPtr glyphRegion = GlyphRegionPtr(new GlyphRegion("Image Region", imageGlyph, CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	glyphRegion->setAspectRule(AspectRule::Unspecified);
	glyphRegion->foregroundColor = MakeColor(COLOR_NONE);
	glyphRegion->backgroundColor = MakeColor(COLOR_NONE);
	glyphRegion->borderColor = MakeColor(COLOR_NONE);


	*/
	
	dragIconPane =
		DrawPtr(
			new Draw("Drag Icon", CoordPX(0.0f, 0.0f), CoordPX(50.0f, 50.0f),
				[this](AlloyContext* context, const box2px& bounds) {
		static std::string nodeStr = CodePointToUTF8(0xf20e);
		static std::string plusStr = CodePointToUTF8(0xf067);
		if (selectedLayer == nullptr)return;
		NVGcontext* nvg = context->nvgContext;
		nvgFontFaceId(nvg, context->getFontHandle(FontType::Icon));
		nvgFontSize(nvg, 50.0f);
		const float th = 20.0f;
		std::string modName = selectedLayer->getName();

		nvgTextAlign(nvg, NVG_ALIGN_TOP | NVG_ALIGN_LEFT);
		if (this->overTarget) {
			nvgFillColor(nvg, context->theme.DARKER);
			nvgBeginPath(nvg);
			nvgEllipse(nvg, bounds.position.x + bounds.dimensions.x*0.5f, bounds.position.y + bounds.dimensions.y*0.5f, bounds.dimensions.x*0.5f, bounds.dimensions.y*0.5f);
			nvgFill(nvg);
			drawText(nvg, bounds.position, nodeStr, FontStyle::Normal, context->theme.LIGHTER, context->theme.DARKER);
			nvgFontSize(nvg, 18.0f);
			nvgBeginPath(nvg);
			nvgFillColor(nvg, context->theme.DARKER);
			nvgStrokeColor(nvg, context->theme.LIGHT);
			nvgStrokeWidth(nvg, 2.0f);
			nvgEllipse(nvg, bounds.position.x + 45.0f, bounds.position.y + 5.0f, 10.0f, 10.0f);
			nvgStroke(nvg);
			nvgFill(nvg);
			nvgTextAlign(nvg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
			drawText(nvg, bounds.position + pixel2(45.0f, 6.0f), plusStr, FontStyle::Normal, context->theme.LIGHTEST, context->theme.DARKEST);
			nvgFontFaceId(nvg, context->getFontHandle(FontType::Normal));
			nvgFontSize(nvg, th);
			float tw = nvgTextBounds(nvg, 0, 0, modName.c_str(), nullptr, nullptr) + 4.0f;
			nvgFillColor(nvg, context->theme.DARKER);
			nvgBeginPath(nvg);
			nvgRoundedRect(nvg, bounds.position.x + bounds.dimensions.x*0.5f - tw*0.5f, bounds.position.y + bounds.dimensions.y, tw, th, 5.0f);
			nvgFill(nvg);
			nvgTextAlign(nvg, NVG_ALIGN_MIDDLE | NVG_ALIGN_CENTER);
			drawText(nvg, bounds.position + pixel2(bounds.dimensions.x*0.5f, bounds.dimensions.y + 0.5f*th), modName, FontStyle::Normal, context->theme.LIGHTEST, context->theme.DARKER);
		}
		else {
			nvgFillColor(nvg, context->theme.DARKER.toSemiTransparent(0.5f));
			nvgBeginPath(nvg);
			nvgEllipse(nvg, bounds.position.x + bounds.dimensions.x*0.5f, bounds.position.y + bounds.dimensions.y*0.5f, bounds.dimensions.x*0.5f, bounds.dimensions.y*0.5f);
			nvgFill(nvg);
			drawText(nvg, bounds.position, nodeStr, FontStyle::Normal, context->theme.LIGHTER.toSemiTransparent(0.5f), context->theme.DARKEST);
			nvgFontFaceId(nvg, context->getFontHandle(FontType::Normal));
			nvgFontSize(nvg, th);
			float tw = nvgTextBounds(nvg, 0, 0, modName.c_str(), nullptr, nullptr) + 4.0f;
			nvgFillColor(nvg, context->theme.DARKER.toSemiTransparent(0.5f));
			nvgBeginPath(nvg);
			nvgRoundedRect(nvg, bounds.position.x + bounds.dimensions.x*0.5f - tw*0.5f, bounds.position.y + bounds.dimensions.y, tw, th, 5.0f);
			nvgFill(nvg);
			nvgTextAlign(nvg, NVG_ALIGN_MIDDLE | NVG_ALIGN_CENTER);
			drawText(nvg, bounds.position + pixel2(bounds.dimensions.x*0.5f, bounds.dimensions.y + 0.5f*th), modName, FontStyle::Normal, context->theme.LIGHTEST.toSemiTransparent(0.5f), context->theme.DARKER);
		}
	}));
	dragIconPane->setIgnoreCursorEvents(true);
	dragIconPane->setVisible(false);
	CompositePtr glass = AlloyApplicationContext()->getGlassPane();
	glass->add(dragIconPane);
	glass->backgroundColor = MakeColor(0, 0, 0, 0);

	renderRegion->backgroundColor = MakeColor(getContext()->theme.DARKER);
	renderRegion->borderColor = MakeColor(getContext()->theme.DARK);
	renderRegion->borderWidth = UnitPX(1.0f);
	Application::addListener(&rootNode);
	rootNode.onEvent = [this](AlloyContext* context, const InputEvent& event) {
		return onEventHandler(context, event);
	};
	return true;
}
void TigerApp::setSelectedLayer(tgr::NeuralLayer* layer) {
	selectedLayer = layer;
}
bool TigerApp::onEventHandler(AlloyContext* context, const aly::InputEvent& e) {
	if (selectedLayer != nullptr) {
		dragIconPane->setDragOffset(e.cursor, pixel2(20.0f, 20.0f));
		AlloyApplicationContext()->getGlassPane()->setVisible(true);
		if (!dragIconPane->isVisible()) {
			dragIconPane->setVisible(true);
			AlloyApplicationContext()->requestPack();
		}
		if (renderRegion->isVisible() && renderRegion->getBounds().contains(e.cursor)) {
			overTarget = true;
		}
		else {
			overTarget = false;
		}
	}
	if (dragIconPane->isVisible() && e.type == InputType::MouseButton && e.isUp()) {
		if (selectedLayer!= nullptr) {
			dragIconPane->setVisible(false);
			AlloyApplicationContext()->getGlassPane()->setVisible(false);
			if (overTarget) {
				selectedLayer->appendTo(renderRegion,e.cursor);
			}
		}
		selectedLayer=nullptr;
		overTarget = false;
	}
	return false;
}
void TigerApp::initialize() {
	ConvolutionFilterPtr conv1(new ConvolutionFilter(this, 32, 16, 5, 6));
	sys.add(conv1);
	AveragePoolFilterPtr avg1(new AveragePoolFilter(this, conv1->getOutputLayers(), 2));
	sys.add(avg1);
	for (int i = 0; i < avg1->getOutputSize(); i++) {
		ConvolutionFilterPtr conv2(new ConvolutionFilter(this,avg1->getOutputLayer(i),  5, 16));
		sys.add(conv2);
	}
	sys.initialize(expandTree);
}
void TigerApp::draw(AlloyContext* context) {
}