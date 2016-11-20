#include "TigerApp.h"
#include "AlloyParameterPane.h"
#include "AlloyExpandTree.h"
using namespace aly;
using namespace tgr;
TigerApp::TigerApp() :
	Application(1800, 800, "Tiger Artifical Intelligence",true), GLYPH_SCALE(8.0f){
}

bool TigerApp::init(Composite& rootNode) {
	initialize();
	parametersDirty = true;
	frameBuffersDirty = true;

	BorderCompositePtr layout = BorderCompositePtr(new BorderComposite("UI Layout", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), true));
	ParameterPanePtr controls = ParameterPanePtr(new ParameterPane("Controls", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	ExpandTreePtr tree = ExpandTreePtr(new ExpandTree("Expand Tree", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	BorderCompositePtr controlLayout = BorderCompositePtr(new BorderComposite("Control Layout", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), true));
	controls->onChange = [this](const std::string& label, const AnyInterface& value) {
		parametersDirty = true;
	};

	controls->setAlwaysShowVerticalScrollBar(false);
	controls->setScrollEnabled(false);
	controls->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controls->borderColor = MakeColor(getContext()->theme.DARK);
	controls->borderWidth = UnitPX(1.0f);

	controlLayout->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controlLayout->borderWidth = UnitPX(0.0f);
	CompositePtr renderRegion = CompositePtr(new Composite("View", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	layout->setWest(controlLayout, UnitPX(400.0f));
	layout->setEast(tree, UnitPX(400.0f));
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

	float2 dims = GLYPH_SCALE*float2(currentLayer->dimensions());
	float scale = (getContext()->getScreenWidth()-840)/dims.x;
	resizeableRegion = AdjustableCompositePtr(new AdjustableComposite("Image", CoordPerPX(0.5, 0.5, -dims.x*0.5f*scale, -dims.y*0.5f*scale),CoordPX(dims.x*scale,dims.y*scale)));
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
	Application::addListener(resizeableRegion.get());
	DrawPtr drawContour = DrawPtr(new Draw("Contour Draw", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), 
		[this](AlloyContext* context, const box2px& bounds) {

		currentLayer->draw(context, bounds);
	}));
	drawContour->onScroll = [this](AlloyContext* context, const InputEvent& event)
	{
		box2px bounds = resizeableRegion->getBounds(false);
		pixel scaling = (pixel)(1 - 0.1f*event.scroll.y);
		pixel2 newBounds = bounds.dimensions*scaling;
		pixel2 cursor = context->cursorPosition;
		pixel2 relPos = (cursor - bounds.position) / bounds.dimensions;
		pixel2 newPos = cursor - relPos*newBounds;
		bounds.position = newPos;
		bounds.dimensions = newBounds;
		resizeableRegion->setDragOffset(pixel2(0, 0));
		resizeableRegion->position = CoordPX(bounds.position - resizeableRegion->parent->getBoundsPosition());
		resizeableRegion->dimensions = CoordPX(bounds.dimensions);
		float2 dims = GLYPH_SCALE*float2(currentLayer->dimensions());
		cursor = aly::clamp(dims*(event.cursor - bounds.position) / bounds.dimensions, float2(0.0f), dims);
		context->requestPack();
		return true;
	};
	resizeableRegion->setAspectRatio(currentLayer->getAspect());
	resizeableRegion->setAspectRule(AspectRule::FixedHeight);
	resizeableRegion->setDragEnabled(true);
	resizeableRegion->setClampDragToParentBounds(false);
	resizeableRegion->borderWidth = UnitPX(2.0f);
	resizeableRegion->borderColor = MakeColor(AlloyApplicationContext()->theme.LIGHTER);

	resizeableRegion->add(drawContour);

	renderRegion->backgroundColor = MakeColor(getContext()->theme.DARKER);
	renderRegion->borderColor = MakeColor(getContext()->theme.DARK);
	renderRegion->borderWidth = UnitPX(1.0f);
	renderRegion->add(resizeableRegion);
	return true;
}
void TigerApp::initialize() {
	currentLayer = NeuronLayerPtr(new NeuronLayer(32, 16, 8, 0));
	for (Neuron& n : *currentLayer) {
		n.value = RandomUniform(0.0f, 1.0f);


	}
}
void TigerApp::draw(AlloyContext* context) {
}