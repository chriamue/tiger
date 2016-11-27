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
#include "TigerApp.h"
#include "AlloyParameterPane.h"
#include "AlloyExpandTree.h"
#include "NeuralFilter.h"
#include "ConvolutionFilter.h"
#include "AveragePoolFilter.h"
#include "AlloyDrawUtil.h"
#include "MNIST.h"
using namespace aly;
using namespace tgr;
TigerApp::TigerApp() :
	Application(1800, 800, "Tiger Machine",true), selectedLayer(nullptr){
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

	epochs = Integer(12);
	iterationsPerEpoch = Integer(100);
	learningRateInitial = Float(0.8f);
	learningRateDelta = Float(0.5);

	controls->setAlwaysShowVerticalScrollBar(false);
	controls->setScrollEnabled(false);
	controls->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controls->borderColor = MakeColor(getContext()->theme.DARK);
	controls->borderWidth = UnitPX(1.0f);
	controls->addGroup("Data", true);
	
	trainFile=getFullPath("data/train-images.idx3-ubyte");
	trainLabelFile = getFullPath("data/train-labels.idx1-ubyte");
	controls->addFileField("Train Images",trainFile);
	controls->addFileField("Train Labels", trainLabelFile);

	controls->addFileField("Evaluation Images", evalFile);
	controls->addFileField("Evaluation Labels", evalLabelFile);
	initialize();
	controls->addGroup("Training",true);
	controls->addNumberField("Epochs",epochs);
	controls->addNumberField("Iterations",iterationsPerEpoch);
	controls->addNumberField("Learning Rate", learningRateInitial,Float(0.0f), Float(1.0f));
	controls->addNumberField("Learning Delta", learningRateDelta, Float(0.0f), Float(1.0f));


	controlLayout->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controlLayout->borderWidth = UnitPX(0.0f);
	renderRegion = NeuralFlowPanePtr(new NeuralFlowPane("View", CoordPX(0.0f, 0.0f), CoordPerPX(1.0f, 1.0f, 0.0f, -80.0f)));
	layout->setWest(controlLayout, UnitPX(400.0f));
	layout->setEast(expandTree, UnitPX(400.0f));
	controlLayout->setCenter(controls);
	
	timelineSlider = TimelineSliderPtr(
		new TimelineSlider("Timeline", CoordPerPX(0.0f, 1.0f, 0.0f, -80.0f), CoordPerPX(1.0f, 0.0f, 0.0f, 80.0f), Integer(0), Integer(0), Integer(0)));
	CompositePtr viewRegion = CompositePtr(new Composite("View", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	timelineSlider->backgroundColor = MakeColor(AlloyApplicationContext()->theme.DARKER);
	timelineSlider->borderColor = MakeColor(AlloyApplicationContext()->theme.DARK);
	timelineSlider->borderWidth = UnitPX(0.0f);
	timelineSlider->onChangeEvent = [this](const Number& timeValue, const Number& lowerValue, const Number& upperValue) {

	};
	timelineSlider->setMajorTick(100);
	timelineSlider->setMinorTick(10);
	timelineSlider->setLowerValue(0);
	timelineSlider->setUpperValue(0);
	timelineSlider->setMaxValue(1000);
	timelineSlider->setVisible(true);
	timelineSlider->setModifiable(false);
	viewRegion->add(renderRegion);
	viewRegion->add(timelineSlider);

	
	layout->setCenter(viewRegion);

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
	
	dragIconPane =
		DrawPtr(
			new Draw("Drag Icon", CoordPX(0.0f, 0.0f), CoordPX(50.0f, 50.0f),
				[this](AlloyContext* context, const box2px& bounds) {
		static std::string nodeStr = CodePointToUTF8(0xf20e);
		static std::string plusStr = CodePointToUTF8(0xf067);
		if (selectedLayer == nullptr)return;
		NVGcontext* nvg = context->nvgContext;
		nvgFontFaceId(nvg, context->getFontHandle(FontType::Icon));
		nvgFontSize(nvg, 40.0f);
		const float th = 20.0f;
		std::string modName = selectedLayer->getName();

		nvgTextAlign(nvg, NVG_ALIGN_TOP | NVG_ALIGN_LEFT);
		if (this->overTarget) {
			drawText(nvg, bounds.position+float2(2.0f,6.0f), nodeStr, FontStyle::Glow, context->theme.LIGHTER, context->theme.DARKEST);
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
			drawText(nvg, bounds.position + float2(2.0f, 6.0f), nodeStr, FontStyle::Glow, context->theme.LIGHTER, context->theme.DARKEST);
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

	renderRegion->backgroundColor = MakeColor(getContext()->theme.DARK);
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
				renderRegion->add(selectedLayer, e.cursor);
				AlloyApplicationContext()->requestPack();
			}
		}
		selectedLayer=nullptr;
		overTarget = false;
	}
	return false;
}
void TigerApp::initialize() {
	std::vector<aly::Image1f> images;
	std::vector<uint8_t> labels;
	parse_mnist_images(trainFile,images);
	parse_mnist_labels(trainLabelFile, labels);

	int K = 4;
	if (images.size() > 0) {
		const aly::Image1f& ref = images[RandomUniform(0,(int)images.size()-1)];
		ConvolutionFilterPtr conv1(new ConvolutionFilter(this,ref.width, ref.height, 5, 6));
		sys.add(conv1);
		for (int i = 0; i < ref.width; i++) {
			for (int j = 0; j < ref.height; j++) {
				sys.addInput(i, j, conv1->getInputLayer(0), ref(i, j).x);
			}
		}
		std::vector<NeuralLayerPtr> all;
		for (int i = 0; i < conv1->getOutputSize(); i++) {
			AveragePoolFilterPtr avg1(new AveragePoolFilter(this, conv1->getOutputLayer(i), 2));
			avg1->setName(MakeString()<<"First Sub-Sample [" << i << "]");
			sys.add(avg1);
			all.push_back(avg1->getOutputLayer(0));
		}
		for (int i = 0; i < all.size(); i++) {
			std::vector<NeuralLayerPtr> in;
			for (int k = 0; k < K; k++) {
				in.push_back(all[(k + i) % all.size()]);
			}
			ConvolutionFilterPtr conv2(new ConvolutionFilter(this, in, 3, 4));
			conv2->setName(MakeString() << "Feature [" << i << "]");
			sys.add(conv2);
			
			for (int i = 0; i < conv2->getOutputSize(); i++) {
				AveragePoolFilterPtr avg2(new AveragePoolFilter(this, conv2->getOutputLayer(i), 2));
				avg2->setName(MakeString() << "Second Sub-Sample [" << i << "]");
				sys.add(avg2);
			}
			
		}

		sys.initialize(expandTree);
		sys.evaluate();
	}
}
void TigerApp::draw(AlloyContext* context) {
}