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
#include "FullyConnectedFilter.h"
#include "MNIST.h"
using namespace aly;
using namespace tgr;
#define O true
#define X false
static const bool MNIST_TABLE[] = {
	O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
	O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
	O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
	X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
	X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
	X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X
TigerApp::TigerApp(int example) :
	Application(1800, 800, "Tiger Machine",true),exampleIndex(example), selectedLayer(nullptr){
}
void TigerApp::setSampleIndex(int idx){
	bool dirty = (sampleIndex.toInteger() != idx);
	sampleIndex.setValue(idx);
	valueRegion->setNumberValue(sampleIndex);
	sys->getInput()->set(trainInputData[idx]);
	sys->evaluate();
}
void TigerApp::setSampleRange(int mn, int mx){
	minIndex = mn;
	maxIndex = mx;
	tweenRegion->setMinValue(Integer(mn));
	tweenRegion->setMaxValue(Integer(mx));
	sampleIndex.setValue(aly::clamp(sampleIndex.toInteger(), mn, mx));
	tweenRegion->setValue(sampleIndex.toDouble());
}
bool TigerApp::init(Composite& rootNode) {

	parametersDirty = true;
	frameBuffersDirty = true;

	CompositePtr toolbar = CompositePtr(new Composite("Toolbar",CoordPX(0.0f,0.0f),CoordPerPX(1.0f,0.0f,0.0f,40.0f)));

	BorderCompositePtr layout = BorderCompositePtr(new BorderComposite("UI Layout", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), true));
	ParameterPanePtr controls = ParameterPanePtr(new ParameterPane("Controls", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	expandTree = ExpandTreePtr(new ExpandTree("Expand Tree", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	expandTree->backgroundColor = MakeColor(getContext()->theme.DARKER);
	expandTree->borderColor = MakeColor(getContext()->theme.DARK);
	expandTree->borderWidth = UnitPX(1.0f);

	BorderCompositePtr controlLayout = BorderCompositePtr(new BorderComposite("Control Layout", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), true));
	controls->onChange = [this](const std::string& label, const AnyInterface& value) {
		parametersDirty = true;
	};

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

	flowRegion = NeuralFlowPanePtr(new NeuralFlowPane("View", CoordPX(0.0f, 40.0f), CoordPerPX(1.0f, 1.0f, 0.0f, -120.0f)));

	toolbar->backgroundColor = MakeColor(getContext()->theme.DARKER);
	sampleIndex = Integer(0);
	const float numAspect = 2.0f;
	const float entryHeight = 40.0f;
	const float aspect = 6.0f;
	TextLabelPtr labelRegion = TextLabelPtr(new TextLabel("Example:", CoordPX(0.0f, 0.0f), CoordPX(105.0f,entryHeight)));
	
	tweenRegion = HorizontalSliderPtr(new HorizontalSlider("sample tween", CoordPX(0.0f,0.0f), CoordPX(aspect * entryHeight, entryHeight),false,Integer(minIndex), Integer(maxIndex), sampleIndex));
	valueRegion = ModifiableNumberPtr(new ModifiableNumber("sample field", CoordPX(0.0f, 0.0f), CoordPX(numAspect * entryHeight, entryHeight), sampleIndex.type()));
	valueRegion->setAlignment(HorizontalAlignment::Center, VerticalAlignment::Middle);
	valueRegion->fontSize = UnitPX(30.0f);
	labelRegion->textColor = MakeColor(AlloyDefaultContext()->theme.LIGHTER);
	labelRegion->setAlignment(HorizontalAlignment::Left, VerticalAlignment::Middle);
	labelRegion->fontSize=UnitPX(30.0f);
	valueRegion->setNumberValue(sampleIndex);
	valueRegion->onTextEntered = [this](NumberField* field) {
		int val = field->getValue().toInteger();
		if (val < minIndex) {
			val = minIndex;
			field->setNumberValue(Integer(val));
		}
		if (val > maxIndex) {
			val = maxIndex;
			field->setNumberValue(Integer(val));
		}
		tweenRegion->setValue(val);
		setSampleIndex(val);
	};
	tweenRegion->setOnChangeEvent([this](const aly::Number& value) {
		valueRegion->setNumberValue(value);
		setSampleIndex(value.toInteger());
	});
	TextLinkPtr initWeightsButton = TextLinkPtr(new TextLink("Initialize Weights", CoordPX(0.0f, 0.0f), CoordPX(205.0f, entryHeight)));
	initWeightsButton->fontSize = UnitPX(30.0f);
	initWeightsButton->setAlignment(HorizontalAlignment::Center, VerticalAlignment::Middle);
	initWeightsButton->textColor = MakeColor(HSVtoColor(HSV(0.2f,0.4f,1.0f)));
	initWeightsButton->onMouseDown = [this](AlloyContext* context, const InputEvent& e) {
		if (e.button == GLFW_MOUSE_BUTTON_LEFT) {
			sys->evaluate();
			return true;
		}
		return false;
	};
	/*
	TextLinkPtr learnButton = TextLinkPtr(new TextLink("Learning Step", CoordPX(0.0f, 0.0f), CoordPX(170.0f, entryHeight)));
	learnButton->fontSize = UnitPX(30.0f);
	learnButton->setAlignment(HorizontalAlignment::Center, VerticalAlignment::Middle);
	learnButton->textColor = MakeColor(HSVtoColor(HSV(0.2f, 0.4f, 1.0f)));
	learnButton->onMouseDown = [this](AlloyContext* context, const InputEvent& e) {
		if (e.button == GLFW_MOUSE_BUTTON_LEFT) {
			train(std::vector<int>{sampleIndex.toInteger()});
			return true;
		}
		return false;
	};
	*/
	valueRegion->textColor = MakeColor(AlloyDefaultContext()->theme.LIGHTER);
	valueRegion->backgroundColor = MakeColor(0, 0, 0, 0);
	valueRegion->borderWidth = UnitPX(0.0f);
	tweenRegion->backgroundColor = MakeColor(0, 0, 0, 0);
	tweenRegion->borderWidth = UnitPX(0.0f);
	toolbar->setOrientation(Orientation::Horizontal, pixel2(0.0f), pixel2(0.0f));
	toolbar->add(labelRegion);
	toolbar->add(tweenRegion);
	toolbar->add(valueRegion);
	toolbar->add(initWeightsButton);
	//toolbar->add(learnButton);
	controlLayout->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controlLayout->borderWidth = UnitPX(0.0f);
	flowRegion->onSelect=[this](NeuralLayer* layer,const InputEvent& e) {
		setSelectedLayer(layer);
		onEventHandler(AlloyDefaultContext().get(), e);
	};
	layout->setWest(controlLayout, UnitPX(400.0f));
	layout->setEast(expandTree, UnitPX(400.0f));
	controlLayout->setCenter(controls);
	

	graphRegion = GraphPanePtr(new GraphPane("Residual Error", CoordPX(0.0f,0.0f), CoordPercent(1.0f,1.0f)));
	graphRegion->setRoundCorners(false);
	graphRegion->borderWidth = UnitPX(0.0f);
	graphRegion->backgroundColor = MakeColor(AlloyDefaultContext()->theme.DARKER);
	controlLayout->setNorth(graphRegion,0.3333f);
	
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
	viewRegion->add(toolbar);
	viewRegion->add(flowRegion);
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
			timelineSlider->setTimeValue(0);
			timelineSlider->setMinorTick(worker->getIterationsPerStep());
			timelineSlider->setMajorTick(worker->getIterationsPerEpoch());
			timelineSlider->setMaxValue((int)worker->getMaxIteration());
			timelineSlider->setVisible(true);
			context->addDeferredTask([this]() {
				worker->init();
				running = true;
			});
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

	flowRegion->backgroundColor = MakeColor(getContext()->theme.DARK);
	flowRegion->borderColor = MakeColor(getContext()->theme.DARK);
	flowRegion->borderWidth = UnitPX(1.0f);
	Application::addListener(&rootNode);
	rootNode.onEvent = [this](AlloyContext* context, const InputEvent& event) {
		return onEventHandler(context, event);
	};
	getContext()->addDeferredTask([this]() {
		box2px bounds=flowRegion->getBounds();
		flowRegion->add(sys->getInput().get(), bounds.position + pixel2(0.25f*bounds.dimensions.x, 0.25f*bounds.dimensions.y));
		flowRegion->add(sys->getOutput().get(), bounds.position + pixel2(0.75f*bounds.dimensions.x,0.25f*bounds.dimensions.y));
		
	});
	initialize();
	worker->onUpdate = [this](uint64_t iteration, bool lastIteration) {
		graphRegion->updateGraphBounds();
		if (lastIteration || (int)iteration == timelineSlider->getMaxValue().toInteger()) {
			running = false;
			stopButton->setVisible(false);
			playButton->setVisible(true);
		}
		AlloyApplicationContext()->addDeferredTask([this]() {
			timelineSlider->setUpperValue((int)worker->getIteration());
			timelineSlider->setTimeValue((int)worker->getIteration());
		});
	};
	worker->setup(controls);
	timelineSlider->setTimeValue(0);
	timelineSlider->setMinorTick(worker->getIterationsPerStep());
	timelineSlider->setMajorTick(worker->getIterationsPerEpoch());
	timelineSlider->setMaxValue((int)worker->getMaxIteration());

	graphRegion->add(sys->getOutput()->getGraph());
	return true;
}
void TigerApp::setSelectedLayer(tgr::NeuralLayer* layer) {
	selectedLayer = layer;
}
void TigerApp::train(const std::vector<int>& sampleIndexes) {
	worker->init();
	worker->setSamples(sampleIndexes);
	worker->step();
}
bool TigerApp::onEventHandler(AlloyContext* context, const aly::InputEvent& e) {
	if (selectedLayer != nullptr) {
		dragIconPane->setDragOffset(e.cursor, pixel2(20.0f, 20.0f));
		AlloyApplicationContext()->getGlassPane()->setVisible(true);
		if (!dragIconPane->isVisible()) {
			dragIconPane->setVisible(true);
			AlloyApplicationContext()->requestPack();
		}
		if (flowRegion->isVisible() && flowRegion->getBounds().contains(e.cursor)) {
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
				flowRegion->add(selectedLayer, e.cursor);
				flowRegion->update();
				AlloyApplicationContext()->requestPack();
			}
		}
		selectedLayer=nullptr;
		overTarget = false;
	}
	return false;
}
bool TigerApp::initializeXOR() {
	FullyConnectedFilterPtr firstFilter(new FullyConnectedFilter("First Layer",2,2, 2, 2));
	sys->add(firstFilter);

	FullyConnectedFilterPtr secondFilter(new FullyConnectedFilter("Second Layer",firstFilter->getOutputLayer(0), 2, 2));
	sys->add(secondFilter);

	FullyConnectedFilterPtr thirdFilter(new FullyConnectedFilter("Output Layer", secondFilter->getOutputLayer(0), 1, 1));
	sys->add(thirdFilter);
	thirdFilter->getOutputLayer(0)->setFunction(tgr::Linear());
	sys->setInput(firstFilter->getInputLayer(0));
	sys->setOutput(thirdFilter->getOutputLayer(0));
	worker.reset(new NeuralRuntime(sys));
	worker->inputSampler = [this](const NeuralLayerPtr& input, int idx) {
		aly::Image1f& inputData = trainInputData[idx];
		input->set(inputData);
	};
	worker->outputSampler = [this](std::vector<float>& outputData, int idx) {
		int out = trainOutputData[idx];
		outputData.resize(1);
		outputData[0] = float(out);
	};
	Image1f img(2, 2);
	std::vector<int> samples;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			int b = i^j;
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					img(ii, jj).x = (ii == i&&jj == j)?1.0f:-1.0f;
				}
			}
			samples.push_back(int(samples.size()));
			trainInputData.push_back(img);
			trainOutputData.push_back(b);
		}
	}
	setSampleRange(0, (int)trainInputData.size() - 1);
	setSampleIndex(0);
	worker->setSamples(samples);
	return true;
}
bool TigerApp::initializeLeNet5() {
	parse_mnist_images(trainFile, trainInputData, 0.0f, 1.0f, 2, 2);
	parse_mnist_labels(trainLabelFile, trainOutputData);
	if (trainInputData.size() > 0) {
		const Image1f& ref = trainInputData[0];
		ConvolutionFilterPtr conv1(new ConvolutionFilter(ref.width, ref.height, 5, 6));
		sys->add(conv1);
		std::vector<NeuralLayerPtr> all;
		for (int i = 0; i < conv1->getOutputSize(); i++) {
			AveragePoolFilterPtr avg1(new AveragePoolFilter(conv1->getOutputLayer(i), 2));
			avg1->setName(MakeString() << "Sub-Sample [" << i << "]");
			sys->add(avg1);
			all.push_back(avg1->getOutputLayer(0));
		}

		ConvolutionFilterPtr conv2(new ConvolutionFilter(all, 5, 16));
		std::vector<std::pair<int, int>> connectionTable;
		for (int ii = 0; ii < 6; ii++) {
			for (int jj = 0; jj < 16; jj++) {
				if (MNIST_TABLE[jj + ii * 16]) {
					connectionTable.push_back(std::pair<int, int>(ii, jj));
				}
			}
		}
		conv2->setConnectionMap(connectionTable);
		sys->add(conv2);
		all.clear();
		for (int i = 0; i < conv2->getOutputSize(); i++) {
			AveragePoolFilterPtr avg2(new AveragePoolFilter(conv2->getOutputLayer(i), 2));
			avg2->setName(MakeString() << "Sub-Sample [" << i << "]");
			sys->add(avg2);
			ConvolutionFilterPtr conv3(new ConvolutionFilter(avg2->getOutputLayer(0), 5, 1));
			sys->add(conv3);
			all.push_back(conv3->getOutputLayer(0));
		}
		FullyConnectedFilterPtr decisionFilter(new FullyConnectedFilter("Decision Layer", all, 10, 1));
		sys->add(decisionFilter);
		sys->setInput(conv1->getInputLayer(0));
		sys->setOutput(decisionFilter->getOutputLayer(0));
		worker.reset(new NeuralRuntime(sys));
		worker->inputSampler = [this](const NeuralLayerPtr& input, int idx) {
			aly::Image1f& inputData = trainInputData[idx];
			input->set(inputData);
		};
		worker->outputSampler = [this](std::vector<float>& outputData, int idx) {
			int out = trainOutputData[idx];
			outputData.resize(10);
			for (int i = 0; i <(int)outputData.size(); i++) {
				outputData[i] = (out == i) ? 1.0f : 0.0f;
			}
		};
		setSampleRange(0, (int)trainInputData.size() - 1);
		setSampleIndex(sampleIndex.toInteger());
		std::vector<int> samples;
		for (int i = 0; i < trainInputData.size(); i += 100) {
			samples.push_back(i);
		}
		worker->setSamples(samples);
		return true;
	}
	return false;
}
void TigerApp::initialize() {
	sys.reset(new NeuralSystem(flowRegion));
	switch (exampleIndex) {
		case 0: initializeXOR(); break;
		case 1: initializeLeNet5(); break;
	}
	sys->initialize(expandTree);
}
void TigerApp::draw(AlloyContext* context) {
	if (running) {
		if (!worker->step()) {
			running = false;
		}
	}
}