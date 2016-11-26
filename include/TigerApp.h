#ifndef _TIGERAPP_H_
#define _TIGERAPP_H_
#include "AlloyApplication.h"
#include "AlloyWidget.h"
#include "AlloyVector.h"
#include "AlloyWorker.h"
#include "AlloyImage.h"
#include "NeuralSystem.h"
#include "AlloyExpandTree.h"
#include "NeuralFlowPane.h"
#include "AlloyTimeline.h"
class TigerApp : public aly::Application {
protected:
	tgr::NeuralLayer* selectedLayer;
	bool parametersDirty;
	bool frameBuffersDirty;
	bool running = false;
	tgr::NeuralSystem sys;
	aly::IconButtonPtr playButton, stopButton;
	aly::ExpandTreePtr expandTree;
	aly::DrawPtr dragIconPane;
	aly::NeuralFlowPanePtr renderRegion;
	aly::TimelineSliderPtr timelineSlider;
	aly::Number epochs;
	aly::Number iterationsPerEpoch;
	aly::Number learningRateInitial;
	aly::Number learningRateDelta;
	std::string trainFile;
	std::string evalFile;
	std::string trainLabelFile;
	std::string evalLabelFile;
	void initialize();
public:

	bool overTarget = false;
	TigerApp();
	virtual void draw(aly::AlloyContext* context) override;
	bool init(aly::Composite& rootNode);
	bool onEventHandler(aly::AlloyContext* context, const aly::InputEvent& e);
	void setSelectedLayer(tgr::NeuralLayer* layer);
};

#endif
