#ifndef _TIGERAPP_H_
#define _TIGERAPP_H_
#include "AlloyApplication.h"
#include "AlloyWidget.h"
#include "AlloyVector.h"
#include "AlloyWorker.h"
#include "AlloyImage.h"
class TigerApp : public aly::Application {
protected:
	float currentIso;
	int example;
	aly::Image1f gray;
	aly::ImageRGBA img;
	bool parametersDirty;
	bool frameBuffersDirty;
	bool running = false;
	aly::Image2f vecField;
	aly::Number lineWidth;
	aly::Number particleSize;
	aly::Color lineColor;
	aly::Color pointColor;
	aly::Color particleColor;
	aly::Color normalColor;
	aly::Color springlColor;
	aly::Color matchColor;
	aly::Color vecfieldColor;
	aly::AdjustableCompositePtr resizeableRegion;
	aly::IconButtonPtr playButton, stopButton;
public:
	TigerApp();
	virtual void draw(aly::AlloyContext* context) override;
	bool init(aly::Composite& rootNode);

};

#endif
