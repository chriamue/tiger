#ifndef _TIGERAPP_H_
#define _TIGERAPP_H_
#include "AlloyApplication.h"
#include "AlloyWidget.h"
#include "AlloyVector.h"
#include "AlloyWorker.h"
#include "AlloyImage.h"
#include "NeuralSystem.h"
class TigerApp : public aly::Application {
protected:
	const float GLYPH_SCALE;
	bool parametersDirty;
	bool frameBuffersDirty;
	bool running = false;
	tgr::NeuralSystem sys;
	tgr::NeuronLayerPtr currentLayer;
	aly::AdjustableCompositePtr resizeableRegion;
	aly::IconButtonPtr playButton, stopButton;
	void initialize();
public:
	TigerApp();
	virtual void draw(aly::AlloyContext* context) override;
	bool init(aly::Composite& rootNode);

};

#endif
