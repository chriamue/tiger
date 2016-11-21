#include "AveragePoolFilter.h"
namespace tgr {
	AveragePoolFilter::AveragePoolFilter(const std::vector<NeuralLayerPtr>& inputLayers, int kernelSize):kernelSize(kernelSize) {
		if (kernelSize % 2 == 1) {
			throw std::runtime_error("Kernel size must be even.");
		}
	}
	void AveragePoolFilter::initialize(NeuralSystem& sys) {

	}
}