
#include "Neuron.h"
namespace jc {
	class Signal;
	class Alpha {
		float value=0;
		std::shared_ptr<Signal> signal;
	};
	class Signal {
		std::vector<std::shared_ptr<Axon>> axons;

	};
}