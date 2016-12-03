#include "NeuralLayer.h"
#include "NeuralKnowledge.h"
#include "AlloyFileUtil.h"
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
using namespace aly;
namespace tgr {
	void NeuralKnowledge::add(const NeuralLayer& layer) {
		const std::vector<std::shared_ptr<Signal>>& signals= layer.getSignals();
		int N = (int)signals.size();
		if (weights.find(layer.getId()) == weights.end()) {
			weights[layer.getId()] = knowledge();
		}
		knowledge& W = weights[layer.getId()];
		W.resize(N);
		for (int n = 0; n < N; n++) {
			W[n] = signals[n]->value;
		}
	}
	void WriteNeuralKnowledgeToFile(const std::string& file, const NeuralKnowledge& params) {
		std::string ext = GetFileExtension(file);
		if (ext == "json") {
			std::ofstream os(file);
			cereal::JSONOutputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
		else if (ext == "xml") {
			std::ofstream os(file);
			cereal::XMLOutputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
		else {
			std::ofstream os(file, std::ios::binary);
			cereal::PortableBinaryOutputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
	}
	void ReadNeuralKnowledgeFromFile(const std::string& file, NeuralKnowledge& params) {
		std::string ext = GetFileExtension(file);
		if (ext == "json") {
			std::ifstream os(file);
			cereal::JSONInputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
		else if (ext == "xml") {
			std::ifstream os(file);
			cereal::XMLInputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
		else {
			std::ifstream os(file, std::ios::binary);
			cereal::PortableBinaryInputArchive archive(os);
			archive(cereal::make_nvp("tigernet", params));
		}
	}
}