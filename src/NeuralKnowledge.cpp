#include "NeuralLayer.h"
#include "NeuralKnowledge.h"
#include "AlloyFileUtil.h"
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
using namespace aly;
namespace tgr {
	void NeuralKnowledge::add(const NeuralLayer& layer) {
		weights[layer.getId()] = layer.getWeights();
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