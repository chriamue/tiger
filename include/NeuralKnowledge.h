#ifndef _NEURAL_KNOWLEDGE_H_
#define _NEURAL_KNOWLEDGE_H_
#include <AlloyOptimizationMath.h>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>

namespace tgr {
class NeuralLayer;
class NeuralSystem;
typedef std::vector<Vec1f> Knowledge;
class NeuralKnowledge {
protected:
	std::map<int, Knowledge> weights;
	std::map<int, Knowledge> biasWeights;
	std::string name;
	std::string file;
public:
	NeuralKnowledge(const std::string& name = "") :
			name(name) {
	}
	std::string getName() const {
		return name;
	}
	void setName(const std::string& n) {
		name = n;
	}
	std::string getFile() const {
		return file;
	}
	void setFile(const std::string& f) {
		file = f;
	}
	Knowledge& getWeights(const NeuralLayer& layer);
	const Knowledge& getWeights(const NeuralLayer& layer) const;

	Knowledge& getBiasWeights(const NeuralLayer& layer);
	const Knowledge& getBiasWeights(const NeuralLayer& layer) const;
	void clear() {
		weights.clear();
	}
	void add(const NeuralLayer& layer);
	void set(const NeuralSystem& sys);
	template<class Archive> void save(Archive & ar) const {
		ar(CEREAL_NVP(name), CEREAL_NVP(file), CEREAL_NVP(weights),
				CEREAL_NVP(biasWeights));
	}
	template<class Archive> void load(Archive & ar) {
		ar(CEREAL_NVP(name), CEREAL_NVP(file), CEREAL_NVP(weights),
				CEREAL_NVP(biasWeights));
	}
};
void WriteNeuralKnowledgeToFile(const std::string& file,
		const NeuralKnowledge& params);
void ReadNeuralKnowledgeFromFile(const std::string& file,
		NeuralKnowledge& params);

}
#endif
