#ifndef _NEURAL_CACHE_H_
#define _NEURAL_CACHE_H_
#include "NeuralKnowledge.h"
#include <map>
#include <mutex>
#include <tuple>
#include <set>
namespace tgr {
	class CacheElement {
	protected:
		bool loaded;
		bool writeOnce;
		std::string knowledgeFile;
		std::shared_ptr<NeuralKnowledge> WeightVec;
		std::mutex accessLock;
	public:
		bool isLoaded() {
			std::lock_guard<std::mutex> lockMe(accessLock);
			return loaded;
		}
		CacheElement() :loaded(false), writeOnce(true) {
		}
		~CacheElement();
		std::string getFile() const {
			return knowledgeFile;
		}
		void load();
		void unload();
		void set(const NeuralKnowledge& WeightVec);
		std::shared_ptr<NeuralKnowledge> getKnowledge();
	};
	struct CacheCompare {
		inline bool operator() (const std::pair<uint64_t, int>& lhs, const std::pair<uint64_t, int>& rhs) const {
			return lhs.first < rhs.first;
		}
	};
	class SpringlCache2D {
	protected:
		std::map<int, std::shared_ptr<CacheElement>> cache;
		std::set<std::pair<uint64_t, int>, CacheCompare> loadedList;
		std::mutex accessLock;
		int maxElements = 32;
		uint64_t counter = 0;
	public:
		std::shared_ptr<CacheElement> set(int frame, const NeuralKnowledge& springl);
		std::shared_ptr<CacheElement> get(int frame);
		void clear();
	};
}
#endif