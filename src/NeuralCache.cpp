#include "NeuralCache.h"
#include <AlloyFileUtil.h>
using namespace aly;
namespace tgr {
	void CacheElement::load() {
		std::lock_guard<std::mutex> lockMe(accessLock);
		if (!loaded) {
			neuralKnowledge.reset(new NeuralKnowledge());
			ReadNeuralKnowledgeFromFile(knowledgeFile, *neuralKnowledge);
			loaded = true;
		}
	}
	void CacheElement::unload() {
		std::lock_guard<std::mutex> lockMe(accessLock);
		if (loaded) {
			if (writeOnce) {
				WriteNeuralKnowledgeToFile(neuralKnowledge->getFile(), *neuralKnowledge);
				writeOnce = false;
			}
			neuralKnowledge.reset();
			loaded = false;
		}
	}
	void CacheElement::set(const NeuralKnowledge& nknow) {
		neuralKnowledge.reset(new NeuralKnowledge());
		*neuralKnowledge = nknow;
		knowledgeFile = nknow.getFile();
		loaded = true;
	}
	std::shared_ptr<NeuralKnowledge> CacheElement::getKnowledge() {
		load();
		return neuralKnowledge;
	}
	std::shared_ptr<CacheElement> NeuralCache::set(int frame, const NeuralKnowledge& nknow) {
		std::lock_guard<std::mutex> lockMe(accessLock);
		auto iter = cache.find(frame);
		std::shared_ptr<CacheElement> elem;
		if (iter != cache.end()) {
			elem = iter->second;
		}
		else {
			elem = std::shared_ptr<CacheElement>(new CacheElement());
			cache[frame] = elem;
		}
		elem->set(nknow);
		if (elem->isLoaded()) {
			while (loadedList.size() >= maxElements) {
				cache[loadedList.begin()->second]->unload();
				loadedList.erase(loadedList.begin());
			}
			loadedList.insert(std::pair<uint64_t, int>(counter++, frame));
		}
		return elem;
	}
	std::shared_ptr<CacheElement> NeuralCache::get(int frame) {
		std::lock_guard<std::mutex> lockMe(accessLock);
		auto iter = cache.find(frame);
		if (iter != cache.end()) {
			std::shared_ptr<CacheElement> elem = iter->second;
			if (!elem->isLoaded()) {
				while (loadedList.size() >= maxElements) {
					cache[loadedList.begin()->second]->unload();
					loadedList.erase(loadedList.begin());
				}
				elem->load();
				loadedList.insert(std::pair<uint64_t, int>(counter++, frame));
			}
			return elem;
		}
		else {
			return std::shared_ptr<CacheElement>();
		}
	}
	CacheElement::~CacheElement() {
		if (FileExists(knowledgeFile)) {
			RemoveFile(knowledgeFile);
			std::string imageFile = GetFileWithoutExtension(knowledgeFile) + ".png";
			if (FileExists(imageFile))RemoveFile(imageFile);
		}
	}
	void NeuralCache::clear() {
		std::lock_guard<std::mutex> lockMe(accessLock);
		counter = 0;
		loadedList.clear();
		cache.clear();

	}
}