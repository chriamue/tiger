#include <AlloyMath.h>
namespace aly {
class BitVolume {
public:
	int rows, cols, slices;
	std::vector<bool> data;
	BitVolume(int rows = 0, int cols = 0, int slices = 0);
	bool operator ()(int i, int j, int k) const;
	inline bool operator ()(int3 ijk) const {
		return operator()(ijk.x, ijk.y, ijk.z);
	}
	void set(int i, int j, int k, bool val);
	inline void set(int3 ijk, bool val) {
		set(ijk.x, ijk.y, ijk.z, val);
	}

	void resize(int r, int c, int s);
	inline void resize(int3 d) {
		resize(d.x, d.y, d.z);
	}
	void resize(int r, int c, int s, bool val);
	inline void resize(int3 d, bool val) {
		resize(d.x, d.y, d.z, val);
	}
	void clear();

};
}
