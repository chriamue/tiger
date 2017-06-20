
rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

EXOBJS := $(patsubst %.cpp, %.o, $(call rwildcard, ./src/, *.cpp))
CXX = g++
CC = gcc

CXXFLAGS:= -DGL_GLEXT_PROTOTYPES=1 -std=gnu++14 -O3 -w -fPIC -MMD -MP -fopenmp -c -g -fmessage-length=0 -I./include/ -I../alloy/include/core/ -I../alloy/include/
# Needed for TinyDNN
CXXFLAGS+= -msse3 -mavx -mavx2 -mfma -march=core-avx2 -Wno-narrowing -msse3 -march=core-avx2
CXXFLAGS+= -DCNN_USE_TBB=1 -DDNN_USE_IMAGE_AP=1 -DCNN_USE_SSE=1 
# -DCNN_USE_AVX=1 -DCNN_USE_AVX2=1

CFLAGS:= -DGL_GLEXT_PROTOTYPES=1 -std=c11 -O3 -w -fPIC -MMD -MP -fopenmp -c -g -fmessage-length=0 -I./include/ -I./ext/alloy/include/core/ -I./ext/alloy/include/
LDLIBS =-L./ -L./ext/alloy/Release/ -L/usr/lib/ -L/usr/local/lib/ -L/usr/lib/x86_64-linux-gnu/ -L./ext/alloy/ext/glfw/src/
LIBS = -ltbb -lAlloy -lglfw3 -lstdc++ -lgcc -lgomp -lGL -lXext -lGLU -lGLEW -lXi -lXrandr -lX11 -lXxf86vm -lXinerama -lXcursor -lXdamage -lpthread -lm -ldl

ifneq ($(wildcard /usr/lib/libOpenCL.so /usr/local/lib/libOpenCL.so /usr/lib/x86_64-linux-gnu/libOpenCL.so), "")
	LIBS+=-lOpenCL
endif

default: all

all: $(EXOBJS)
	mkdir -p ./Release
	$(CXX) -o ./Release/tiger $(EXOBJS) $(LDLIBS) -L./Release $(LIBS) -Wl,-rpath="./:./Release/:../ext/alloy/Release/:./ext/alloy/Release/"

clean:
	rm -f $(EXOBJS)
	rm -f ./Release/tiger
	
.PHONY : all

