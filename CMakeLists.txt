cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tiger C CXX)

find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(TBB QUIET)
find_package(OpenMP QUIET)

option(USE_SSE        "Build tiny-dnn with SSE library support"     ON)
option(USE_AVX        "Build tiny-dnn with AVX library support"     ON)
option(USE_AVX2       "Build tiny-dnn with AVX2 library support"   OFF)
option(USE_TBB        "Build tiny-dnn with TBB library support"    ON)
option(USE_OMP        "Build tiny-dnn with OMP library support"    OFF)
option(USE_NNPACK     "Build tiny-dnn with NNPACK library support" OFF)
option(USE_OPENCL     "Build tiny-dnn with OpenCL library support" OFF) 
option(USE_LIBDNN     "Build tiny-dnn with GreenteaLibDNN library support" OFF)
option(USE_SERIALIZER "Build tiny-dnn with Serialization support"   ON)
option(USE_DOUBLE     "Build tiny-dnn with double precision computations"  OFF)
option(USE_IMAGE_API  "Build tiny-dnn with Image API support"       ON)
option(USE_GEMMLOWP   "Build tiny-dnn with gemmlowp support"       OFF)

# based on https://github.com/cginternals/cmake-init

set(DEFAULT_COMPILE_OPTIONS)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++14 support has been enabled by default.")

add_subdirectory(ext)
link_directories("${CMAKE_BINARAY_DIR}/ext/alloy/" "${CMAKE_BINARAY_DIR}/ext/alloy/ext_build/glfw/src/")

# Unix
if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-msse3" COMPILER_HAS_SSE_FLAG)
    check_cxx_compiler_flag("-mavx"  COMPILER_HAS_AVX_FLAG)
    check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2_FLAG)
    check_cxx_compiler_flag("-mfma" COMPILER_HAS_AVX2_FLAG)

    # set Streaming SIMD Extension (SSE) instructions
    if(USE_SSE AND COMPILER_HAS_SSE_FLAG)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
    endif(USE_SSE AND COMPILER_HAS_SSE_FLAG)
    # set Advanced Vector Extensions (AVX)
    if(USE_AVX AND COMPILER_HAS_AVX_FLAG)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx")
    endif(USE_AVX AND COMPILER_HAS_AVX_FLAG)
    # set Advanced Vector Extensions 2 (AVX2)
    if(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
        add_definitions(-DCNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx2 -mfma -march=core-avx2")
    endif(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)

    # include extra flags to the compiler
    # TODO: add info about those flags.
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing")
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    set(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pthread")
elseif(MSVC)
    if(USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:SSE2")
    endif(USE_SSE)
    if(USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX")
    endif(USE_AVX)
    if(USE_AVX2)
        add_definitions(-DCNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX2")
    endif(USE_AVX2)
    # include specific flags for release and debug modes.
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}
        /Ox /Oi /Ot /Oy /GL /fp:fast /GS-")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG}")
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /W4 /bigobj")
    # this is fine
    add_definitions(-D _CRT_SECURE_NO_WARNINGS)
    add_definitions(-D _SCL_SECURE_NO_WARNINGS)
    # prolly powerless with header-only project
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /MP")
endif()

####
# Set compiler options
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS} ${OpenMP_CXX_FLAGS}")

file(GLOB_RECURSE headers
    "include/*.h"
)

file(GLOB_RECURSE sources
    "src/*.cpp"
)

add_executable(tiger
    ${sources}
    ${headers}
)

add_dependencies(tiger alloy)

target_include_directories(tiger
    PRIVATE
    ${OpenCL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ext/alloy/include
    ${CMAKE_SOURCE_DIR}/ext/alloy/include/core
)
message (status "libs"${ALLOY_EXTRA_LIBS} )
target_link_libraries(tiger
    PRIVATE
    PUBLIC
    alloy  stdc++ gcc gomp Xext ${TBB_LIBRARIES}
    /home/chriamue/workspace/tiger/build/ext/alloy/ext_build/glfw/src/libglfw3.a
    ${OpenCL_LIBRARY} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} 
    gomp Xi Xrandr X11 Xxf86vm Xinerama Xcursor Xdamage pthread m dl
)

# MSVC compiler options
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
        /MP           # -> build with multiple processes
        /W4           # -> warning level 4
        # /WX         # -> treat warnings as errors

        /wd4251       # -> disable warning: 'identifier': class 'type' needs to have dll-interface to be used by clients of class 'type2'
        /wd4592       # -> disable warning: 'identifier': symbol will be dynamically initialized (implementation limitation)
        # /wd4201     # -> disable warning: nonstandard extension used: nameless struct/union (caused by GLM)
        # /wd4127     # -> disable warning: conditional expression is constant (caused by Qt)
        
        #$<$<CONFIG:Debug>:
        #/RTCc         # -> value is assigned to a smaller data type and results in a data loss
        #>

        $<$<CONFIG:Release>: 
        /Gw           # -> whole program global optimization
        /GS-          # -> buffer security check: no 
        /GL           # -> whole program optimization: enable link-time code generation (disables Zi)
        /GF           # -> enable string pooling
        >
        
        # No manual c++11 enable for MSVC as all supported MSVC versions for cmake-init have C++11 implicitly enabled (MSVC >=2013)
    )
endif ()

# GCC and Clang compiler options
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}


        -Wall
        -Wextra
        -Wunused

        -Wreorder
        -Wignored-qualifiers
        -Wmissing-braces
        -Wreturn-type
        -Wswitch
        -Wswitch-default
        -Wuninitialized
        -Wmissing-field-initializers
        
        $<$<CXX_COMPILER_ID:GNU>:
            -Wmaybe-uninitialized
            
            $<$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,4.8>:
                -Wpedantic
                
                -Wreturn-local-addr
            >
        >
        
        $<$<CXX_COMPILER_ID:Clang>:
            -Wpedantic
            
            -Wreturn-stack-address
        >
        
        $<$<PLATFORM_ID:Darwin>:
            -pthread
        >
        
        # Required for CMake < 3.1; should be removed if minimum required CMake version is raised.
        $<$<VERSION_LESS:${CMAKE_VERSION},3.1>:
            -std=c++11
        >
    )
endif ()

target_compile_options(tiger
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)