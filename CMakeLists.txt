cmake_minimum_required(VERSION 3.27)

# Build for the GPU of the current machine; override with -DCMAKE_CUDA_ARCHITECTURES=<arch>
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda-plasma-simulation LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Keep the exe directly in <build>/bin without per-config subfolder:
# the app searches configs/ and res/ at most two levels above the exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(CONFIG_TYPE ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG_TYPE} CONFIG_TYPE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPE} ${CMAKE_BINARY_DIR}/bin)
endforeach()

file(GLOB CPP_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/cuda/*.cu")
file(GLOB GL_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/openGL/*.cpp")

file(GLOB CPP_HEADERS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB CUDA_HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/cuda/*.cuh")
file(GLOB GL_HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/openGL/*.h")

add_executable(simulation ${CPP_SOURCES} ${CUDA_SOURCES} ${GL_SOURCES} ${CPP_HEADERS} ${CUDA_HEADERS} ${GL_HEADERS})

target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(simulation PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(WIN32)
    source_group("Source Files" FILES ${CPP_SOURCES})
    source_group("Source Files\\cuda" FILES ${CUDA_SOURCES})
    source_group("Source Files\\openGL" FILES ${GL_SOURCES})

    source_group("Headers Files" FILES ${CPP_HEADERS})
    source_group("Headers Files\\cuda" FILES ${CUDA_HEADERS})
    source_group("Headers Files\\openGL" FILES ${GL_HEADERS})

    target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/external/glew/include)
    target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/external/glfw/include)
    target_include_directories(simulation PRIVATE ${CMAKE_SOURCE_DIR}/external/yaml/include)

    target_link_directories(simulation PRIVATE  ${CMAKE_SOURCE_DIR}/external/glew/lib)
    target_link_directories(simulation PRIVATE  ${CMAKE_SOURCE_DIR}/external/glfw/lib)
    target_link_directories(simulation PRIVATE  ${CMAKE_SOURCE_DIR}/external/yaml/lib)

    target_link_libraries(simulation
        glew32s.lib
        glfw3.lib
        opengl32.lib
        CUDA::cufft
        debug yaml-cppd.lib
        optimized yaml-cpp.lib
    )

    # cudart_static pulls LIBCMT; prebuilt glew/glfw pull release MSVCRT into Debug;
    # /ignore:4099 - prebuilt libs ship without PDB files
    target_link_options(simulation PRIVATE
        /NODEFAULTLIB:LIBCMT
        /ignore:4099
        $<$<CONFIG:Debug>:/NODEFAULTLIB:MSVCRT>
    )
elseif(UNIX)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW REQUIRED glfw3)
    pkg_check_modules(GLEW REQUIRED glew)
    pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

    target_include_directories(simulation PRIVATE ${GLFW_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})

    target_link_libraries(simulation
        ${GLEW_LIBRARIES}
        ${GLFW_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        GL
        CUDA::cufft
    )
endif()

target_compile_definitions(simulation PRIVATE YAML_CPP_STATIC_DEFINE)
target_compile_definitions(simulation PRIVATE GLEW_STATIC)
# CUDA 13 deprecates legacy vector types (double4 etc.) used inside cuRAND headers
target_compile_definitions(simulation PRIVATE __NV_NO_VECTOR_DEPRECATION_DIAG)
