cmake_minimum_required(VERSION 2.8.12)
project(infer_superpoint)
add_compile_options(-std=c++11)


set(OpenCV_DIR /home/cyh/opencv455/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

Set(ONNXRUNTIME_DIR /home/cyh/CodeFile/CPP/algorithm_identification/superPointMatch/onnxruntime-linux-x64-1.10.0)
# Set(ONNXRUNTIME_DIR /home/cyh/CodeFile/CPP/SuperPointPretrainedNetwork-master/superpoint_ort/onnxruntime_gpu)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

include_directories(${OpenCV_INCLUDE_DIRS}
   ${ONNXRUNTIME_DIR}/include)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME}
    src/infer_superpoint.hpp
    src/infer_superpoint.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
    
    # ${ONNXRUNTIME_DIR}/build/Linux/Release/libonnxruntime_providers_cuda.so
    # ${ONNXRUNTIME_DIR}/build/Linux/Release/libonnxruntime_providers_shared.so
    # ${ONNXRUNTIME_DIR}/build/Linux/Release/libcustom_op_library.so
    # ${ONNXRUNTIME_DIR}/build/Linux/Release/libtest_execution_provider.so
    # ${ONNXRUNTIME_DIR}/build/Linux/Release/libonnxruntime_providers_shared.so
    # ${ONNXRUNTIME_DIR}/build/Linux/Release/libonnxruntime_providers_shared.so
    # ${ONNXRUNTIME_DIR}/build/Linux/Release/libonnxruntime_providers_shared.so
    # ${ONNXRUNTIME_DIR}/build/Linux/Release/libonnxruntime_providers_shared.so

)