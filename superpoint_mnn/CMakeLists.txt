cmake_minimum_required(VERSION 3.9)
project(superpoint-mnn)

set(CMAKE_CXX_STANDARD 11)

# opencv 
set(OpenCV_DIR   "/home/cyh/opencv455/lib/cmake/opencv4/") 
find_package(OpenCV REQUIRED)

# find_package(CUDA REQUIRED)

include_directories(
        /home/cyh/opencv455/include/opencv4 
        ${PROJECT_SOURCE_DIR}/MNN/include
        ${PROJECT_SOURCE_DIR}/src
)

link_directories(${PROJECT_SOURCE_DIR}/MNN/build)

FILE(GLOB SRC_LIST_CPP
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_executable(${PROJECT_NAME} ${SRC_LIST_CPP})

target_link_libraries(${PROJECT_NAME}
        MNN
        ${OpenCV_LIBS}
)
