cmake_minimum_required(VERSION 2.8)
project(FACE)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)
link_directories(..)

set(SOURCE_FILES main.cpp FACE.cpp FACE.h)
add_executable(Detect ${SOURCE_FILES})
target_link_libraries(Detect ${OpenCV_LIBS} )
target_link_libraries(Detect havon_ffd)

