cmake_minimum_required(VERSION 2.8)
project(FACE)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

find_package(Caffe)
#message(FATAL_ERROR ${Caffe_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})
link_directories(..)

set(SOURCE_FILES extract_feature.cpp FACE.cpp FACE.h)
add_executable(ExtractFeature ${SOURCE_FILES})
target_link_libraries(ExtractFeature ${OpenCV_LIBS} )
target_link_libraries(ExtractFeature ${Caffe_LIBRARIES})
target_link_libraries(ExtractFeature havon_ffd)

set(SOURCE_FILES face_veri.cpp FACE.cpp FACE.h)
add_executable(FaceVeri ${SOURCE_FILES})
target_link_libraries(FaceVeri ${OpenCV_LIBS} )
target_link_libraries(FaceVeri ${Caffe_LIBRARIES})
target_link_libraries(FaceVeri havon_ffd)

set(SOURCE_FILES face_recog1.cpp FACE.cpp FACE.h)
add_executable(FaceRecog1 ${SOURCE_FILES})
target_link_libraries(FaceRecog1 ${OpenCV_LIBS} )
target_link_libraries(FaceRecog1 ${Caffe_LIBRARIES})
target_link_libraries(FaceRecog1 havon_ffd)
