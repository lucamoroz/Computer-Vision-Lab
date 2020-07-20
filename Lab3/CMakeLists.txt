cmake_minimum_required(VERSION 3.15)
project(Lab3)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( include ${OpenCV_INCLUDE_DIRS} )

add_executable( ${PROJECT_NAME} src/main.cpp)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )