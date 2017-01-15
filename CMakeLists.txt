# Introspectable Physics
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_COMPILER "clang++-3.6")
set(CMAKE_C_COMPILER "clang-3.6")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-deprecated-register -O2 -Wno-overloaded-virtual" )

find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${Box2D_SOURCE_DIR}
)

add_executable(realPhysics src/realPhysics.cpp)

target_link_libraries(realPhysics
	libliquidfun.a
	${OpenCV_LIBRARIES}
)
