cmake_minimum_required(VERSION 2.8)
#cmake_policy(SET CMP0003 OLD)

set(PROJECT_NAME bezier_solve)

include_directories( . )
# these are specific to my systems, but shouldn't
# hurt anything if invalid.  May want to edit them
# if a user directory exists where ceres or other
# dependencies are required.
#include_directories( $ENV{HG_CLONE_DIR}/external/include )
include_directories( $ENV{HOME}/other/install/include )
link_directories($ENV{HOME}/other/install/lib)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${CMAKE_BINARY_DIR}/other)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

###############################################################################

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDES})
###############################################################################

add_executable(${PROJECT_NAME}
bezier_solve.cpp
)

target_link_libraries(${PROJECT_NAME} 
${CERES_LIBRARIES}
${OpenCV_LIBS}
glog
gflags
#boost_thread
#boost_filesystem
#boost_system
)

