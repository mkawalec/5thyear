cmake_minimum_required(VERSION 2.8)
project(casestudy)
include_directories(src include)
set(HEADER_FILES include/pgmio.h include/arralloc.h)
message(status "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(edges src/edges.c src/pgmio.c src/arralloc.c ${HEADER_FILES})
target_link_libraries(edges -lm ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(edges PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(edges PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
    message(status "Setting GCC flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall -Wextra -W -pedantic")
else()
    message(status "Setting MSVC flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHc-")
endif()

