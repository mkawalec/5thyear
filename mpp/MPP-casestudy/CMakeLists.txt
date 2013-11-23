cmake_minimum_required(VERSION 2.6)
project(casestudy)
include_directories(src include)
set(HEADER_FILES include/pgmio.h include/arralloc.h include/helpers.h)
message(status "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(edges src/edges.c src/pgmio.c src/arralloc.c src/helpers.c ${HEADER_FILES})
target_link_libraries(edges -lm ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(edges PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(edges PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

message(status "Setting GCC flags")
set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -O2 -Wall -Wextra -W -pedantic -g")

