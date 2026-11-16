cmake_minimum_required(VERSION 3.21)
project(quickstart LANGUAGES C)

# Copy this folder anywhere and point IMPROC_DIR at the opencv_c checkout.
set(IMPROC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH
    "opencv_c source directory")
add_subdirectory(${IMPROC_DIR} improc)

add_executable(quickstart main.c)
target_link_libraries(quickstart PRIVATE improc)
