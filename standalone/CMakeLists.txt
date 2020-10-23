cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(TheMendimeterStandalone LANGUAGES CXX)

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

CPMAddPackage(
  NAME cxxopts
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 2.2.0
  OPTIONS "CXXOPTS_BUILD_EXAMPLES Off" "CXXOPTS_BUILD_TESTS Off"
)

CPMAddPackage(NAME TheMendimeter SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# ---- Create standalone executable ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)

add_executable(TheMendimeterStandalone ${sources})

set_target_properties(TheMendimeterStandalone PROPERTIES CXX_STANDARD 17 OUTPUT_NAME "TheMendimeter")

target_link_libraries(TheMendimeterStandalone TheMendimeter cxxopts)
