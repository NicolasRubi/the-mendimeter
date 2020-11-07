cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif()

# ---- Project ----

# Note: update this to your new project's name and version
project(
  TheMendimeter
  VERSION 1.0
  LANGUAGES CXX
)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info

include(cmake/CPM.cmake)

# include("$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

# Set OpenCVConfig.cmake Search Directory
set( OpenCV_DIR )
if( NOT CMAKE_CL_64 )
  set( OpenCV_DIR "$ENV{VCPKG_ROOT}/installed/x86-windows/share/opencv" )
else()
  set( OpenCV_DIR "$ENV{VCPKG_ROOT}/installed/x64-windows/share/opencv" )
endif()

find_package(OpenCV REQUIRED)

if( opencv_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )
endif()

# PackageProject.cmake will be used to make our target installable
CPMAddPackage(
  NAME PackageProject.cmake
  GITHUB_REPOSITORY TheLartians/PackageProject.cmake
  VERSION 1.3
)

# set(USE_CCACHE YES CACHE INTERNAL "")
# set(ENABLE_CCACHE YES CACHE INTERNAL "")

# ---- Add source files ----

# Note: globbing sources is considered bad practice as CMake's generators may not detect new files
# automatically. Keep that in mind when changing files, or explicitly mention them here.
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")

# ---- Create library ----

# Note: for header-only libraries change all PUBLIC flags to INTERFACE and create an interface
# target: add_library(TheMendimeter INTERFACE) set_target_properties(TheMendimeter PROPERTIES
# INTERFACE_COMPILE_FEATURES cxx_std_17)

add_library(TheMendimeter ${headers} ${sources})

set_target_properties(TheMendimeter PROPERTIES CXX_STANDARD 17)

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(TheMendimeter PUBLIC "$<$<BOOL:${MSVC}>:/permissive->")

# Link dependencies (if required) 
target_link_libraries(TheMendimeter PUBLIC ${OpenCV_LIBS})

target_include_directories(
  TheMendimeter PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                 $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

add_executable(TheMendimeterApp ${headers} ${sources})

# Link app dependencies (if required) 
target_link_libraries(TheMendimeterApp LINK_PUBLIC TheMendimeter)

# ---- Create an installable target ----
# this allows users to install and find the library via `find_package()`.

# the location where the project's version header will be placed should match the project's regular
# header paths
string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)

packageProject(
  NAME ${PROJECT_NAME}
  VERSION ${PROJECT_VERSION}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
  VERSION_HEADER "${VERSION_HEADER_LOCATION}"
  DEPENDENCIES ""
)
