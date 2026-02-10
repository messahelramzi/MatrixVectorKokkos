cmake_minimum_required(VERSION 3.31)

project(MATVEC LANGUAGES CXX CUDA)

set(EXPORT_COMPILE_COMMANDS ON)

# Find packages requirements
find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
    if(EXISTS ${Kokkos_COMMON_SOURCE_DIR})
      add_subdirectory(${Kokkos_COMMON_SOURCE_DIR} Kokkos)
    else()
      include(FetchContent)
      FetchContent_Declare(
        Kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        4.7.01
      )
    FetchContent_MakeAvailable(Kokkos)
  endif()
endif()

find_package(KokkosKernels CONFIG)

find_package(CLI11 CONFIG)
if(NOT CLI11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.1.3
  )
  FetchContent_MakeAvailable(CLI11)
endif()

find_package(CUDAToolkit REQUIRED)

# Build Matvec executable
add_executable(matvec matvec.cpp)
target_link_libraries(matvec PRIVATE Kokkos::kokkos CLI11::CLI11)

# Build matvec_cublas executable (CUDA/cuBLAS)
add_executable(matvec_cublas matvec_cublas.cu)
target_link_libraries(matvec_cublas PRIVATE CUDA::cublas)

add_custom_command(TARGET matvec POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/data/A.csv
    ${CMAKE_BINARY_DIR}/A.csv
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/data/x.csv
    ${CMAKE_BINARY_DIR}/x.csv
  COMMENT "Copying CSV files to build directory"
)

# Build Matvec_kk executable
if(KokkosKernels_FOUND)
  message(STATUS "Found KokkosKernels: ${KokkosKernels_DIR} (version \"${KokkosKernels_VERSION}\")")
  add_executable(matvec_kk matvec_kk.cpp)
  target_link_libraries(matvec_kk PRIVATE Kokkos::kokkos Kokkos::kokkoskernels CLI11::CLI11)
else()
  message(FATAL_ERROR "Could not find KokkosKernels matvec_kk executable will not be built.")
endif()