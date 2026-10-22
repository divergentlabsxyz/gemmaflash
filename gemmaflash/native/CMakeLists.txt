cmake_minimum_required(VERSION 3.27)
set(CMAKE_OSX_DEPLOYMENT_TARGET 26.2)
project(xgate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mlx, os; print(os.path.join(mlx.__path__[0], 'share/cmake/MLX'))"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_DIR)
find_package(MLX CONFIG REQUIRED)

# Same nanobind (3.1.0) and ABI as mlx.core, so mx.array crosses over as mlx::core::array.
nanobind_add_module(_xgate NB_STATIC FREE_THREADED NOMINSIZE NB_DOMAIN mlx xgate.cpp)
target_include_directories(_xgate PRIVATE ${MLX_INCLUDE_DIRS})
target_link_libraries(_xgate PRIVATE mlx)
# Resolve libmlx from the installed mlx package at run time.
get_filename_component(MLX_LIB_DIR "${MLX_LIBRARY}" DIRECTORY)
set_target_properties(_xgate PROPERTIES BUILD_RPATH "${MLX_LIB_DIR}")
