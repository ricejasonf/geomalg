cmake_minimum_required(VERSION 3.13.4)

add_library(Geomalg MODULE Geomalg.cpp GeomalgDialect.cpp)
target_compile_options(Geomalg PRIVATE -fno-rtti -fno-exceptions)
add_dependencies(Geomalg GeomalgOpsIncGen)
install(TARGETS Geomalg
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
