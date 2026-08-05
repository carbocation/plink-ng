cmake_minimum_required(VERSION 3.20)

project(pgen_rans VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(CTest)
  option(
    PGEN_RANS_BUILD_TESTING
    "Build the conditional-rANS pgenlib adapter tests"
    ${BUILD_TESTING}
  )
  option(
    PGEN_RANS_INSTALL
    "Install the conditional-rANS pgenlib adapter package"
    ON
  )
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

add_library(pgen_rans_pgenlib_core STATIC
  include/plink2_base.cc
  include/plink2_bits.cc
  include/plink2_simd.cc
  include/pgenlib_misc.cc
  include/pgenlib_read.cc
  include/pgenlib_ffi_support.cc
)
add_library(pgen_rans::pgenlib_core ALIAS pgen_rans_pgenlib_core)
set_target_properties(pgen_rans_pgenlib_core PROPERTIES
  EXPORT_NAME pgenlib_core
)
target_compile_features(pgen_rans_pgenlib_core PUBLIC cxx_std_17)
target_include_directories(pgen_rans_pgenlib_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pgen_rans/include>
)
target_link_libraries(pgen_rans_pgenlib_core PUBLIC ${CMAKE_DL_LIBS})

add_library(pgen_rans_decode STATIC
  pgen_rans_codec.cc
  pgen_rans_hybrid.cc
  pgen_rans_container.cc
  pgen_rans_cpu.cc
  pgen_rans_reader.cc
)
add_library(pgen_rans::decode ALIAS pgen_rans_decode)
set_target_properties(pgen_rans_decode PROPERTIES
  EXPORT_NAME decode
)
target_compile_features(pgen_rans_decode PUBLIC cxx_std_17)
target_include_directories(pgen_rans_decode PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pgen_rans>
)
target_link_libraries(pgen_rans_decode PUBLIC Threads::Threads)

add_library(pgen_rans_pgenlib STATIC pgen_rans_pgenlib.cc)
add_library(pgen_rans::pgenlib ALIAS pgen_rans_pgenlib)
set_target_properties(pgen_rans_pgenlib PROPERTIES
  EXPORT_NAME pgenlib
)
target_compile_features(pgen_rans_pgenlib PUBLIC cxx_std_17)
target_include_directories(pgen_rans_pgenlib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pgen_rans>
)
target_link_libraries(pgen_rans_pgenlib PUBLIC
  pgen_rans_decode
  pgen_rans_pgenlib_core
)

if(PGEN_RANS_BUILD_TESTING)
  add_executable(pgen_rans_codec_test pgen_rans_codec_test.cc)
  target_link_libraries(pgen_rans_codec_test PRIVATE pgen_rans_decode)
  add_test(NAME pgen_rans_codec_test COMMAND pgen_rans_codec_test)

  add_executable(pgen_rans_hybrid_test pgen_rans_hybrid_test.cc)
  target_link_libraries(pgen_rans_hybrid_test PRIVATE pgen_rans_decode)
  add_test(NAME pgen_rans_hybrid_test COMMAND pgen_rans_hybrid_test)

  add_executable(pgen_rans_container_test pgen_rans_container_test.cc)
  target_link_libraries(pgen_rans_container_test PRIVATE pgen_rans_decode)
  add_test(NAME pgen_rans_container_test COMMAND pgen_rans_container_test)

  add_executable(
    pgenlib_sparse_backend_test pgenlib_sparse_backend_test.cc
  )
  target_link_libraries(
    pgenlib_sparse_backend_test PRIVATE pgen_rans_pgenlib_core
  )
  add_test(
    NAME pgenlib_sparse_backend_test COMMAND pgenlib_sparse_backend_test
  )

  add_executable(pgen_rans_pgenlib_test pgen_rans_pgenlib_test.cc)
  target_link_libraries(pgen_rans_pgenlib_test PRIVATE pgen_rans_pgenlib)
  add_test(NAME pgen_rans_pgenlib_test COMMAND pgen_rans_pgenlib_test)
endif()

if(PGEN_RANS_INSTALL)
  install(TARGETS
      pgen_rans_pgenlib_core
      pgen_rans_decode
      pgen_rans_pgenlib
    EXPORT PgenRansTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  install(FILES
      pgen_rans_codec.h
      pgen_rans_container.h
      pgen_rans_cpu.h
      pgen_rans_hybrid.h
      pgen_rans_pgenlib.h
      pgen_rans_reader.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pgen_rans
  )
  install(FILES
      include/pgenlib_ffi_support.h
      include/pgenlib_misc.h
      include/pgenlib_read.h
      include/plink2_base.h
      include/plink2_bits.h
      include/plink2_simd.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pgen_rans/include
  )
  install(DIRECTORY simde/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pgen_rans/simde
    FILES_MATCHING PATTERN "*.h"
  )
  install(EXPORT PgenRansTargets
    FILE PgenRansTargets.cmake
    NAMESPACE pgen_rans::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pgen_rans
  )

  configure_package_config_file(
    cmake/pgen_ransConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/pgen_ransConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pgen_rans
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/pgen_ransConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )
  install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/pgen_ransConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/pgen_ransConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pgen_rans
  )
  install(FILES COPYING COPYING.LESSER
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/pgen_rans
  )
  install(FILES pgen_rans_format.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/pgen_rans
  )
endif()
