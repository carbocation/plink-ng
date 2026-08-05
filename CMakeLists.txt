cmake_minimum_required(VERSION 3.20)

project(plink_ng_rans VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  include(CTest)
  set(PGEN_RANS_BUILD_TESTING_DEFAULT ${BUILD_TESTING})
  set(PGEN_RANS_INSTALL_DEFAULT ON)
else()
  set(PGEN_RANS_BUILD_TESTING_DEFAULT OFF)
  set(PGEN_RANS_INSTALL_DEFAULT OFF)
endif()
option(
  PGEN_RANS_BUILD_TESTING
  "Build the conditional-rANS pgenlib adapter tests"
  ${PGEN_RANS_BUILD_TESTING_DEFAULT}
)
unset(PGEN_RANS_BUILD_TESTING_DEFAULT)
option(
  PGEN_RANS_INSTALL
  "Install the conditional-rANS pgenlib adapter package"
  ${PGEN_RANS_INSTALL_DEFAULT}
)
unset(PGEN_RANS_INSTALL_DEFAULT)
if(PGEN_RANS_BUILD_TESTING)
  enable_testing()
endif()

add_subdirectory(2.0)
