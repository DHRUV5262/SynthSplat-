# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/C++/3D SynthSplat-/renderer/build/_deps/glad_repo-src")
  file(MAKE_DIRECTORY "D:/C++/3D SynthSplat-/renderer/build/_deps/glad_repo-src")
endif()
file(MAKE_DIRECTORY
  "D:/C++/3D SynthSplat-/renderer/build/_deps/glad_repo-build"
  "D:/C++/3D SynthSplat-/renderer/build/_deps/glad_repo-subbuild/glad_repo-populate-prefix"
  "D:/C++/3D SynthSplat-/renderer/build/_deps/glad_repo-subbuild/glad_repo-populate-prefix/tmp"
  "D:/C++/3D SynthSplat-/renderer/build/_deps/glad_repo-subbuild/glad_repo-populate-prefix/src/glad_repo-populate-stamp"
  "D:/C++/3D SynthSplat-/renderer/build/_deps/glad_repo-subbuild/glad_repo-populate-prefix/src"
  "D:/C++/3D SynthSplat-/renderer/build/_deps/glad_repo-subbuild/glad_repo-populate-prefix/src/glad_repo-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/C++/3D SynthSplat-/renderer/build/_deps/glad_repo-subbuild/glad_repo-populate-prefix/src/glad_repo-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/C++/3D SynthSplat-/renderer/build/_deps/glad_repo-subbuild/glad_repo-populate-prefix/src/glad_repo-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
