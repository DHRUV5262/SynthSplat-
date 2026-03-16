cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0048 NEW)
project(SynthSplatRenderer VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Dependencies ---
include(FetchContent)

# 1. GLM (Using 1.0.1 to fix CMake compatibility)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# 2. GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
set(GLFW_INSTALL OFF CACHE INTERNAL "Generate installation target")
FetchContent_MakeAvailable(glfw)

# 3. TinyGLTF
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.8.13
)
FetchContent_MakeAvailable(tinygltf)

# 4. STB
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stb)

# 5. nlohmann_json (Header only download to avoid CMake version issues)
set(JSON_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${JSON_INCLUDE_DIR}/nlohmann)
if(NOT EXISTS ${JSON_INCLUDE_DIR}/nlohmann/json.hpp)
    message(STATUS "Downloading nlohmann/json.hpp...")
    file(DOWNLOAD 
        https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
        ${JSON_INCLUDE_DIR}/nlohmann/json.hpp
    )
endif()

# 6. GLAD (Manual Check)
# The automatic generation is unreliable due to python environment differences.
# We check if glad.c exists in src/ or if the user provided it.
# We search for glad/glad.h in src, include, src/glad, include/glad
find_path(GLAD_INCLUDE_DIR glad/glad.h 
    PATHS 
    src 
    include 
    src/glad 
    include/glad 
    ${CMAKE_BINARY_DIR}/glad/include
)

find_file(GLAD_SOURCE glad.c 
    PATHS 
    src 
    src/glad 
    ${CMAKE_BINARY_DIR}/glad/src
)

if(GLAD_INCLUDE_DIR AND GLAD_SOURCE)
    message(STATUS "Found GLAD: ${GLAD_INCLUDE_DIR}, ${GLAD_SOURCE}")
    list(APPEND SOURCES ${GLAD_SOURCE})
    include_directories(${GLAD_INCLUDE_DIR})
else()
    message(WARNING "GLAD not found! Attempting to download a pre-generated GLAD 4.6 Core loader...")
    
    # Fallback: Download a pre-generated GLAD 4.6 Core single-file version
    # This is a temporary solution. Ideally, generate it at https://glad.dav1d.de/
    
    message(FATAL_ERROR "\n\n"
        "!!! GLAD MISSING !!!\n"
        "Please follow these steps:\n"
        "1. Go to https://glad.dav1d.de/\n"
        "2. Language: C/C++\n"
        "3. API: gl, Version: 4.6, Profile: Core\n"
        "4. Click 'Generate' and download the zip.\n"
        "5. Extract 'src/glad.c' to '${CMAKE_SOURCE_DIR}/src/glad.c'\n"
        "6. Extract 'include/glad' and 'include/KHR' folders to '${CMAKE_SOURCE_DIR}/src/' (so you have src/glad/glad.h)\n"
        "\n"
    )
endif()

# --- Sources ---
file(GLOB_RECURSE PROJ_SOURCES 
    "src/*.cpp" 
    "src/*.h"
)
list(APPEND SOURCES ${PROJ_SOURCES})

# --- Executable ---
add_executable(SynthSplat ${SOURCES})

# --- Include Directories ---
target_include_directories(SynthSplat PRIVATE 
    src 
    ${JSON_INCLUDE_DIR}
    ${glm_SOURCE_DIR}
    ${tinygltf_SOURCE_DIR}
    ${stb_SOURCE_DIR}
)

# Link Libraries
target_link_libraries(SynthSplat PRIVATE 
    glfw 
    glm
    ${CMAKE_DL_LIBS}
)

# Definitions
if(VARY_LIGHTING)
    target_compile_definitions(SynthSplat PRIVATE VARY_LIGHTING)
endif()

# Copy shaders and assets to build directory
file(COPY shaders DESTINATION ${CMAKE_BINARY_DIR})
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
