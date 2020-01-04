cmake_minimum_required(VERSION 3.13)
project(CLion)

# Require OpenMP for multithreading
find_package(OpenMP REQUIRED)

# Set requirements and C++ standard
set(CMAKE_CXX_STANDARD 14)

# Bring in all cpp files in /src. Note: When a .cpp file is added, you must rerun cmake due to this line
file(GLOB SOURCES
        "../../src/*.cpp"
)

# Specify the executable we want to build
add_executable(CLion ${SOURCES})

# Set include directory
target_include_directories(CLion PUBLIC ../../header)

# Use OpenMP
target_link_libraries(CLion PRIVATE OpenMP::OpenMP_CXX)
