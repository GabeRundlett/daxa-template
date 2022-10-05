cmake_minimum_required(VERSION 3.12)

project(daxa-template)

find_package(daxa CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE daxa::daxa glm::glm glfw)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

set(DAXA_INCLUDE_DIR "$<TARGET_FILE_DIR:${PROJECT_NAME}>/../vcpkg_installed/x64-$<LOWER_CASE:$<PLATFORM_ID>>/include")
target_compile_definitions(${PROJECT_NAME} PRIVATE DAXA_SHADER_INCLUDE_DIR="${DAXA_INCLUDE_DIR}")
