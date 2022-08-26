cmake_minimum_required(VERSION 3.12)

project(daxa-template)

find_package(daxa CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} "main.cpp" "imgui/imgui_impl_glfw.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE daxa::daxa glm::glm glfw)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
