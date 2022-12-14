cmake_minimum_required(VERSION 3.21)
project(c__qt)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_PREFIX_PATH F:\\Qt\\5.9\\mingw53_32)


find_package(Qt5 REQUIRED COMPONENTS
        Core
        Gui
        Widgets)

aux_source_directory(. DIRS_SRCS)

add_executable(demo ${DIRS_SRCS} mainwindow.cpp mainwindow.h mainwindow.ui)

target_link_libraries(demo Qt5::Widgets)