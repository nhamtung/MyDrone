# Install script for directory: /home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/example_pkg

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/example_pkg/cmake" TYPE FILE FILES "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/build/example_pkg/catkin_generated/installspace/example_pkg-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/devel/share/roseus/ros/example_pkg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python2" -m compileall "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/devel/lib/python2.7/dist-packages/example_pkg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/devel/lib/python2.7/dist-packages/example_pkg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/build/example_pkg/catkin_generated/installspace/example_pkg.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/example_pkg/cmake" TYPE FILE FILES "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/build/example_pkg/catkin_generated/installspace/example_pkg-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/example_pkg/cmake" TYPE FILE FILES
    "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/build/example_pkg/catkin_generated/installspace/example_pkgConfig.cmake"
    "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/build/example_pkg/catkin_generated/installspace/example_pkgConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/example_pkg" TYPE FILE FILES "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/example_pkg/package.xml")
endif()

