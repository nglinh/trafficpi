cmake_minimum_required(VERSION 2.8)
project(trafficpi)
find_package(OpenCV REQUIRED)

SET(RASPICAM_CV $ENV{HOME}/robidouille/raspicam_cv)
SET(USERLAND $ENV{HOME}/raspberrypi/userland)
include_directories(${RASPICAM_CV})
include_directories(${Tesseract_INCLUDE_DIRS})
SET(MMAL_LIBS
    ${USERLAND}/build/lib/libmmal_core.so
    ${USERLAND}/build/lib/libmmal_util.so
    ${USERLAND}/build/lib/libmmal_vc_client.so
    ${USERLAND}/build/lib/libvcos.so)
SET(BCM_LIBS ${USERLAND}/build/lib/libbcm_host.so)
SET(OPENALPR_LIB /usr/lib/libopenalpr.so)
SET(DISPLAY_CAMERA_LIBS
    ${RASPICAM_CV}/libraspicamcv.a
    ${MMAL_LIBS} ${BCM_LIBS} ${OpenCV_LIBS})

# Pi executables
SET(TRAFFIC_PI trafficpi)
add_executable(${TRAFFIC_PI} main.cpp)
target_link_libraries(${TRAFFIC_PI} ${DISPLAY_CAMERA_LIBS} ${Tesseract_LIBRARIES} ${OPENALPR_LIB})
