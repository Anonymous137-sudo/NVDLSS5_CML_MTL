cmake_minimum_required(VERSION 3.28)
project(obs-neural-post LANGUAGES C OBJC)
set(CMAKE_OSX_DEPLOYMENT_TARGET 14.0)
set(OBS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../obs-studio" CACHE PATH "OBS source matching target app ABI")
set(OBS_APP "/Applications/OBS.app" CACHE PATH "OBS application for linking")
set(NEURAL_POST_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/.build/release" CACHE PATH "Built Swift release products")
set(SIMDE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps/simde-0.8.2" CACHE PATH "SIMDe include root")
set(OBS_DATA_PATH "data/obs-studio")
set(OBS_PLUGIN_PATH "obs-plugins")
set(OBS_PLUGIN_DESTINATION "obs-plugins")
set(OBS_RELEASE_CANDIDATE 0)
set(OBS_BETA 0)
configure_file("${OBS_SOURCE_DIR}/libobs/obsconfig.h.in" "${CMAKE_CURRENT_BINARY_DIR}/generated/obsconfig.h")
add_library(obs-neural-post MODULE src/obs-neural-post.m)
target_include_directories(obs-neural-post PRIVATE "${OBS_SOURCE_DIR}/libobs" "${CMAKE_CURRENT_BINARY_DIR}/generated" "${SIMDE_INCLUDE_DIR}")
target_compile_options(obs-neural-post PRIVATE -fobjc-arc -fmodules -Wall -Wextra)
if(TARGET OBS::libobs)
  target_link_libraries(obs-neural-post PRIVATE OBS::libobs)
else()
  target_link_libraries(obs-neural-post PRIVATE "${OBS_APP}/Contents/Frameworks/libobs.framework/libobs")
endif()
target_link_libraries(obs-neural-post PRIVATE "${NEURAL_POST_BUILD}/libNeuralPost.dylib"
    "-framework AVFoundation" "-framework CoreVideo" "-framework IOSurface" "-framework Foundation")
set_target_properties(obs-neural-post PROPERTIES PREFIX "" SUFFIX "" BUNDLE TRUE BUNDLE_EXTENSION plugin
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/src/Info.plist.in"
    BUILD_RPATH "@loader_path/../Frameworks;${OBS_APP}/Contents/Frameworks")
add_custom_command(TARGET obs-neural-post POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_BUNDLE_DIR:obs-neural-post>/Contents/MacOS/mlx-swift_Cmlx.bundle"
    COMMAND ${CMAKE_COMMAND} -E rm -f "$<TARGET_BUNDLE_DIR:obs-neural-post>/Contents/Frameworks/mlx.metallib"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_DIR:obs-neural-post>/Contents/Frameworks"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_DIR:obs-neural-post>/Contents/Resources"
    COMMAND ${CMAKE_COMMAND} -E copy "${NEURAL_POST_BUILD}/libNeuralPost.dylib" "$<TARGET_BUNDLE_DIR:obs-neural-post>/Contents/Frameworks/"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/models/rt4ksr-x3.safetensors" "$<TARGET_BUNDLE_DIR:obs-neural-post>/Contents/Resources/rt4ksr-x3.safetensors"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/licenses/RT4KSR-LICENSE" "$<TARGET_BUNDLE_DIR:obs-neural-post>/Contents/Resources/RT4KSR-LICENSE"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" "$<TARGET_BUNDLE_DIR:obs-neural-post>/Contents/Resources/LICENSE"
    COMMAND ${CMAKE_COMMAND} -E copy "${NEURAL_POST_BUILD}/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" "$<TARGET_BUNDLE_DIR:obs-neural-post>/Contents/Resources/mlx.metallib")
if(NOT TARGET OBS::libobs)
  add_executable(obs-neural-post-load-test tests/obs-load.m)
  target_include_directories(obs-neural-post-load-test PRIVATE "${OBS_SOURCE_DIR}/libobs" "${CMAKE_CURRENT_BINARY_DIR}/generated" "${SIMDE_INCLUDE_DIR}")
  target_compile_definitions(obs-neural-post-load-test PRIVATE NEURAL_POST_OBS_APP="${OBS_APP}")
  target_link_libraries(obs-neural-post-load-test PRIVATE "${OBS_APP}/Contents/Frameworks/libobs.framework/libobs" "-framework AppKit" "-framework CoreVideo")
  set_target_properties(obs-neural-post-load-test PROPERTIES BUILD_RPATH "${OBS_APP}/Contents/Frameworks")
endif()
