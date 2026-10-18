file(GLOB VAD_SOURCES "webrtc/common_audio/vad/*.c"
                      "webrtc/common_audio/signal_processing/*.c")
idf_component_register(SRCS ${VAD_SOURCES} INCLUDE_DIRS "." REQUIRES pthread)
target_compile_definitions(${COMPONENT_LIB} PRIVATE WEBRTC_POSIX)
# This pinned fixed-point upstream code predates the current GCC warning set.
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error -fwrapv)
