
if(NOT CMAKE_BUILD_EARLY_EXPANSION)
# add_definitions(-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=1) # enables ESP-NN optimizations by Espressif
# disabled by default - enable if you use ESP32S3
add_definitions(-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN_S3=1) # enables ESP-NN S3 optimizations by Espressif
endif()

idf_component_register(
    SRCS "app_mem.cpp" "model.cpp" "app_audio.cpp" "main.cpp" "app_mqtt.cpp"
    INCLUDE_DIRS ""
)