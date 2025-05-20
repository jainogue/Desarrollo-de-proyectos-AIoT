#include <stdio.h>
#include "esp_err.h" 
#include "esp_log.h"
#include "freertos/FreeRTOS.h" 
#include "freertos/task.h"
#include "string"
#include <iostream>
#include <cstdlib>
#include "sdkconfig.h" 

#include "app_mem.hpp"
#include "app_audio.hpp"
#include "app_mqtt.hpp"
#include "AppWifiBle.hpp"



#define TAG "app"

namespace
{
    app::AppAudio app_audio;
    app::AppMem app_mem;
    app::AppMqtt app_mqtt;
    app::AppWifiBle app_wifi;

    const char *CATEGORIES[] = {"Llanto", "Ruido"};
    // Callback de detección de audio
    void audio_state_callback(int state, int inference_time_ms, float output_class_0, float output_class_1, int used_ram)
    {
        const char *detected_state = CATEGORIES[state];
        
        // Construir el mensaje JSON con las métricas relevantes
        char mqtt_msg[256];
        snprintf(mqtt_msg, sizeof(mqtt_msg),
                 "%s, %d, %.6f, %.6f, %d",
                 detected_state, inference_time_ms, output_class_0, output_class_1, used_ram);
    
        // Imprimir en terminal el mensaje completo para fines de depuración
        ESP_LOGI("BabyMonitor", "Detección: %s", mqtt_msg);
    
        // Publicar el mensaje en MQTT
        app_mqtt.publish("Bedroom/sensor/baby_monitor", mqtt_msg);
    }
}

extern "C" void app_main(void)
{
    ESP_LOGI("Main", "Iniciando ESP32-S3 Box 3...");
    auto wifi_connected = [=](esp_ip4_addr_t *ip)
    {
        app_mqtt.init(CONFIG_MQTT_USER, CONFIG_MQTT_PWD);
        ESP_LOGI(TAG,"wifi connected  and mqtt init");
    };

    auto  wifi_disconnected =[]()
    {
        ESP_LOGW(TAG,"wifi disconnected");
    };

    app_wifi.init(wifi_connected,wifi_disconnected);
    app_wifi.connect();
    app_audio.init(audio_state_callback);
    // Print memory stats after initializing audio
    app_mem.print();
    app_audio.start();
    ESP_LOGI("Main", "Detección de audio iniciada");
    app_mem.monitor();
    // Bucle infinito (el ESP32 ejecuta las tareas en FreeRTOS)
    while (true)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

}

