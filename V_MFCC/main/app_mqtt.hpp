#pragma once

#include "mqtt_client.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace app
{
    class AppMqtt
    {
    private:
        esp_mqtt_client_handle_t m_mqtt_client;
        esp_mqtt_client_config_t m_mqtt_cfg;
        static void mqtt_event_handler_cb(void* handler_args, esp_event_base_t base, int32_t event_id, void* event_data);
        static void mqtt_task(void* pvParameters);

    public:
        void init(const char* username, const char* password);
        void publish(const char* topic, const char* data);
    };
}