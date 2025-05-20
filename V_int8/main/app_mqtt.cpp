#include "app_mqtt.hpp"


namespace app
{
    static const char* TAG = "AppMqtt";
    void AppMqtt::mqtt_event_handler_cb(void* handler_args, esp_event_base_t base, int32_t event_id, void* event_data)
    {
        esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
        esp_mqtt_client_handle_t client = event->client;

        switch (event->event_id)
        {
            case MQTT_EVENT_CONNECTED:
                ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
                break;

            case MQTT_EVENT_DISCONNECTED:
                ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
                break;

            case MQTT_EVENT_DATA:
                ESP_LOGI(TAG, "MQTT_EVENT_DATA, received topic: %.*s, data: %.*s",
                         event->topic_len, event->topic, event->data_len, event->data);
                break;

            default:
                ESP_LOGI(TAG, "Unhandled event ID: %d", event->event_id);
                break;
        }
    }
    void AppMqtt::mqtt_task(void* pvParameters)
    {
        AppMqtt* self = static_cast<AppMqtt*>(pvParameters);
        ESP_LOGI(TAG, "MQTT Task iniciada en Core 0");

        self->m_mqtt_client = esp_mqtt_client_init(&self->m_mqtt_cfg);

        esp_mqtt_client_register_event(self->m_mqtt_client, (esp_mqtt_event_id_t)ESP_EVENT_ANY_ID, mqtt_event_handler_cb, self);

        esp_err_t err = esp_mqtt_client_start(self->m_mqtt_client);
        if (err != ESP_OK)
        {
            ESP_LOGE(TAG, "Error iniciando MQTT: %s", esp_err_to_name(err));
            vTaskDelete(NULL);
        }

        while (true)
        {
            vTaskDelay(pdMS_TO_TICKS(5000)); // Mantener la tarea en espera
        }
    }

    void AppMqtt::init(const char* username, const char* password)
    {
        // Configurar parámetros de MQTT
        memset(&m_mqtt_cfg, 0, sizeof(m_mqtt_cfg));
        m_mqtt_cfg.broker.address.uri = "mqtt://" CONFIG_MQTT_BROKER_IP;
        m_mqtt_cfg.broker.address.port = CONFIG_MQTT_PORT;
        m_mqtt_cfg.credentials.username = username;
        m_mqtt_cfg.credentials.authentication.password = password;

        // Crear tarea para ejecutar MQTT en Core 0
        xTaskCreatePinnedToCore(mqtt_task, "mqtt_task", 4096, this, 5, NULL, 0);
    }

    void AppMqtt::publish(const char* topic, const char* data)
    {
        if (!m_mqtt_client)
        {
            ESP_LOGE(TAG, "Error: MQTT no está inicializado.");
            return;
        }
        int msg_id = esp_mqtt_client_publish(m_mqtt_client, topic, data, 0, 1, 0);
        if (msg_id == -1)
        {
            ESP_LOGE(TAG, "Error: No se pudo publicar en el tópico %s", topic);
        }
        else
        {
            ESP_LOGI(TAG, "Publicado: msg_id=%d, topic=%s, data=%s", msg_id, topic, data);
        }
    }
}
