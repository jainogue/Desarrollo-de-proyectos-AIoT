#include "app_audio.hpp"

namespace app
{
    // Inicializa la aplicación de audio:
    // - Configura semáforos para sincronización de tareas
    // - Reserva memoria para buffers de audio
    // - Inicializa el micrófono y abre el códec
    // - Inicializa el intérprete de TensorFlow Lite Micro y reserva el tensor arena
    void AppAudio::init(std::function<void(int, int, float, float,int)> state_callback)
    {
        // Guardar el callback para notificar el estado (llanto o ruido)
        m_state_fn = state_callback;

        // Crear semáforos para sincronizar las tareas de captura y procesamiento
        xFeedSemaphore = xSemaphoreCreateBinary();
        xDetectSemaphore = xSemaphoreCreateBinary();

        // Reservar memoria para los buffers de audio
        m_audio_buffer = new float[AUDIO_BUFFER_SIZE];
        m_features = new float[AUDIO_BUFFER_SIZE];

        // Reservar memoria para los buffers de las tareas
        m_detect_task_buffer = new uint8_t[DETECT_TASK_BUFF_SIZE];
        m_action_task_buffer = new uint8_t[ACTION_TASK_BUFF_SIZE];
        m_feed_task_buffer = new uint8_t[FEED_TASK_BUFF_SIZE];

        // Inicializar el micrófono usando el BSP oficial
        ESP_LOGI("Init", "Inicializando micrófono...");
        m_mic_handle = bsp_audio_codec_microphone_init();
        assert(m_mic_handle);
        esp_codec_dev_set_in_gain(m_mic_handle, 50.0); // Configura la ganancia de entrada
        if (!m_mic_handle)
        {
            ESP_LOGE("Init", "❌ ERROR: No se pudo inicializar el micrófono.");
            return;
        }
        ESP_LOGI("Init", "Micrófono inicializado.");

        // Configurar y abrir el códec de audio
        esp_codec_dev_sample_info_t fs = {
            .bits_per_sample = 16,
            .channel = 2,            // Se solicitan 2 canales (el hardware del ESP32-S3 Box 3 suministra 2)
            .channel_mask = 3,       // Bitmask: 0b11 (habilita ambos canales)
            .sample_rate = 16000,
            .mclk_multiple = I2S_MCLK_MULTIPLE_384,
        };
        esp_err_t err = esp_codec_dev_open(m_mic_handle, &fs);
        if (err != ESP_OK)
        {
            ESP_LOGE("Init", "❌ ERROR: No se pudo abrir el códec. Código: %s", esp_err_to_name(err));
            return;
        }
        ESP_LOGI("Init", "✅ Códec de audio abierto correctamente para captura.");

        // Inicialización del intérprete de TensorFlow Lite Micro
        static tflite::MicroErrorReporter error_reporter;
        static tflite::MicroMutableOpResolver<20> resolver;
        resolver.AddAdd();
        resolver.AddConv2D();
        resolver.AddDequantize();
        resolver.AddFill();
        resolver.AddExpandDims();
        resolver.AddFullyConnected();
        resolver.AddMaxPool2D();
        resolver.AddMean();
        resolver.AddMul();
        resolver.AddPack();
        resolver.AddQuantize();
        resolver.AddReshape();
        resolver.AddShape();
        resolver.AddSoftmax();
        resolver.AddStridedSlice();
        resolver.AddTanh();
        resolver.AddTranspose();
        resolver.AddUnpack();

        ESP_LOGI("AppAudio", "Memoria PSRAM libre antes de asignar TENSOR_ARENA: %d bytes",
                  heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

        // Verificar y reservar memoria en PSRAM para el tensor arena
        if (heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) < TENSOR_ARENA_SIZE)
        {
            ESP_LOGE("init_AppAudio", "No hay suficiente memoria en PSRAM para el TENSOR_ARENA");
            return;
        }
        m_tensor_arena = (uint8_t *)heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
        if (!m_tensor_arena)
        {
            ESP_LOGE("init_AppAudio", "Error: No se pudo asignar memoria en PSRAM para TENSOR_ARENA");
            return;
        }
        ESP_LOGI("init_AppAudio", "Memoria PSRAM libre después de asignar TENSOR_ARENA: %d bytes",
                  heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
        ESP_LOGI("init_AppAudio", "TENSOR_ARENA asignado en PSRAM");

        // Cargar el modelo TFLite y crear el intérprete
        static const tflite::Model *model = tflite::GetModel(model_tflite);
        m_interpreter = new tflite::MicroInterpreter(model, resolver, m_tensor_arena, TENSOR_ARENA_SIZE, nullptr, nullptr);
        if (m_interpreter->AllocateTensors() != kTfLiteOk)
        {
            ESP_LOGE("init_AppAudio", "Error al asignar tensores del modelo");
        }
        m_input = m_interpreter->input(0);
    }

    // Crea las tareas de FreeRTOS en núcleos distintos:
    // - Núcleo 0: captura de audio (feedTask)
    // - Núcleo 1: procesamiento y ejecución de la inferencia (detectTask y actionTask)
    void AppAudio::start(void)
    {
        xTaskCreateStaticPinnedToCore(feedTask, "feed", FEED_TASK_BUFF_SIZE, this, 5, m_feed_task_buffer, &m_feed_task_data, 0);
        xTaskCreateStaticPinnedToCore(detectTask, "detect", DETECT_TASK_BUFF_SIZE, this, 5, m_detect_task_buffer, &m_detect_task_data, 1);
        xTaskCreateStaticPinnedToCore(actionTask, "action", ACTION_TASK_BUFF_SIZE, this, 5, m_action_task_buffer, &m_action_task_data, 1);
    }

    // Tarea feedTask:
    // - Captura audio a través del códec
    // - Imprime las primeras 10 muestras capturadas (para diagnóstico)
    // - Normaliza las muestras y las almacena en un buffer circular
    void AppAudio::feedTask(void *arg)
    {
        AppAudio *obj = static_cast<AppAudio *>(arg);

        // Configuración: se leen AUDIO_BUFFER_SIZE muestras por canal
        int audio_chunksize = AUDIO_BUFFER_SIZE;
        int feed_channel = 2;
        int16_t *i2s_buff = (int16_t *)heap_caps_malloc(audio_chunksize * feed_channel * sizeof(int16_t), MALLOC_CAP_SPIRAM);
        if (!i2s_buff)
        {
            ESP_LOGE("feedTask", "❌ ERROR: No se pudo asignar memoria para i2s_buff.");
            vTaskDelete(NULL);
        }

        while (true)
        {
            esp_err_t err = esp_codec_dev_read(obj->m_mic_handle, i2s_buff, audio_chunksize * sizeof(int16_t) * feed_channel);
            if (err != ESP_OK)
            {
                ESP_LOGE("feedTask", "❌ ERROR en esp_codec_dev_read(): %s", esp_err_to_name(err));
                vTaskDelay(pdMS_TO_TICKS(100));
                continue;
            }

            // Imprimir las primeras 10 muestras del buffer capturado para diagnóstico
            /*ESP_LOGI("feedTask", "Primeras muestras del i2s_buff:");
            for (int i = 0; i < 10; i++) {
                ESP_LOGI("feedTask", "i2s_buff[%d] = %d", i, i2s_buff[i]);
            }*/

            // Convertir cada muestra int16_t a float en el rango [-1,1]
            // y almacenar en el buffer circular m_audio_buffer
            int16_t max_val = 0;
            for (int i = 0; i < audio_chunksize * feed_channel; i++) {
                int16_t v = std::abs(i2s_buff[i]);
                if (v > max_val) {
                    max_val = v;
                }
            }
            float divisor = (max_val > 0) ? static_cast<float>(max_val) : 1.0f;
            for (int i = 0; i < audio_chunksize * feed_channel; i++)
            {
                float sample = static_cast<float>(i2s_buff[i]) / divisor;
                obj->m_audio_buffer[obj->m_audio_buffer_ptr] = sample;
                obj->m_audio_buffer_ptr = (obj->m_audio_buffer_ptr + 1) % AUDIO_BUFFER_SIZE;
            }
            // Liberar el semáforo para que detectTask procese los datos
            xSemaphoreGive(xFeedSemaphore);
            vTaskDelay(pdMS_TO_TICKS(1400));  // Delay para permitir el procesamiento
        }
        esp_codec_dev_close(obj->m_mic_handle);
        free(i2s_buff);
        vTaskDelete(NULL);
    }

    // Tarea detectTask:
    // - Copia AUDIO_BUFFER_SIZE muestras del buffer circular m_audio_buffer a m_features
    // - Libera el semáforo para que actionTask ejecute la inferencia
    void AppAudio::detectTask(void *arg)
    {
        AppAudio *obj = static_cast<AppAudio *>(arg);
        while (true)
        {
            // Espera a que feedTask libere el semáforo
            if (xSemaphoreTake(xFeedSemaphore, portMAX_DELAY) == pdTRUE)
            {
                for (int i = 0; i < AUDIO_BUFFER_SIZE; i++)
                {
                    obj->m_features[i] = obj->m_audio_buffer[(obj->m_audio_buffer_ptr + i) % AUDIO_BUFFER_SIZE];
                }
                xSemaphoreGive(xDetectSemaphore);
            }
        }
    }

    // Tarea actionTask:
    // - Normaliza y cuantiza los datos en m_features para adaptarlos al tensor de entrada
    // - Ejecuta la inferencia del modelo TFLite
    // - Notifica el resultado mediante el callback m_state_fn
    void AppAudio::actionTask(void *arg)
    {
        AppAudio *obj = static_cast<AppAudio *>(arg);
        while (true)
        {
            if (xSemaphoreTake(xDetectSemaphore, portMAX_DELAY) == pdTRUE)
            {

                // Suponiendo que m_features[] contiene los valores en float ya escalados (por ejemplo, en el rango [-1, 1])
                TfLiteTensor *input_tensor = obj->m_interpreter->input(0);

                /* Calcular el máximo valor absoluto para normalizar la ventana
                float max_abs = 0.0f;
                for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
                    float abs_val = fabs(obj->m_features[i]);
                    if (abs_val > max_abs)
                        max_abs = abs_val;
                }
                // Evitar división por cero
                if (max_abs < 1e-6) max_abs = 1.0f;
                */

                for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
                    // Aplicar la cuantización estricta: usar directamente la escala y el zero_point del modelo
                    obj->m_input->data.f[i] = obj->m_features[i];
                }



                /*ESP_LOGI("actionTask", "Primeros valores de m_input->data.int8:");
                for (int i = 0; i < 10; i++) {
                    ESP_LOGI("actionTask", "m_input->data.int8[%d] = %d", i, static_cast<int>(obj->m_input->data.int8[i]));
                }*/

                // Ejecutar la inferencia y medir el tiempo de ejecución
                uint32_t start_time = xTaskGetTickCount();
                if (obj->m_interpreter->Invoke() == kTfLiteOk)
                {
                    float *output = obj->m_interpreter->output(0)->data.f;
                    TfLiteTensor *output_tensor = obj->m_interpreter->output(0);

                    uint32_t end_time = xTaskGetTickCount();
                    int inference_time_ms = (end_time - start_time) * portTICK_PERIOD_MS;
                    int used_ram = obj->m_interpreter->arena_used_bytes();
                    //ESP_LOGI("actionTask", "Clase 0 (Cry): %.6f", output[0]);
                    //ESP_LOGI("actionTask", "Clase 1 (Noise): %.6f", output[1]);

                    // Notificar si hay un cambio en el estado
                    int result_idx = (output[0] > output[1]) ? 0 : 1;
                    obj->m_state_fn(result_idx, inference_time_ms, output[0], output[1], used_ram);
                    obj->m_last_state = result_idx;

                    //ESP_LOGI("actionTask", "Tiempo de inferencia: %d ms", (end_time - start_time) * portTICK_PERIOD_MS);
                }
                else
                {
                    ESP_LOGE("actionTask", "Error ejecutando modelo TFLite");
                    continue;
                }
            }
        }
    }
}
