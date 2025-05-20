#include "app_audio.hpp"

using Complex = std::complex<float>;
namespace app
{
    float* AppAudio::mfcc_frame_buffer = nullptr;
    float* AppAudio::mfcc_power_spectrum = nullptr;
    float* AppAudio::mfcc_dct_buffer = nullptr;
    float* AppAudio::mfcc_hamming_window = nullptr;
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
        m_audio_buffer_int16 = new int[AUDIO_BUFFER_SIZE];
        m_features = new float[AUDIO_BUFFER_SIZE];

        __attribute__((aligned(16))) mfcc_frame_buffer = new float[FRAME_LENGTH]; // Buffer para el frame de audio
        __attribute__((aligned(16))) mfcc_dct_buffer = new float[NUM_MFCC_COEFFS]; // Buffer para la DCT
        __attribute__((aligned(16))) mfcc_power_spectrum = new float[FRAME_LENGTH / 2 + 1]; // Buffer para el espectro de potencia
        __attribute__((aligned(16))) mfcc_hamming_window = new float[FRAME_LENGTH]; // Buffer para la ventana de Hamming


        // Reservar memoria para los buffers de las tareas
        m_detect_task_buffer = new uint8_t[DETECT_TASK_BUFF_SIZE];
        m_action_task_buffer = new uint8_t[ACTION_TASK_BUFF_SIZE];
        m_feed_task_buffer = new uint8_t[FEED_TASK_BUFF_SIZE];

        esp_err_t err = dsps_fft2r_init_fc32(NULL, FRAME_LENGTH >> 1); // Inicializar la FFT de 512 puntos (256 complejos)

        // Inicializar el micrófono usando el BSP oficial
        ESP_LOGI("Init", "Inicializando micrófono...");
        m_mic_handle = bsp_audio_codec_microphone_init();
        assert(m_mic_handle);
        esp_codec_dev_set_in_gain(m_mic_handle, 50.0); // Configura la ganancia de entrada
        //esp_codec_dev_set_in_gain(m_mic_handle, 50.0); // Configura la ganancia de entrada
        if (!m_mic_handle)
        {
            ESP_LOGE("Init", "❌ ERROR: No se pudo inicializar el micrófono.");
            return;
        }
        ESP_LOGI("Init", "Micrófono inicializado.");

        // Configurar y abrir el códec de audio
        esp_codec_dev_sample_info_t fs = {
            .bits_per_sample = 16,
            .channel = 1,            // Se solicitan 2 canales (el hardware del ESP32-S3 Box 3 suministra 2)
            .sample_rate = 16000,   // // Frecuencia de muestreo de 16 kHz
            .mclk_multiple = I2S_MCLK_MULTIPLE_384, // Múltiplo de MCLK para la frecuencia de muestreo
        };
        esp_err_t err2 = esp_codec_dev_open(m_mic_handle, &fs);
        if (err2 != ESP_OK)
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
        ESP_LOGI("init_AppAudio", "TENSOR_ARENA asignado en PSRAM");

        // Cargar el modelo TFLite y crear el intérprete
        static const tflite::Model *model = tflite::GetModel(model_tflite);
        m_interpreter = new tflite::MicroInterpreter(model, resolver, m_tensor_arena, TENSOR_ARENA_SIZE);
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
        // Con 2 canales, se leen AUDIO_BUFFER_SIZE * 2 muestras en total
        int audio_chunksize = AUDIO_BUFFER_SIZE;
        int feed_channel = 1;
        int16_t *i2s_buff = (int16_t *)heap_caps_malloc(audio_chunksize * feed_channel * sizeof(int16_t), MALLOC_CAP_DMA);
        if (!i2s_buff)
        {
            ESP_LOGE("feedTask", "ERROR: No se pudo asignar memoria para i2s_buff.");
            vTaskDelete(NULL);
        }

        while (true)
        {
            esp_err_t err = esp_codec_dev_read(obj->m_mic_handle, i2s_buff, audio_chunksize * sizeof(int16_t) * feed_channel);
            if (err != ESP_OK)
            {
                ESP_LOGE("feedTask", "ERROR en esp_codec_dev_read(): %s", esp_err_to_name(err));
                vTaskDelay(pdMS_TO_TICKS(100));
                continue;
            }

            // Convertir la señal estéreo a mono promediando ambos canales.
            // Cada par de muestras (left, right) se convierte en una sola muestra mono.
            for (int i = 0; i < audio_chunksize; i++)
            {
                /*int index = i * 2;  // Índice para el canal izquierdo
                int16_t left = i2s_buff[index];
                int16_t right = i2s_buff[index + 1];
                int16_t mono = (left + right) / 2;*/
                obj->m_audio_buffer_int16[obj->m_audio_buffer_ptr] = i2s_buff[i]; // Almacenar la muestra en el buffer circular
                obj->m_audio_buffer_ptr = (obj->m_audio_buffer_ptr + 1) % AUDIO_BUFFER_SIZE;
            }

            obj->m_audio_buffer_read_ptr = obj->m_audio_buffer_ptr;  // Guarda la posición de inicio para la lectura
            // Notificar a detectTask que hay nuevos datos disponibles
            xSemaphoreGive(xFeedSemaphore);
            vTaskDelay(pdMS_TO_TICKS(1000));  // Delay para permitir el procesamiento
        }
        esp_codec_dev_close(obj->m_mic_handle);
        free(i2s_buff);
        vTaskDelete(NULL);
    }


    void AppAudio::detectTask(void *arg)
    {
        AppAudio *obj = static_cast<AppAudio *>(arg);
        while (true)
        {
            if (xSemaphoreTake(xFeedSemaphore, portMAX_DELAY) == pdTRUE)
            {
                // Copiar AUDIO_BUFFER_SIZE muestras del buffer circular de int16_t
                float float_audio_buffer[AUDIO_BUFFER_SIZE];
                int read_index = obj->m_audio_buffer_read_ptr;  // Índice fijo del bloque a procesar

                for (int i = 0; i < AUDIO_BUFFER_SIZE; i++)
                {
                    float_audio_buffer[i] = static_cast<float>(obj->m_audio_buffer_int16[(read_index + i) % AUDIO_BUFFER_SIZE]);
                }

                // Llamar directamente a GenerateMicroFeatures con datos
                GenerateMicroFeatures(float_audio_buffer, AUDIO_BUFFER_SIZE, obj->m_features);
                // Se podría verificar que el número de coeficientes generados es el esperado
                ESP_LOGI("detectTask", "m_features[0]: %f", obj->m_features[60]);
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
                for (int i = 0; i < MFCC_FEATURE_SIZE; i++) {
                    obj->m_input->data.f[i] = obj->m_features[i];
                }
                /*printf("Contenido del tensor de entrada:\n");
                TfLiteTensor *input_tensor = obj->m_interpreter->input(0);
                int batch = input_tensor->dims->data[0];
                int frames = input_tensor->dims->data[1];
                int coeffs = input_tensor->dims->data[2];
                for (int b = 0; b < batch; b++) {
                    printf("Batch %d:\n", b);
                    for (int f = 0; f < frames; f++) {
                        printf("Frame %d: ", f);
                        for (int c = 0; c < coeffs; c++) {
                            int index = b * frames * coeffs + f * coeffs + c;
                            printf("%f ", input_tensor->data.f[index]);
                        }
                        printf("\n");
                    }
                }*/

                // Ejecutar la inferencia y medir el tiempo de ejecución
                uint32_t start_time = xTaskGetTickCount();
                if (obj->m_interpreter->Invoke() == kTfLiteOk)
                {
                    float *output = obj->m_interpreter->output(0)->data.f;
                    uint32_t end_time = xTaskGetTickCount();
                    int inference_time_ms = (end_time - start_time) * portTICK_PERIOD_MS;
                    int result_idx = (output[CRYING_IDX] > output[NOISE_IDX]) ? CRYING_IDX : NOISE_IDX;
                    int used_ram = obj->m_interpreter->arena_used_bytes();
                    obj->m_state_fn(result_idx, inference_time_ms, output[0], output[1], used_ram);
                    obj->m_last_state = result_idx;
                }
                else
                {
                    ESP_LOGE("actionTask", "Error ejecutando modelo TFLite");
                    continue;
                }
            }
        }
    }

    void AppAudio::GenerateMicroFeatures(float* audio_data, int audio_data_size, float* output_features) {
        const float epsilon = 1e-8f;

        // 1. Normalización del audio: quitar la media y ajustar a rango [-1, 1]
        float sum = 0.0f;
        for (int i = 0; i < audio_data_size; i++) {
            sum += audio_data[i];
        }
        float mean = sum / audio_data_size;
        for (int i = 0; i < audio_data_size; i++) {
            audio_data[i] -= mean;  // Quitar la media (normalización a cero-mean)
        }

        float max_value = 0.0f;
        for (int i = 0; i < audio_data_size; i++) {
            float abs_val = fabsf(audio_data[i]);
            if (abs_val > max_value) {
                max_value = abs_val;
            }
        }

        if (max_value > 0.0f) {
            for (int i = 0; i < audio_data_size; i++) {
                audio_data[i] /= (max_value + epsilon);
            }
        }
        float max_value2 = 0.0f;
        for (int i = 0; i < audio_data_size; i++) {
            float abs_val = fabsf(audio_data[i]);
            if (abs_val > max_value2) {
                max_value2 = abs_val;
            }
        }

        // 2. Parámetros básicos
        const float sample_rate = 16000.0f;           // Frecuencia de muestreo
        const float low_freq = 0.0f;
        const float high_freq = sample_rate / 2.0f;     // 8000 Hz
        const int num_mel_points = NUM_MEL_FILTERS + 2; // Número de puntos Mel (incluyendo extremos)
        const int nfft = FRAME_LENGTH;                // Número de puntos para la FFT
        const int num_nfft_bins = nfft / 2 + 1;         // Número de bins no redundantes

        // Calcular el número de frames disponibles en el audio
        int num_frames = (audio_data_size - FRAME_LENGTH) / FRAME_STEP + 1;
        // Procesamos hasta MAX_LEN frames; si son menos, se rellenan
        int frames_to_process = (num_frames > MAX_LEN) ? MAX_LEN : num_frames;

        // Funciones de conversión entre Hz y Mel
        auto hz_to_mel = [](float hz) -> float {
            return 2595.0f * log10f(1.0f + hz / 700.0f);
        };
        auto mel_to_hz = [](float mel) -> float {
            return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
        };


        // 3. Inicializar la ventana de Hamming
        for (int n = 0; n < FRAME_LENGTH; n++) {
            mfcc_hamming_window[n] = 0.54f - 0.46f * cosf(2.0f * M_PI * n / (FRAME_LENGTH - 1));
        }

        // 4. Calcular los puntos en la escala Mel (uniformemente espaciados)
        float low_mel = hz_to_mel(low_freq);
        float high_mel = hz_to_mel(high_freq);
        std::vector<float> mel_points(num_mel_points);
        for (int i = 0; i < num_mel_points; i++) {
            mel_points[i] = low_mel + (high_mel - low_mel) * i / (num_mel_points - 1);
        }
        // Convertir los puntos Mel a frecuencia (Hz)
        std::vector<float> hz_points(num_mel_points);
        for (int i = 0; i < num_mel_points; i++) {
            hz_points[i] = mel_to_hz(mel_points[i]);
        }

        // 5. Mapear las frecuencias a índices de bins de la FFT
        std::vector<int> bin_indices(num_mel_points);
        for (int i = 0; i < num_mel_points; i++) {
            bin_indices[i] = static_cast<int>(floor((nfft + 1) * hz_points[i] / sample_rate));
        }

        int out_index = 0;  // Índice para copiar los MFCC al buffer de salida

        // 6. Iterar por cada frame
        for (int m = 0; m < frames_to_process; m++) {
            // Copiar las FRAME_LENGTH muestras correspondientes del audio normalizado
            for (int n = 0; n < FRAME_LENGTH; n++) {
                mfcc_frame_buffer[n] = audio_data[m * FRAME_STEP + n];
            }
            // Aplicar la ventana de Hamming a cada frame
            for (int n = 0; n < FRAME_LENGTH; n++) {
                mfcc_frame_buffer[n] *= mfcc_hamming_window[n];
            }

            /* (Opcional) Imprimir las primeras muestras del frame antes de la FFT
            for (int n = 0; n < 10; n++) {
                ESP_LOGI("mfcc_frame_buffer", "muestra antes %d: %f", n, mfcc_frame_buffer[n]);
            }*/
                // IMPLEMENTACIÓN MANUAL DE LA DFT
            // Se asume FRAME_LENGTH = N, y no se aplica normalización adicional.
            float fft_buffer[FRAME_LENGTH];
            for (int n = 0; n < FRAME_LENGTH; n++) {
                fft_buffer[n] = mfcc_frame_buffer[n];
            }
            // Calcular la FFT real del frame (utilizando la función dsps_fft2r_fc32)
            dsps_fft2r_fc32(fft_buffer, FRAME_LENGTH >> 1);

            /* (Opcional) Imprimir las primeras muestras del frame después de la FFT
            for (int n = 0; n < 10; n++) {
                ESP_LOGI("mfcc_frame_buffer", "muestra después %d: %f", n, fft_buffer[n]);
            }*/

            // Calcular la potencia del espectro:
            mfcc_power_spectrum[0] = (fft_buffer[0] * fft_buffer[0]) / static_cast<float>(FRAME_LENGTH);
            for (int i = 1; i < num_nfft_bins - 1; i++) {
                int idx = i * 2;
                float real = fft_buffer[idx];
                float imag = fft_buffer[idx + 1];
                mfcc_power_spectrum[i] = (real * real + imag * imag) / static_cast<float>(FRAME_LENGTH);
            }

            // 7. Calcular las energías de los filtros Mel para este frame
            float mel_energies[NUM_MEL_FILTERS] = {0.0f};
            // Para cada filtro (índices 1 a NUM_MEL_FILTERS en el vector de puntos)
            for (int i = 1; i <= NUM_MEL_FILTERS; i++) {
                // Lado ascendente: desde bin_indices[i-1] hasta bin_indices[i]
                for (int k = bin_indices[i - 1]; k < bin_indices[i]; k++) {
                    float weight = static_cast<float>(k - bin_indices[i - 1]) / (bin_indices[i] - bin_indices[i - 1] + epsilon);
                    mel_energies[i - 1] += mfcc_power_spectrum[k] * weight;
                }
                // Lado descendente: desde bin_indices[i] hasta bin_indices[i+1]
                for (int k = bin_indices[i]; k < bin_indices[i + 1]; k++) {
                    float weight = static_cast<float>(bin_indices[i + 1] - k) / (bin_indices[i + 1] - bin_indices[i] + epsilon);
                    mel_energies[i - 1] += mfcc_power_spectrum[k] * weight;
                }
            }

            // 8. Calcular el logaritmo de la energía de cada filtro Mel (evitando log(0))
            float log_mel[NUM_MEL_FILTERS];
            for (int i = 0; i < NUM_MEL_FILTERS; i++) {
                float energy = (mel_energies[i] < 1e-10f) ? 1e-10f : mel_energies[i];
                log_mel[i] = logf(energy);
            }

            // 9. Aplicar la DCT tipo II ortonormal para obtener los coeficientes MFCC
            // Copiar los logaritmos de las energías al buffer de la DCT
            for (int i = 0; i < NUM_MEL_FILTERS; i++) {
                mfcc_dct_buffer[i] = log_mel[i];
            }
            // Calcular la DCT
            float result[NUM_MEL_FILTERS];
            for (int k = 0; k < NUM_MEL_FILTERS; k++) {
                float sum = 0.0f;
                for (int n = 0; n < NUM_MEL_FILTERS; n++) {
                    sum += mfcc_dct_buffer[n] * cosf(M_PI * (n + 0.5f) * k / NUM_MEL_FILTERS);
                }
                if (k == 0)
                    result[k] = sum * sqrtf(1.0f / NUM_MEL_FILTERS);
                else
                    result[k] = sum * sqrtf(2.0f / NUM_MEL_FILTERS);
            }
            // Extraer los primeros NUM_MFCC_COEFFS coeficientes MFCC al buffer de salida
            for (int i = 0; i < NUM_MFCC_COEFFS; i++) {
                output_features[out_index + i] = result[i];
            }
            out_index += NUM_MFCC_COEFFS;
        }
    }
}
