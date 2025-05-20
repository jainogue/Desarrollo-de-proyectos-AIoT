#pragma once

#include <functional> //maneja funciones callback
#include <mutex> //Proyteger datos compartidos entre tareas
#include <cstring> //Operaciones con cadenas
#include <math.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "sdkconfig.h" //Acceso a configuraciones definidas en sdkconfig
#include "model.hpp"  //Modelo  de IA


// TensorFlow Lite headers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h" //Motor de inferencia
#include "tensorflow/lite/schema/schema_generated.h" //Define  la estrucutra del proceso
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" //resuelve lass operaciones  que el modelo necesita
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

// ESP32-specific headers
#include "bsp/esp-bsp.h"
#include "esp_heap_caps.h"
#include "esp_codec_dev_defaults.h"
#include "esp_log.h"

#define TENSOR_ARENA_SIZE (8110 * 1024) //Reserva memoria para ejecucion del modelo de  IA en TFLite
static SemaphoreHandle_t xFeedSemaphore;
static SemaphoreHandle_t xDetectSemaphore;


namespace app
{
    class AppAudio
    {
    private:
        esp_codec_dev_handle_t m_mic_handle;

        // Constantes para clasificación
        static constexpr int CRYING_IDX{0};   // Índice para llanto
        static constexpr int NOISE_IDX{1};   // Índice para ruido general


        static constexpr int AUDIO_BUFFER_SIZE{16000}; // buffer de  16000 muestras (1s a 16kHz)
        // Buffers de audio y características
        float *m_audio_buffer;
        float *m_features; //  Almacena  características extraidas  del audio
        int m_audio_buffer_ptr{0}; //Indice  del buffer de audio
        std::mutex m_features_mutex;  //Protege el acceso a m_features en   tareas concurrentes

        // Tamaños de los buffers para tareas
        static constexpr int DETECT_TASK_BUFF_SIZE{4 * 1024}; //Buffer de 4KB para  detección
        inline static uint8_t *m_detect_task_buffer;
        inline static StaticTask_t m_detect_task_data;

        static constexpr int ACTION_TASK_BUFF_SIZE{4 * 1024}; //Buffer de 4KB para  acción
        inline static uint8_t *m_action_task_buffer;
        inline static StaticTask_t m_action_task_data;

        static constexpr int FEED_TASK_BUFF_SIZE{8 * 1024}; //Buffer de 4KB para  captura   de audio
        inline static uint8_t *m_feed_task_buffer;
        inline static StaticTask_t m_feed_task_data;

        // Callback para notificar cambios de estado
        std::function<void(int, int, float, float, int)> m_state_fn;
        int m_last_state{-1};  //guarda el último estado detectado

        // TensorFlow Lite Micro Interpreter
        uint8_t *m_tensor_arena; //espacio reservado para  inferencia
        tflite::MicroInterpreter *m_interpreter{nullptr};  //Intérprete de Tensorflow  para correr el modelo
        TfLiteTensor *m_input{nullptr};  //Tensor de entrada del modelo

        // Declaración de tareas
        static void feedTask(void *arg);   // Captura de audio
        static void detectTask(void *arg); // Procesamiento del modelo
        static void actionTask(void *arg); // Manejo de resultados


    public:
        void init(std::function<void(int, int, float, float, int)> state_callback); // Inicialización
        void start(void); // Inicio del sistema de tareas
    };
}
