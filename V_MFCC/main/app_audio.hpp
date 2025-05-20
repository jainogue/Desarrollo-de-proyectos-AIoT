#pragma once

#include <functional> //maneja funciones callback
#include <mutex> //Proteger datos compartidos entre tareas
#include <cstring> //Operaciones con cadenas
#include <math.h> //Funciones matemáticas
#include <cmath> //Funciones matemáticas
#include <complex> //Funciones complejas
#include <freertos/FreeRTOS.h> //Manejo de tareas y semáforos
#include <freertos/task.h> //Manejo de tareas y semáforos
#include "sdkconfig.h" //Acceso a configuraciones definidas en sdkconfig
#include "model.hpp"  //Modelo  de IA
#include <cstdint> //Definiciones de enteros sin signo


// TensorFlow Lite headers
#include "tensorflow/lite/builtin_ops.h"  //Operaciones de Tensorflow
#include "tensorflow/lite/micro/micro_log.h"  //Manejo de logs
#include "tensorflow/lite/micro/micro_interpreter.h" //Motor de inferencia
#include "tensorflow/lite/schema/schema_generated.h" //Define  la estrucutra del proceso
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" //resuelve las operaciones  que el modelo necesita
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h" //Manejo de errores

// ESP32-specific headers
#include "bsp/esp-bsp.h"  //Biblioteca para manejar el hardware
#include "esp_heap_caps.h"  //Gestión de memoria dinámica
#include "esp_codec_dev_defaults.h" //Configuraciones por defecto para el codec
#include "esp_log.h" //Manejo de logs
#include "esp_dsp.h" //Funciones DSP procesamiento de audio
#include "esp_err.h" //Manejo de errores

#define TENSOR_ARENA_SIZE (2560 * 1024) //Reserva memoria para ejecucion del modelo de  IA en TFLite
static SemaphoreHandle_t xFeedSemaphore;
static SemaphoreHandle_t xDetectSemaphore;

using Complex = std::complex<float>;
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
        int *m_audio_buffer_int16;
        float *m_features; //  Almacena  características extraidas  del audio
        int m_audio_buffer_ptr{0}; //Indice  del buffer de audio
        int m_audio_buffer_read_ptr{0}; //Indice  del buffer de audio para lectura
        std::mutex m_features_mutex;  //Protege el acceso a m_features en   tareas concurrentes


        // Tamaños de los buffers para tareas
        static constexpr int DETECT_TASK_BUFF_SIZE{100 * 1024}; //Buffer de 4KB para  detección
        inline static uint8_t *m_detect_task_buffer;
        inline static StaticTask_t m_detect_task_data;

        static constexpr int ACTION_TASK_BUFF_SIZE{8 * 1024}; //Buffer de 4KB para  acción
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


        static void GenerateMicroFeatures(float *audio_data, int audio_data_size, float *output_features); //Extrae características de audio
        static void my_fft_real(float *data, int N, std::vector<Complex>& output); //FFT de 512 puntos (256 reales)
        static void my_fft(Complex* data, int N); //FFT de 512 puntos (256 complejos)
        // Parámetros de extracción
        static const int FRAME_LENGTH = 640;        // Tamaño del frame en muestras
        static const int FRAME_STEP   = 320;          // Desplazamiento entre frames (con sobreposición)
        static const int NUM_MEL_FILTERS = 26;        // Número de filtros en el banco Mel
        static const int NUM_MFCC_COEFFS = 13;        // Número de coeficientes MFCC por frame
        static const int MAX_LEN = 49;              // Longitud máxima de la secuencia de coeficientes MFCC
        static constexpr int MFCC_FEATURE_SIZE{MAX_LEN * NUM_MFCC_COEFFS};

        // Funciones auxiliares para conversión entre Hz y escala Mel
        static float *mfcc_frame_buffer; // Buffer para el frame de audio
        static float *mfcc_power_spectrum; // Buffer para el espectro de potencia
        static float *mfcc_dct_buffer; // Buffer para el DCT
        static float *mfcc_hamming_window; // Buffer para la ventana de Hamming



    public:
        void init(std::function<void(int, int, float, float, int)> state_callback); // Inicialización
        void start(void); // Inicio del sistema de tareas
    };
}
