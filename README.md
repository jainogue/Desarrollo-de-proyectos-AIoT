# AIoT Baby Cry Monitor

Este repositorio contiene todo el código y materiales asociados al proyecto de fin de grado “Desarrollo de proyectos IoT que implementen IA/ML en dispositivos empotrados” sobre ESP32-S3-Box 3.

## Descripción general

El sistema implementa un flujo completo de desarrollo AIoT:

1. **Entrenamiento y optimización de modelos** en Python (Keras / TensorFlow)
2. **Preprocesado de audio** (amplitud vs. MFCC) y generación de métricas
3. **Conversión y cuantización** de los modelos a TensorFlow Lite
4. **Firmware embebido** en ESP32-S3 para:
   - Captura de audio
   - Inferencia en tiempo real
   - Monitorización de memoria
   - Provisión y conexión Wi-Fi / BLE
   - Publicación MQTT de resultados

**Nota:** Las bases de datos de audio no están incluidas en este repositorio.

## DIrectorios del proyecto

- **general_notebooks/**: Notebooks para preprocesado, entrenamiento y evaluación de modelos.
- **testing/**: Pruebas de inferencia y recopilación de métricas finales.
- **V_float**, **V_int8**, **V_MFCC**: Tres variantes de firmware ESP-IDF según el tipo de modelo (float32, int8 o MFCC). Cada carpeta incluye:
  - `main/`: código fuente en ESP-IDF
  - `CMakeLists.txt` y `partitions.csv`
  - `sdkconfig.defaults`

---

## Requisitos previos

- **ESP-IDF v4.4** (o superior)
- **Python 3.8+** con TensorFlow y dependencias de audio
- Broker MQTT accesible en la red local
- Módulo ESP32-S3-Box 3 (u otro ESP32-S3 compatible)


## Despliegue del firmware

1. **Seleccionar la variante**
```bash
cd V_float    # Float32
cd V_int8     # INT8
cd V_MFCC     # MFCC
```

2. **Configurar proyecto a medida**

Insertar credenciales de MQTT en archivo kconfig.projbuild.


3. **Compilar y flashear**

```bash
idf.py build
idf.py flash
idf.py monitor
```
El monitor mostrará logs de provisión Wi-Fi/BLE, inferencia, uso de RAM/PSRAM y publicaciones MQTT.

## Uso de los notebooks

### general_notebooks/

- **audio_processing.ipynb**
  Extracción de amplitud y MFCC de archivos de audio crudo.

- **model_training.ipynb**
  Entrenamiento de las arquitecturas MLP/CNN sobre datos de audio.

- **model_compression.ipynb**
  Conversión y cuantización de modelos a TFLite.

- **model_test.ipynb**
  Evaluación de precisión y consumo de recursos.

### testing/

Notebooks y scripts para generar datos de prueba, comparar resultados y recopilar métricas finales (tiempo de inferencia, RAM/Flash, precisión).
