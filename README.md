# VibeVoice ComfyUI Nodes

Una integración completa para ComfyUI del modelo de texto a voz VibeVoice de Microsoft, que permite una síntesis de voz de alta calidad para uno o múltiples hablantes directamente dentro de tus flujos de trabajo en ComfyUI.

## ✨ Características

### Funcionalidad Principal
- 🎤 **TTS de un solo hablante**: Genera habla natural con clonación de voz opcional.
- 👥 **Conversaciones multi-hablante**: Soporte para hasta 4 hablantes distintos.
- 🎯 **Clonación de Voz**: Clona voces a partir de muestras de audio.
- 🏋️ **Entrenamiento LoRA (NUEVO)**: Entrena tus propios modelos de voz directamente en ComfyUI.
- 💾 **Smart Checkpointing (NUEVO)**: Sistema inteligente que guarda solo los mejores modelos basados en métricas de pérdida dual (difusión y texto).
- 🛡️ **Protección OOM (NUEVO)**: Sistema de reintento automático que ajusta el tamaño del lote si se detecta falta de memoria VRAM.
- 🎨 **Soporte LoRA**: Afina voces con adaptadores LoRA personalizados (v1.4.0+).
- 🎚️ **Control de Velocidad de Voz**: Ajusta la velocidad del habla modificando la velocidad de la voz de referencia (v1.5.0+).
- 📝 **Carga de Archivos de Texto**: Carga guiones desde archivos de texto.
- 📚 **División Automática de Texto**: Maneja textos largos sin problemas con tamaño de fragmento configurable.
- ⏸️ **Etiquetas de Pausa Personalizadas**: Inserta silencios con las etiquetas `[pause]` y `[pause:ms]`.
- 🔄 **Encadenamiento de Nodos**: Conecta múltiples nodos VibeVoice para flujos de trabajo complejos.
- ⏹️ **Soporte de Interrupción**: Cancela operaciones antes o entre generaciones.
- 🔧 **Configuración Flexible**: Controla temperatura, muestreo y escala de guía.

### Rendimiento y Optimización
- ⚡ **Mecanismos de Atención**: Elige entre auto, eager, sdpa, flash_attention_2 o sage.
- 🎛️ **Pasos de Difusión**: Equilibrio ajustable entre calidad y velocidad (por defecto: 20).
- 💾 **Gestión de Memoria**: Alterna la limpieza automática de VRAM después de la generación.
- 🧹 **Nodo de Liberación de Memoria**: Control manual de memoria para flujos de trabajo complejos.
- 🍎 **Soporte Apple Silicon**: Aceleración nativa por GPU en Macs M1/M2/M3 vía MPS.
- 🔢 **Cuantización de 8-Bits**: Calidad de audio perfecta con alta reducción de VRAM.
- 🔢 **Cuantización de 4-Bits**: Máximo ahorro de VRAM con mínima pérdida de calidad.

### Compatibilidad e Instalación
- 📦 **Autocontenido**: Código VibeVoice embebido, sin dependencias externas complejas.
- 🔄 **Compatibilidad Universal**: Soporte adaptativo para transformers v4.51.3+.
- 🖥️ **Multiplataforma**: Funciona en Windows, Linux y macOS.
- 🎮 **Multi-Backend**: Soporta CUDA, CPU y MPS (Apple Silicon).

## 🎥 Video Demo
<p align="center">
  <a href="https://www.youtube.com/watch?v=fIBMepIBKhI">
    <img src="https://img.youtube.com/vi/fIBMepIBKhI/maxresdefault.jpg" alt="VibeVoice ComfyUI Wrapper Demo" />
  </a>
  <br>
  <strong>Haz clic para ver el video de demostración</strong>
</p>

## 📦 Instalación

### Instalación Automática (Recomendada)
1. Clona este repositorio en tu carpeta `custom_nodes` de ComfyUI:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
```

2. Reinicia ComfyUI - los nodos instalarán automáticamente los requisitos en el primer uso.

## 📥 Instalación de Modelos

### Descarga Manual Requerida
Desde la versión 1.6.0, los modelos y el tokenizador deben descargarse manualmente y colocarse en la carpeta correcta. El wrapper ya no los descarga automáticamente.

### Enlaces de Descarga

#### Modelos
Puedes descargar los modelos VibeVoice desde HuggingFace:

| Modelo                 | Tamaño | Enlace de Descarga |
|------------------------|--------|--------------------|
| **VibeVoice-1.5B**     | ~5.4GB | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| **VibeVoice-Large**    | ~18.7GB | [aoi-ot/VibeVoice-Large](https://huggingface.co/aoi-ot/VibeVoice-Large) |
| **VibeVoice-Large-Q8** | ~11.6GB | [FabioSarracino/VibeVoice-Large-Q8](https://huggingface.co/FabioSarracino/VibeVoice-Large-Q8) |
| **VibeVoice-Large-Q4** | ~6.6GB | [DevParker/VibeVoice7b-low-vram](https://huggingface.co/DevParker/VibeVoice7b-low-vram) |

#### Tokenizador (Requerido)
VibeVoice utiliza el tokenizador Qwen2.5-1.5B:
- Descargar de: [Qwen2.5-1.5B Tokenizer](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main)
- Archivos requeridos: `tokenizer_config.json`, `vocab.json`, `merges.txt`, `tokenizer.json`

### Pasos de Instalación
1. Crea la carpeta de modelos si no existe:
   ```
   ComfyUI/models/vibevoice/
   ```

2. Descarga y organiza los archivos en la carpeta vibevoice:
   ```
   ComfyUI/models/vibevoice/
   ├── tokenizer/                 # Coloca los archivos del tokenizador Qwen aquí
   │   ├── tokenizer_config.json
   │   ├── vocab.json
   │   ├── merges.txt
   │   └── tokenizer.json
   ├── VibeVoice-1.5B/           # Carpeta del modelo
   │   ├── config.json
   │   ├── model-00001-of-00003.safetensors
   │   ├── model-00002-of-00003.safetensors
   │   └── ... (otros archivos del modelo)
   ├── VibeVoice-Large/
   │   └── ... (archivos del modelo)
   └── my-custom-vibevoice/      # se soportan nombres personalizados
       └── ... (archivos del modelo)
   ```

3. Para modelos descargados de HuggingFace usando git-lfs o HF CLI, puedes usar la estructura de caché:
   ```
   ComfyUI/models/vibevoice/
   └── models--microsoft--VibeVoice-1.5B/
       └── snapshots/
           └── [hash]/
               └── ... (archivos del modelo)
   ```

4. Refresca tu navegador - los modelos aparecerán en el menú desplegable.

## 🏋️ Entrenamiento de LoRA (NUEVO)

VibeVoice ComfyUI ahora incluye un potente sistema de entrenamiento LoRA integrado.

### 1. Preparación del Dataset (VibeVoice Dataset Preparator)
Este nodo procesa tus archivos de audio crudos y crea un dataset listo para entrenar.
- **Entrada**: Directorio con archivos de audio (.wav, .mp3, .flac, .ogg, .m4a, .mp4).
- **Procesamiento**:
  - Utiliza Whisper para transcribir el audio automáticamente.
  - Normaliza el audio a 24kHz mono.
  - Realiza "Smart Slicing" para cortar el audio en fragmentos óptimos (hasta 20s) preservando silencios internos.
- **Salida**: Ruta al dataset procesado.

### 2. Entrenador LoRA (VibeVoice LoRA Trainer)
Entrena un adaptador LoRA personalizado usando el dataset preparado.
- **Características Avanzadas**:
  - **Smart Checkpointing**: Guarda solo los N mejores modelos basados en la calidad real (suma de pérdidas), no en la antigüedad.
  - **Dual-Loss Early Stopping**: Monitorea independientemente la pérdida de difusión y la pérdida de texto (CE). Si cualquiera mejora, el entrenamiento continúa. Esto previene la degradación acústica.
  - **Protección OOM (Out of Memory)**: Si tu GPU se queda sin memoria, el entrenamiento se pausa, reduce el `batch_size`, aumenta los `gradient_accum_steps` para compensar, y se reinicia automáticamente.
  - **Restauración del Mejor Modelo**: Al finalizar, el sistema garantiza que el modelo guardado en la carpeta de salida es matemáticamente el mejor (menor pérdida) de toda la sesión, no solo el último.
- **Parámetros Clave**:
  - `save_total_limit`: Número máximo de mejores checkpoints a conservar.
  - `early_stopping_patience`: Pasos sin mejora antes de detenerse.
  - `early_stopping_threshold`: Mejora mínima requerida para reiniciar el contador de paciencia.

### Uso del LoRA Entrenado
El nodo `VibeVoiceLoRANode` ahora soporta estructuras anidadas automáticamente. Simplemente selecciona tu LoRA entrenado en el menú desplegable; el nodo detectará si los archivos están en la raíz o en una subcarpeta `lora/` (estructura de salida del entrenamiento).

## 🔧 Nodos Disponibles

### 1. VibeVoice Load Text From File
Carga contenido de texto desde archivos en los directorios input/output/temp de ComfyUI.
- **Formatos soportados**: .txt
- **Salida**: Cadena de texto para nodos TTS.

### 2. VibeVoice Single Speaker
Genera voz a partir de texto usando una sola voz.
- **Entrada de Texto**: Texto directo o conexión desde nodo Load Text.
- **Modelos**: Selecciona del menú desplegable.
- **Clonación de Voz**: Entrada de audio opcional.
- **Parámetros**:
  - `text`: Texto a convertir.
  - `model`: Modelo VibeVoice a usar.
  - `attention_type`: Tipo de atención (auto recomendado).
  - `quantize_llm`: Cuantización dinámica del LLM ("full precision", "4bit", "8bit").
  - `free_memory_after_generate`: Liberar VRAM tras generar.
  - `diffusion_steps`: Pasos de desruido (calidad vs velocidad).
  - `seed`: Semilla para reproducibilidad.
  - `voice_speed_factor`: Ajuste de velocidad del habla.

### 3. VibeVoice Multiple Speakers
Genera conversaciones multi-hablante con voces distintas.
- **Formato**: Usa la notación `[N]:` donde N es 1-4.
- **Asignación de Voces**: Muestras de voz opcionales para cada hablante.
- **Recomendación**: Usar VibeVoice-Large para mejor calidad.

### 4. VibeVoice Free Memory
Libera manualmente todos los modelos VibeVoice cargados.
- **Uso**: Inserta entre nodos para limpiar VRAM en puntos específicos.

### 5. VibeVoice LoRA
Configura y carga adaptadores LoRA.
- **Detección Inteligente**: Soporta carpetas de LoRA estándar y anidadas (salida de entrenamiento).
- **Parámetros**: Fuerza del LLM y activación de componentes (difusión, conectores).

## 🧠 Información de Modelos

### VibeVoice-1.5B
- **VRAM**: ~6GB
- **Uso**: Prototipado rápido, voz única.

### VibeVoice-Large
- **VRAM**: ~20GB
- **Uso**: Máxima calidad de producción, multi-hablante.

### VibeVoice-Large-Q8
- **VRAM**: ~12GB
- **Calidad**: Idéntica a precisión completa (cuantización selectiva).
- **Uso**: Producción en GPUs de 12GB (RTX 3060, 4070 Ti).

### VibeVoice-Large-Q4
- **VRAM**: ~8GB
- **Uso**: Máximo ahorro de memoria.

## 📄 Licencia

Este wrapper de ComfyUI se publica bajo la Licencia MIT.
**Nota**: El modelo VibeVoice en sí está sujeto a los términos de licencia de Microsoft (solo investigación).

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor prueba los cambios exhaustivamente y sigue el estilo de código existente.

## 📝 Historial de Cambios (Reciente)

### Versión 1.9.0 (Actual)
- **Sistema de Entrenamiento Completo**:
  - Nuevo nodo `VibeVoice_Dataset_Preparator` para creación automática de datasets con Whisper.
  - Nuevo nodo `VibeVoice_LoRA_Trainer` para entrenamiento robusto.
  - **Smart Checkpointing**: Guarda solo los mejores modelos basado en métricas reales.
  - **Dual-Loss Early Stopping**: Previene degradación monitoreando pérdidas acústicas y textuales.
  - **OOM Auto-Retry**: Recuperación automática ante errores de memoria VRAM.
  - **Restauración del Mejor Modelo**: Garantiza que el resultado final es el mejor checkpoint.
- **Soporte LoRA Mejorado**:
  - Detección automática de estructuras de carpetas anidadas (`lora/`).
- **Correcciones de Estabilidad**:
  - Parches seguros para inyección de código en tiempo de ejecución.
  - Manejo robusto de sintaxis Python en scripts parcheados.

### Versión 1.8.1
- Instalación forzada de bitsandbytes>=0.48.1 para corregir bugs críticos en modelos Q8.

### Versión 1.8.0
- Soporte oficial para modelo VibeVoice-Large-Q8 (calidad perfecta, 12GB VRAM).
- Cuantización dinámica de 8-bits para componentes LLM.
