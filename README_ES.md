# Nodos VibeVoice para ComfyUI

Una integración completa para ComfyUI del modelo de texto a voz **VibeVoice** de Microsoft. Este conjunto de nodos personalizados te permite generar voces de alta calidad y sonido natural para uno o múltiples hablantes, realizar clonación de voz y entrenar modelos LoRA de voz personalizados—todo directamente dentro de tus flujos de trabajo de ComfyUI.

---

## ✨ Características

### Funcionalidad Principal
- 🎤 **TTS de Un Solo Hablante:** Genera habla natural con clonación de voz opcional usando audio de referencia.
- 👥 **Conversaciones Multi-Hablante:** Soporte para generar guiones complejos con hasta 4 hablantes distintos y sus correspondientes voces clonadas.
- 🎯 **Clonación de Voz:** Imita con precisión las características de un archivo de audio de entrada.
- 🏋️ **Entrenamiento LoRA (NUEVO):** Pipelines completos de preparación de datasets y entrenamiento QLoRA integrados en la interfaz de ComfyUI.
- 💾 **Checkpoints Inteligentes (Smart Saver):** Evalúa los modelos guardados usando una métrica de "Media Real" de doble pérdida (texto + audio) e implementa Auto-Resume para prevenir pérdida de datos.
- 🛡️ **Protección OOM:** Monitoreo automático de memoria durante el entrenamiento que recorta inteligentemente el tamaño del lote (batch size) y reintenta la operación sin que la interfaz crashee.
- 🎚️ **Control de Velocidad:** Ajusta dinámicamente la velocidad del habla generada sin alterar el tono.
- ⏸️ **Pausas Personalizadas:** Inserta silencios deliberados en tu generación mediante la sintaxis `[pause]` o `[pause:ms]`.
- 🔄 **Encadenamiento y Carga de Textos:** Carga guiones largos nativamente vía archivos de texto, aprovechando la fragmentación automática para generación de longitud ilimitada.

### Rendimiento y Estabilidad
- ⚡ **Atención Optimizada:** Soporte nativo para implementaciones de atención eficientes en memoria como `sdpa` y `sage`.
- 🔒 **Precisión Estable:** Mecanismos de seguridad embebidos (Escudos Anti-NaN) en `bfloat16` para evitar colapsos matemáticos durante el muestreo multinomial.
- 🧹 **Gestión de Memoria:** Nodos de limpieza y palancas de descarga automática para vaciar las cachés de CUDA entre tareas intensivas.
- 🍎 **Apple Silicon:** Aceleración nativa por GPU en macOS (M1/M2/M3) mediante el backend MPS.

---

## 📦 Instalación

### Instalación Automática (Recomendada)
1. Ve a tu directorio `custom_nodes` de ComfyUI:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
```
2. Reinicia ComfyUI. El wrapper instalará automáticamente los requerimientos de Python necesarios en su primer uso.

---

## 📥 Instalación de Modelos

A partir de la v1.6.0, los modelos y el tokenizador deben descargarse manualmente y colocarse en los directorios correctos para evitar fallos de tiempo de espera y descargas corruptas.

*(Nota: Los modelos altamente experimentales de 0.5B y las variantes de la comunidad pre-cuantizadas han sido descontinuados para garantizar la estabilidad de los nodos).*

### Enlaces de Descarga
| Modelo                 | Tamaño  | Enlace de Descarga |
|------------------------|---------|--------------------|
| **VibeVoice-1.5B**     | ~5.4GB  | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| **VibeVoice-Large**    | ~18.7GB | [aoi-ot/VibeVoice-Large](https://huggingface.co/aoi-ot/VibeVoice-Large) |
| **VibeVoice-Large-Q8** | ~11.6GB | [FabioSarracino/VibeVoice-Large-Q8](https://huggingface.co/FabioSarracino/VibeVoice-Large-Q8) |
| **VibeVoice-Large-Q4** | ~6.6GB  | [DevParker/VibeVoice7b-low-vram](https://huggingface.co/DevParker/VibeVoice7b-low-vram) |

### Requisito del Tokenizador
VibeVoice requiere explícitamente el tokenizador Qwen2.5-1.5B.
- **Descargar:** [Qwen2.5-1.5B Tokenizer](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main)
- **Archivos requeridos:** `tokenizer_config.json`, `vocab.json`, `merges.txt`, `tokenizer.json`

### Estructura de Carpetas
Coloca los archivos descargados exactamente de la siguiente manera:
```
ComfyUI/models/vibevoice/
├── tokenizer/                 # Archivos del tokenizador Qwen aquí
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── tokenizer.json
├── VibeVoice-1.5B/           # Modelo Base 1.5B
│   ├── config.json
│   ├── model-00001-of-00003.safetensors
│   └── ...
└── VibeVoice-Large/          # Modelo Base Large
    └── ...
```

---

## 🔧 Documentación de Nodos y Detalles Técnicos

### 1. VibeVoice Single Speaker
El nodo central de generación TTS.
- **Entradas:** Cadena de texto, audio de referencia (opcional), selección de modelo base, ruta de LoRA.
- **Parámetros:**
  - `attention_type`: Recomendado dejar en `auto` o `sdpa`.
  - `diffusion_steps`: Cantidad de pasos de eliminación de ruido. Un valor más alto significa mejor calidad, pero más lentitud (por defecto: 20).
  - `temperature` y `top_p`: Controlan la varianza de la generación (qué tan plana o expresiva es la voz).
- **Salida:** Formato de Audio crudo compatible con nodos nativos de `SaveAudio` de ComfyUI.

### 2. VibeVoice Multiple Speakers
TTS avanzado para diálogo.
- **Uso:** Introduce el texto usando el formato `[1]: ¡Hola! [2]: ¡Buenas!`.
- **Entradas:** Hasta 4 pistas de audio de referencia opcionales, asignadas a `Speaker 1` hasta `Speaker 4`.
- **Recomendación:** Usa `VibeVoice-Large` para una diferenciación semántica robusta entre personajes.

### 3. 🎙️ VibeVoice Dataset Preparator
Procesa automáticamente audio en bruto en un conjunto de datos válido para entrenamiento.
- **Entradas:** Un directorio que contenga archivos de audio en bruto (`.wav`, `.mp3`, etc.).
- **Funcionalidad:** Utiliza el modelo Whisper para transcribir el audio, normaliza las pistas a 24kHz Mono y aplica "Corte Inteligente" (Smart Slicing) para recortar pistas en segmentos óptimos de 20 segundos preservando las pausas naturales de respiración.
- **Smart Caching (Caché Inteligente):** Si el nodo detecta que ya existe un dataset compilado (`prompts.jsonl`) en la carpeta de salida, omitirá instantáneamente el pesado proceso de transcripción con Whisper y pasará el directorio de salida directamente al siguiente nodo para ahorrar tiempo.
- **Salida:** Una ruta absoluta al directorio que contiene el archivo `prompts.jsonl` listo para entrenar.

### 4. 🚀 VibeVoice LoRA Trainer
Un pipeline de entrenamiento QLoRA avanzado y aislado.
- **Arquitectura Avanzada:**
  - **Protector OOM:** Si la memoria falla, corta automáticamente el `batch_size` a la mitad, escala los pasos de acumulación de gradiente y reanuda desde el último checkpoint (Auto-Resume).
  - **Smart Saver:** Monitorea ambas pérdidas (Acústica + Texto) para calcular una "Media Real". Descarta dinámicamente los checkpoints viejos con bajo rendimiento mientras **preserva estrictamente intacto** el estado del último checkpoint generado y el Top N de los mejores modelos.
  - **Auto-Resume:** Detecta automáticamente los checkpoints existentes en el directorio de salida. Pasa la ruta exacta en formato string del último checkpoint válido directamente al Hugging Face Trainer, asegurando que no se pierda nada de progreso si la interfaz crashea o el Protector OOM se activa.
- **⭐ Recomendaciones Oficiales de Entrenamiento:**
  - **Learning Rate (Tasa de Aprendizaje):** Se recomienda configurarlo estrictamente en `2e-6` para evitar el olvido catastrófico (catastrophic forgetting) y la generación de ruidos NaN.
  - **Early Stopping Patience (Paciencia de Parada Temprana):**
    - Para **Datasets Pequeños** (ej. unos pocos minutos de audio): Ajusta la paciencia a `20`.
    - Para **Datasets Grandes** (ej. horas de audio): Ajusta la paciencia a `5` para evitar el sobreajuste (overfitting).
  - **Batch Size:** `8` con Acumulación de Gradiente (Gradient Accumulation) en `16` (Baja el batch a `4` si usas una GPU de 12GB VRAM o menos).

### 5. VibeVoice LoRA Node
Conecta un adaptador LoRA entrenado a los nodos de generación principales.
- **Funcionalidad:** Detecta automáticamente estructuras de carpetas de salida anidadas (`lora/`) generadas por el Trainer de VibeVoice. Aplica los pesos dinámicamente al `language_model`, `prediction_head` y a los `connectors` antes del muestreo.

### 6. Utilidades: Load Text & Free Memory
- **Load Text From File (Cargar Texto):** Lee un archivo `.txt` como entrada de cadena, ideal para audiolibros masivos.
- **Free Memory (Liberar Memoria):** Descarga instantáneamente el modelo VibeVoice y vacía la caché de CUDA. Útil para liberar VRAM en flujos de trabajo mixtos (que combinen audio y video/imágenes).

---

## 📄 Licencia
Este wrapper de ComfyUI está publicado bajo la Licencia MIT.
*Nota: La arquitectura base de Microsoft VibeVoice está sujeta a su Licencia de Microsoft original (No Comercial / Solo Investigación).*