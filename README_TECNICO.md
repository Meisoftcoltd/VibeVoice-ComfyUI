# VibeVoice ComfyUI - Documentación Técnica del Repositorio

Este documento proporciona una descripción técnica detallada de cada archivo y directorio dentro del repositorio **VibeVoice-ComfyUI**. El objetivo es facilitar la comprensión de la arquitectura del proyecto, la función de cada módulo y cómo interactúan entre sí dentro del ecosistema de ComfyUI.

---

## Directorio Raíz

### `__init__.py`
El punto de entrada principal para ComfyUI.
- **Función:** Detecta e inicializa los nodos personalizados al arrancar ComfyUI.
- **Detalles técnicos:** Configura el sistema de registro (logging), inyecta parches de compatibilidad (por ejemplo, para conflictos con la librería `timm`), verifica que la versión empaquetada del modelo (`vvembed`) esté presente, y expone los diccionarios `NODE_CLASS_MAPPINGS` y `NODE_DISPLAY_NAME_MAPPINGS` que ComfyUI requiere para cargar y registrar los nodos en la interfaz visual.

### `README.md`
- **Función:** El archivo estándar de documentación para los usuarios. Contiene instrucciones de instalación, ejemplos de uso y descripciones generales del proyecto.

### `requirements.txt` / `requirements_training.txt`
- **Función:** Listas de dependencias de Python necesarias para ejecutar la inferencia y el entrenamiento de VibeVoice, respectivamente. (e.g., `transformers`, `torch`, `peft`, `bitsandbytes`).

### `pyproject.toml`
- **Función:** Archivo de configuración para empaquetado y herramientas modernas de Python (como formateadores o linters), definiendo metadatos del proyecto.

### `node_list.json`
- **Función:** Metadatos en formato JSON requeridos por ComfyUI Manager para identificar, catalogar y gestionar la instalación del conjunto de nodos.

### `test_*.py` (`test_escape.py`, `test_syntax_validity.py`, `test_type_hint_patch.py`)
- **Función:** Scripts de prueba unitaria y scripts para verificar sintaxis, aserciones y parches lógicos sin necesidad de iniciar la interfaz de ComfyUI completa.

---

## Directorio `/nodes`
Este directorio contiene la lógica de integración específica de ComfyUI para interactuar con la arquitectura de VibeVoice.

### `base_vibevoice.py`
- **Función:** Clase base abstracta (`BaseVibeVoiceNode`) compartida por todos los nodos de inferencia.
- **Detalles técnicos:**
  - Maneja la **descarga JIT (Just-In-Time)** y carga de modelos desde HuggingFace y su caché.
  - Detecta y aplica **cuantización híbrida** y atenciones optimizadas (como `SageAttention` o `sdpa`).
  - Gestiona la **inyección de adaptadores LoRA** (`_apply_lora`) en componentes específicos del modelo (LLM, cabezales de difusión, conectores acústicos/semánticos).
  - Implementa el bucle de generación núcleo (`_generate_with_vibevoice`), incluyendo un escudo contra *NaNs/Infs* usando un `NaNSanitizerLogitsProcessor` personalizado para prevenir bloqueos y crashes de CUDA durante el muestreo.

### `single_speaker_node.py` y `multi_speaker_node.py`
- **Función:** Implementaciones concretas de nodos de ComfyUI para generación de voz.
- **Detalles técnicos:** Heredan de `base_vibevoice.py`. Definen los tipos de entrada (texto, modelo base, audio de referencia para clonación, semilla, velocidad, parámetros de muestreo) y formatos de salida para ComfyUI. El nodo multi-hablante divide el texto entrante y enruta diferentes clips de referencia dependiendo del formato de guion (e.g., `Speaker 1: ... Speaker 2: ...`).

### `free_memory_node.py`
- **Función:** Nodo de utilidad en ComfyUI.
- **Detalles técnicos:** Fuerza la liberación manual del modelo de VibeVoice y su procesador de la VRAM, ejecutando la recolección de basura (`gc.collect()`) y limpiando la caché de CUDA (`torch.cuda.empty_cache()`).

### `load_text_node.py`
- **Función:** Nodo auxiliar simple de ComfyUI para leer un archivo de texto (`.txt`) desde el disco y convertir su contenido en un parámetro tipo cadena (STRING) para usarlo en los flujos de trabajo de VibeVoice.

### `lora_node.py`
- **Función:** Nodo para seleccionar y encadenar modelos LoRA dentro de la interfaz.
- **Detalles técnicos:** Escanea el directorio `models/vibevoice/loras` y proporciona una interfaz visual (desplegable) para elegir qué adaptador LoRA debe combinarse y aplicarse en tiempo de inferencia sobre el modelo base.

### `nodes_training.py`
- **Función:** Nodos responsables del pipeline de ajuste fino (Fine-Tuning) y entrenamiento QLoRA.
- **Detalles técnicos:**
  - **`VibeVoice_Dataset_Preparator`:** Formatea y estructura el texto y el audio de entrada requeridos por VibeVoice, creando archivos `.jsonl` estrictos listos para el entrenamiento.
  - **`VibeVoice_LoRA_Trainer`:** Genera un entorno de entrenamiento aislado. Clona dinámicamente scripts de entrenamiento, inyecta lógica mediante parches (regex y sustitución de código) para gestionar modelos cuantizados, configura métricas tempranas de detención (`SmartEarlyStoppingAndSaveCallback`) y lanza la sesión en un subproceso asíncrono que reporta el progreso mediante barras de estado nativas de ComfyUI.

---

## Directorio `/vvembed`
Contiene una copia (o bifurcación modificada) del código nativo de inferencia y arquitectura del modelo de **VibeVoice** para evitar dependencias externas frágiles y garantizar la compatibilidad (Embedded VibeVoice).

### `/vvembed/modular`
- **Función:** Los bloques de construcción y la arquitectura neuronal del modelo.
- **Detalles técnicos:** Archivos como `modeling_vibevoice.py` (y sus variantes de inferencia) definen las clases de PyTorch que forman el pipeline acústico y lingüístico. Aquí se define el modelo auto-regresivo (típicamente Qwen2 en el backend) interactuando con módulos de difusión, codificadores y conectores acústicos.

### `/vvembed/processor`
- **Función:** Lógica de preprocesamiento multimodal.
- **Detalles técnicos:** Archivos como `vibevoice_processor.py` se encargan de tokenizar el texto, cargar/remuestrear formas de onda de audio de referencia, fusionar secuencias de audio y texto, y prepararlas en tensores adecuadamente empaquetados para ser consumidos por el modelo LLM subyacente.

### `/vvembed/schedule`
- **Función:** Controladores del proceso de difusión.
- **Detalles técnicos:** Gestionan cómo el ruido es iterativamente eliminado para generar o condicionar la señal de audio, controlando variables como la escala de fuerza (CFG Scale) y el número de pasos de inferencia.

### `/vvembed/scripts`
- **Función:** Herramientas y funciones auxiliares nativas proporcionadas por el repositorio original de VibeVoice (por ejemplo, scripts independientes para conversión, validación u otras tareas operativas de bajo nivel).

---

## Directorio `/examples`
- **Función:** Contiene archivos JSON con *workflows* (flujos de trabajo) listos para arrastrar y soltar en la interfaz visual de ComfyUI. Ayudan al usuario a entender cómo conectar los distintos nodos de este repositorio para lograr inferencia de voz, clonación y entrenamiento.