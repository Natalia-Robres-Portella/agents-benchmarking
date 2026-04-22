# Agent Benchmark Suite

Un framework de investigación para evaluar y comparar agentes LLM de forma rigurosa, reproducible y extensible.

---

## Índice

1. [¿Para qué sirve?](#1-para-qué-sirve)
2. [¿Por qué se hace?](#2-por-qué-se-hace)
3. [¿Qué compara?](#3-qué-compara)
4. [¿Cómo funciona por dentro?](#4-cómo-funciona-por-dentro)
5. [¿De dónde saca los datos?](#5-de-dónde-saca-los-datos)
6. [¿Cómo se evalúa?](#6-cómo-se-evalúa)
7. [Estructura de carpetas](#7-estructura-de-carpetas)
8. [Instalación](#8-instalación)
9. [Cómo ejecutar un experimento](#9-cómo-ejecutar-un-experimento)
10. [Qué hay que rellenar / configurar](#10-qué-hay-que-rellenar--configurar)
11. [Referencia completa de configuración YAML](#11-referencia-completa-de-configuración-yaml)
12. [Qué se guarda en los resultados](#12-qué-se-guarda-en-los-resultados)
13. [Cómo añadir cosas nuevas](#13-cómo-añadir-cosas-nuevas)
14. [Tests](#14-tests)
15. [Papers de referencia](#15-papers-de-referencia)

---

## 1. ¿Para qué sirve?

Este proyecto permite **ejecutar experimentos controlados sobre agentes LLM** y medir objetivamente cómo se comportan en tareas de razonamiento y búsqueda de información.

Un "agente LLM" es un sistema que usa un modelo de lenguaje (GPT-4o, Claude, etc.) no sólo para generar texto, sino para **razonar, decidir qué herramientas usar, ejecutarlas, observar el resultado y repetir** hasta llegar a una respuesta. Este framework responde preguntas como:

- ¿ReAct es mejor que un agente sin herramientas en HotPotQA?
- ¿Cuántos pasos necesita el agente de media para responder correctamente?
- ¿Qué porcentaje de llamadas a herramientas tienen argumentos válidos?
- ¿Cuántos tokens consume cada estrategia por tarea?
- ¿El agente es capaz de recuperarse cuando una herramienta falla?

---

## 2. ¿Por qué se hace?

La mayoría de los benchmarks de LLMs evalúan el modelo directamente (¿qué tan bien responde a una pregunta?). Este proyecto evalúa el **sistema agente completo**: estrategia de razonamiento + herramientas + memoria + modelo.

### Problemas que resuelve

**Reproducibilidad.** Sin fijar la semilla aleatoria, el mismo experimento da resultados distintos en cada ejecución. Aquí cada run tiene un `config_hash` (SHA-256 del YAML de configuración) que se incrusta en cada trajectory, garantizando que dos runs con el mismo hash son idénticos.

**Comparación justa.** Para comparar ReAct vs. Reflexion vs. Direct, todos deben correr con el mismo conjunto de tareas (misma semilla), el mismo modelo y las mismas métricas. El framework lo garantiza por diseño.

**Trazabilidad completa.** Cada paso del agente — qué pensó, qué herramienta llamó, con qué argumentos, qué respuesta recibió, cuántos tokens gastó, cuánto tardó — se guarda en JSONL. Si algo falla, se puede reproducir exactamente lo que pasó.

**Base académica.** Cada decisión de diseño está justificada con papers:
- ReAct (Yao et al., ICLR 2023)
- Reflexion (Shinn et al., NeurIPS 2023)
- Pass@k (Chen et al., 2021 / Shinn et al., 2023)
- LLM-as-judge bias mitigation (Zheng et al., 2023)
- HELM multi-metric evaluation (Liang et al., 2022)

---

## 3. ¿Qué compara?

El framework compara agentes a lo largo de **cuatro ejes independientes** que se pueden combinar libremente:

### Eje 1 — Estrategia de planificación

| Estrategia | Estado | Descripción |
|---|---|---|
| `direct` | ✅ Implementado | Sin herramientas. El modelo responde directamente. Baseline. |
| `react` | ✅ Implementado | Thought → Action → Observation en bucle (Yao et al., 2023). |
| `reflexion` | 🔜 Pendiente | ReAct + reflexión verbal entre trials (Shinn et al., 2023). |
| `plan_execute` | 🔜 Pendiente | Genera un plan completo antes de ejecutar (Wang et al., 2023). |
| `tot` | 🔜 Pendiente | Tree of Thoughts: explora múltiples ramas de razonamiento (Yao et al., 2023). |

### Eje 2 — Tipo de memoria

| Tipo | Descripción |
|---|---|
| `no_memory` | Sin memoria. Cada trial empieza desde cero. |
| `window_buffer` | Mantiene los últimos N pasos en contexto. |
| `episodic` | Buffer de reflexiones verbales (necesario para Reflexion). Persiste entre trials de la misma tarea; se borra entre tareas. |
| `vector_store` | (Avanzado) Retrieval con embeddings (requiere ChromaDB). |

### Eje 3 — Modelo LLM

Cualquier modelo de OpenAI (GPT-4o, GPT-4o-mini…) o Anthropic (Claude Opus, Sonnet…). Se configura en el YAML.

### Eje 4 — Herramientas disponibles

| Herramienta | Descripción |
|---|---|
| `search` | Búsqueda web. En modo offline usa el fixture local; en modo live usa DuckDuckGo. |
| `calculator` | Evaluador de expresiones matemáticas seguro (allowlist de operadores). |
| `finish` | Señal de terminación. El agente la llama cuando tiene la respuesta final. |

---

## 4. ¿Cómo funciona por dentro?

El flujo completo de un experimento:

```
run_experiment.py
    └── ExperimentOrchestrator.run()
            │
            ├── 1. seed_everything(seed)          ← reproducibilidad
            ├── 2. Crea results/{run_id}/
            ├── 3. Snapshot de config.yaml
            │
            ├── 4. Construye componentes
            │       ├── Agent (estrategia + memoria + LLM + herramientas)
            │       ├── TaskLoader (carga las tareas del dataset)
            │       ├── TraceLogger (escribe JSONL por pasos)
            │       ├── ExecutionEngine (loop de pasos)
            │       ├── EvaluationModule (métricas)
            │       └── ReportGenerator (escribe resultados)
            │
            ├── 5. Para cada tarea:
            │       └── ExecutionEngine.run(task)
            │               │
            │               └── Para cada trial (n_trials veces):
            │                       ├── post_episode_hook() ← Reflexion escribe aquí
            │                       ├── agent.reset(seed)
            │                       └── Loop hasta max_steps:
            │                               ├── agent.act(state)   → Action
            │                               ├── tools.execute()    → ToolResult
            │                               ├── agent.observe(obs)
            │                               └── logger.log_step()
            │
            ├── 6. EvaluationModule.compute_all()
            │       ├── Stage 1: validator.validate() → score por trajectory
            │       └── Stage 2: metric.compute()     → MetricResult por métrica
            │
            └── 7. ReportGenerator.emit()
                    ├── metrics.json
                    └── report.md
```

### El agente y la estrategia

La `PlanningStrategy` es **sin estado** — es un transformador puro de (estado, memoria, herramientas) → prompt → acción. Toda la lógica de cómo razonar está aquí.

El `BaseAgent` es **con estado** — mantiene la memoria y conecta la estrategia con el LLM y las herramientas. Su `agent_id` es una huella legible: `"react__no_memory__gpt-4o"`.

### El bucle ReAct

En cada paso:

```
Thought: Necesito buscar quién escribió Hamlet.
Action: search
Action Input: {"query": "autor de Hamlet"}
                    ↓ (el engine ejecuta la herramienta)
Observation: Hamlet fue escrito por William Shakespeare...
                    ↓ (siguiente iteración)
Thought: Ya sé la respuesta.
Action: finish
Action Input: {"answer": "William Shakespeare"}
```

El stop sequence `"\nObservation:"` impide que el modelo alucine su propia observación — el engine la suministra con el resultado real de la herramienta.

---

## 5. ¿De dónde saca los datos?

### HotPotQA (dataset principal implementado)

HotPotQA es un dataset de preguntas de razonamiento multi-hop: para responderlas hay que combinar información de múltiples fuentes. Ejemplo:

> *"¿En qué año nació el fundador de la empresa que fabricó el primer iPhone?"*

El loader intenta descargarlo automáticamente de HuggingFace Hub:

```python
load_dataset("hotpot_qa", "fullwiki", split="validation")
```

**Si no hay conexión a internet** (CI, entorno offline), usa automáticamente el fixture local en `fixtures/hotpotqa_sample.json` (20 preguntas de ejemplo incluidas en el repo).

Cada tarea cargada tiene esta estructura interna:

```python
TaskInstance(
    task_id="hotpotqa_5abc123",
    input="¿Cuál es la capital de Francia?",   # la pregunta para el agente
    gold="París",                               # respuesta correcta para evaluar
    metadata={"type": "bridge", "level": "easy"}
)
```

### Añadir otros datasets

Implementar `TaskLoader` en `src/tasks/loaders/` y registrar en `src/tasks/loaders/__init__.py`. Ver sección [13. Cómo añadir cosas nuevas](#13-cómo-añadir-cosas-nuevas).

---

## 6. ¿Cómo se evalúa?

La evaluación tiene dos etapas.

### Etapa 1 — Puntuación de cada trajectory (Validator)

Compara la respuesta final del agente con la respuesta correcta (`gold`):

| Validator | Cuándo usarlo | Cómo funciona |
|---|---|---|
| `exact_match` | Respuestas con formato muy estricto | Case-insensitive string equality. 1.0 o 0.0. |
| `fuzzy_match` | Preguntas abiertas (HotPotQA, etc.) | Token-level F1 con stop words. Igual que SQuAD. Permite "William Shakespeare" == "Shakespeare". |
| `llm_judge` | Generación larga, múltiples respuestas válidas | LLM externo puntúa de 0–10. Con mitigación de sesgos (ver abajo). |

**Resultado:** cada `Trajectory` recibe un `score` ∈ [0,1] y un flag `success` (True si score ≥ 1.0).

### Etapa 2 — Métricas agregadas (Metrics)

Se calculan sobre el conjunto completo de trajectories del run:

| Métrica | Valor principal | Desglose |
|---|---|---|
| `success_rate` | Fracción de tareas donde al menos 1 trial tuvo éxito | Por nivel de dificultad (easy/medium/hard) |
| `pass_at_k` | P(al menos 1 de k trials aleatorios tiene éxito) — fórmula combinatoria exacta | pass@1, pass@3, pass@5 |
| `tokens_per_task` | Media de tokens totales por trajectory | — |
| `step_count` | Media de pasos ReAct por trajectory | Desglose success vs. failure |
| `tool_accuracy` | Fracción de llamadas a herramientas con argumentos válidos | error_rate, calls_per_episode, por herramienta |
| `failure_recovery` | Fracción de episodios con errores que aun así tuvieron éxito | episodes_with_errors, recovered |
| `latency` | P50 de latencia total por episodio (ms) | p50, p95, mean |

### El juez LLM (`llm_judge`) y mitigación de sesgos

Cuando se usa `validator: "llm_judge"`, cada predicción se evalúa con **4 prompts × n_samples llamadas independientes**:

- **2 perspectivas:** precisión factual + equivalencia semántica
- **2 órdenes:** predicción primero / gold primero → cancela el sesgo posicional
- **temperature > 0:** mide la varianza entre llamadas
- **Agregación:** media de todos los scores; si std_dev > 0.15, se emite WARNING de baja confianza

Para una explicación completa de los sesgos y mitigaciones, ver [docs/llm_judge_notes.md](docs/llm_judge_notes.md).

---

## 7. Estructura de carpetas

```
agents_benchmarking/
│
├── run_experiment.py          ← PUNTO DE ENTRADA. CLI para lanzar experimentos.
├── requirements.txt           ← Dependencias MVP (instalar primero).
├── requirements-advanced.txt  ← Dependencias opcionales (Anthropic, ChromaDB, etc.).
├── pyproject.toml             ← Config de pytest y ruff.
├── .env.example               ← Plantilla de variables de entorno (API keys).
│
├── configs/                   ← CONFIGURACIÓN DE EXPERIMENTOS
│   ├── base_config.yaml       ← Valores por defecto. Todo experimento hereda de aquí.
│   └── experiments/
│       └── react_hotpotqa.yaml ← Experimento concreto. Sobreescribe los defaults.
│
├── fixtures/                  ← DATOS OFFLINE para CI y desarrollo sin internet
│   ├── hotpotqa_sample.json   ← 20 preguntas de HotPotQA. Fallback automático.
│   └── search_responses.json  ← Respuestas de búsqueda simuladas (MockSearchTool).
│
├── results/                   ← RESULTADOS DE EXPERIMENTOS (generado automáticamente)
│   └── {run_id}/              ← Una carpeta por run. Nombre: "{id}__{timestamp}__{hash}"
│       ├── config.yaml        ← Snapshot exacto del config usado.
│       ├── metrics.json       ← Métricas en formato machine-readable.
│       ├── report.md          ← Tabla de métricas en Markdown (legible en GitHub).
│       ├── trajectories.jsonl ← Una línea JSON por trajectory completada.
│       └── traces.jsonl       ← Una línea JSON por paso de agente (si save_traces=true).
│
├── docs/
│   └── llm_judge_notes.md     ← Limitaciones y mitigaciones del juez LLM.
│
├── src/                       ← TODO EL CÓDIGO FUENTE
│   │
│   ├── schema.py              ← Modelos de datos centrales (Pydantic). IMPORTADO POR TODOS.
│   │                            Define: Action, Observation, Step, Trajectory,
│   │                            TaskInstance, AgentState, MetricResult, ToolResult.
│   │
│   ├── config.py              ← Modelos de configuración + load_config() + config_hash.
│   │                            Define: ExperimentConfig, AgentConfig, LLMConfig,
│   │                            MemoryConfig, ToolConfig, EvaluationConfig, etc.
│   │
│   ├── utils.py               ← Utilidades compartidas: seed_everything(), make_run_id(),
│   │                            configure_logging().
│   │
│   ├── orchestrator.py        ← ExperimentOrchestrator. Coordina todo el ciclo de vida.
│   │
│   ├── execution_engine.py    ← ExecutionEngine. El bucle de pasos act→observe→log.
│   │
│   ├── trace_logger.py        ← TraceLogger. Escribe traces.jsonl y trajectories.jsonl.
│   │                            Flush después de cada paso → crash-safe.
│   │
│   ├── agents/
│   │   ├── base.py            ← Agent (ABC) + BaseAgent (concreto).
│   │   │                        BaseAgent conecta: estrategia + memoria + LLM + herramientas.
│   │   └── factory.py         ← build_agent(config) → Agent.
│   │
│   ├── strategies/
│   │   ├── base.py            ← PlanningStrategy (ABC). Contrato: build_prompt + parse_response.
│   │   ├── direct.py          ← Respuesta directa sin herramientas. Baseline.
│   │   ├── react.py           ← ReAct completo con parser de Thought/Action/Action Input.
│   │   └── factory.py         ← build_strategy(name) → PlanningStrategy.
│   │
│   ├── memory/
│   │   ├── base.py            ← MemoryModule (ABC). Contrato: read/write/reset.
│   │   ├── no_memory.py       ← Sin memoria. read() → []. No-op en write/reset.
│   │   ├── window_buffer.py   ← Últimos N pasos en un deque.
│   │   ├── episodic.py        ← Buffer de reflexiones verbales para Reflexion.
│   │   │                        reset() es no-op. hard_reset() borra entre tareas.
│   │   └── factory.py         ← build_memory(config) → MemoryModule.
│   │
│   ├── llm/
│   │   ├── base.py            ← LLMBackend (ABC) + LLMResponse.
│   │   ├── openai_backend.py  ← OpenAIBackend con reintentos exponenciales (tenacity).
│   │   └── factory.py         ← build_llm_backend(config) → LLMBackend.
│   │
│   ├── tools/
│   │   ├── base.py            ← BaseTool (ABC) + ToolRegistry.
│   │   │                        ToolRegistry.validate_and_execute() es el único punto
│   │   │                        de entrada — los errores se convierten en ToolResult,
│   │   │                        nunca en excepciones.
│   │   ├── finish.py          ← FinishTool. Señal de terminación del episodio.
│   │   ├── calculator.py      ← Calculadora segura con allowlist de operadores.
│   │   ├── search.py          ← MockSearchTool (offline) + LiveSearchTool (DuckDuckGo).
│   │   └── factory.py         ← build_tool_registry(tool_configs) → ToolRegistry.
│   │
│   ├── tasks/
│   │   ├── base.py            ← TaskLoader (ABC) + TaskValidator (ABC) + TaskRegistry.
│   │   └── loaders/
│   │       └── hotpotqa.py    ← HotPotQALoader. HuggingFace Hub + fallback a fixture.
│   │
│   ├── evaluation/
│   │   ├── module.py          ← EvaluationModule. Orquesta Stage 1 y Stage 2.
│   │   ├── validators/
│   │   │   ├── base.py        ← ValidatorRegistry.
│   │   │   ├── exact_match.py ← Igualdad exacta case-insensitive.
│   │   │   ├── fuzzy_match.py ← Token F1 con stop words (estilo SQuAD).
│   │   │   └── llm_judge.py   ← LLMJudgeValidator con mitigación de sesgos.
│   │   └── metrics/
│   │       ├── base.py        ← Metric (ABC) + MetricRegistry.
│   │       ├── success_rate.py
│   │       ├── pass_at_k.py
│   │       ├── tokens.py
│   │       ├── steps.py
│   │       ├── tool_accuracy.py
│   │       ├── failure_recovery.py
│   │       └── latency.py
│   │
│   └── reporting/
│       ├── aggregator.py      ← ResultAggregator. Agrupa trajectories por task_id.
│       └── report_generator.py ← Escribe metrics.json y report.md.
│
└── tests/
    ├── conftest.py
    ├── unit/                  ← Tests unitarios (sin LLM real, sin red)
    │   ├── test_schema.py
    │   ├── test_config.py
    │   ├── test_agent.py
    │   ├── test_react_strategy.py
    │   ├── test_tools.py
    │   ├── test_tools_base.py
    │   ├── test_execution_engine.py
    │   ├── test_trace_logger.py
    │   ├── test_evaluation.py
    │   ├── test_metrics_phase5.py
    │   └── test_llm_judge.py
    └── integration/
        └── test_react_smoke.py ← Pipeline completo con LLM mockeado.
```

---

## 8. Instalación

### Requisitos previos

- Python 3.11+
- Una API key de OpenAI (mínimo) o Anthropic (opcional)

### Pasos

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd agents_benchmarking

# 2. Crear entorno virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 3. Instalar dependencias MVP
pip install -r requirements.txt

# 4. (Opcional) Dependencias avanzadas: Anthropic, ChromaDB, ALFWorld...
pip install -r requirements-advanced.txt

# 5. Configurar API keys
cp .env.example .env
# Editar .env y poner las keys reales
```

### Contenido de `.env`

```bash
OPENAI_API_KEY=sk-...       # Obligatorio para cualquier experimento con OpenAI
ANTHROPIC_API_KEY=sk-ant-... # Solo necesario si usas provider: "anthropic"
```

**Importante:** `.env` está en `.gitignore`. Nunca lo subas al repositorio.

---

## 9. Cómo ejecutar un experimento

### Experimento real (requiere API key)

```bash
python run_experiment.py --config configs/experiments/react_hotpotqa.yaml
```

### Dry run (valida el config sin llamar a ninguna API)

```bash
python run_experiment.py --config configs/experiments/react_hotpotqa.yaml --dry-run
```

Esto muestra un resumen del experimento y termina sin ejecutar nada:

```
╭─────────────── Agent Benchmark Suite ───────────────╮
│ react_hotpotqa_gpt4o_v1                              │
│ strategy=react | model=gpt-4o | memory=no_memory     │
│ tasks=hotpotqa n=100 | trials=5 | seed=42            │
│ config_hash=3f8a1c2b…                                │
╰──────────────────────────────────────────────────────╯
Dry run — config valid, exiting.
```

### Tests (sin API key, completamente offline)

```bash
# Todos los tests
python3 -m pytest

# Solo tests unitarios
python3 -m pytest tests/unit/

# Solo el smoke test de integración
python3 -m pytest tests/integration/

# Con salida detallada
python3 -m pytest -v

# Sólo los tests de un módulo específico
python3 -m pytest tests/unit/test_llm_judge.py -v
```

### Dónde aparecen los resultados

Después de un run real se crea automáticamente:

```
results/
└── react_hotpotqa_gpt4o_v1__20260422T143022__3f8a1c2b/
    ├── config.yaml        ← snapshot del YAML exacto usado
    ├── metrics.json       ← todas las métricas en JSON
    ├── report.md          ← tabla Markdown lista para GitHub
    ├── trajectories.jsonl ← una línea por trajectory
    └── traces.jsonl       ← una línea por paso de agente
```

---

## 10. Qué hay que rellenar / configurar

### Obligatorio para correr experimentos reales

| Qué | Dónde | Ejemplo |
|---|---|---|
| API key de OpenAI | `.env` | `OPENAI_API_KEY=sk-proj-...` |
| ID del experimento | Config YAML | `id: "mi_experimento_v1"` |

### Para crear un experimento nuevo

1. Copia `configs/experiments/react_hotpotqa.yaml`
2. Dale un `id` único (se usa como nombre de carpeta en results/)
3. Cambia lo que quieras comparar (estrategia, modelo, n_samples, etc.)
4. Ejecuta con `--config tu_nuevo_config.yaml`

### Para comparar dos estrategias

Crea dos YAMLs que sólo difieran en `agent.strategy`. El resto idéntico (misma semilla, mismas tareas, misma métrica). Ejemplo:

```yaml
# react_baseline.yaml
experiment:
  id: "baseline_react"
  seed: 42
agent:
  strategy: "react"
  ...

# direct_baseline.yaml
experiment:
  id: "baseline_direct"
  seed: 42
agent:
  strategy: "direct"
  ...
```

Ejecuta los dos y compara sus `metrics.json`.

### Para usar el juez LLM

Añade al YAML:

```yaml
evaluation:
  validator: "llm_judge"
  llm_judge_model: "claude-opus-4-7"   # modelo DIFERENTE al que se evalúa
```

Y asegúrate de tener `ANTHROPIC_API_KEY` en `.env` si usas Claude como juez.

---

## 11. Referencia completa de configuración YAML

Los YAMLs de experimento se fusionan con `configs/base_config.yaml`. Los campos del experimento sobreescriben los defaults.

```yaml
experiment:
  id: "nombre_unico"          # OBLIGATORIO. Identifica el run. No usar espacios.
  seed: 42                    # Semilla global. Afecta al shuffle de tareas y al LLM.
  n_trials: 5                 # Cuántas veces repetir cada tarea. Necesario para pass@k.
  max_steps: 25               # Máximo de pasos ReAct por trial. Evita bucles infinitos.
  output_dir: "results/"      # Dónde guardar los resultados.
  tags: ["react", "v1"]       # Etiquetas libres para organizar experimentos.

agent:
  strategy: "react"           # "direct" | "react" | "reflexion" | "plan_execute" | "tot"
  llm:
    provider: "openai"        # "openai" | "anthropic" | "local"
    model: "gpt-4o"           # Nombre del modelo (gpt-4o, gpt-4o-mini, claude-opus-4-7...)
    temperature: 0.0          # 0.0 para reproducibilidad. > 0 para muestreo estocástico.
    max_tokens: 1024          # Límite de tokens de salida por llamada.
  memory:
    type: "no_memory"         # "no_memory" | "window_buffer" | "episodic" | "vector_store"
    window_size: 10           # Solo para window_buffer: número de pasos a recordar.
    top_k: 5                  # Solo para vector_store: k más cercanos.
  tools:
    - name: "search"          # Nombre de la herramienta (debe estar en src/tools/).
      config:
        engine: "duckduckgo"  # Configuración específica de la herramienta.
        max_results: 3
    - name: "calculator"      # Sin config adicional.
    - name: "finish"          # SIEMPRE incluir finish si usas ReAct.

tasks:
  dataset: "hotpotqa"         # Dataset a usar. Debe estar registrado en TASK_REGISTRY.
  split: "validation"         # "train" | "validation" | "test"
  n_samples: 100              # Número de tareas a evaluar. Más = más caro pero más fiable.
  filter:                     # (Opcional) Filtrar por metadatos de la tarea.
    difficulty: ["hard"]      # Solo tareas difíciles.

evaluation:
  validator: "fuzzy_match"    # "exact_match" | "fuzzy_match" | "llm_judge"
  llm_judge_model: null       # Solo si validator="llm_judge". Modelo para el juez.
  metrics:                    # Lista de métricas a calcular.
    - "success_rate"
    - "pass_at_k"
    - "tokens_per_task"
    - "step_count"
    - "tool_accuracy"
    - "failure_recovery"
    - "latency"
  pass_k_values: [1, 3, 5]    # Para qué valores de k calcular pass@k.

logging:
  level: "INFO"               # "DEBUG" | "INFO" | "WARNING"
  save_traces: true           # Si false, no escribe traces.jsonl (más rápido, menos disco).
  trace_format: "jsonl"       # Solo "jsonl" soportado por ahora.
```

---

## 12. Qué se guarda en los resultados

### `trajectories.jsonl`

Una línea JSON por trajectory completada. Cada línea es un objeto con:

```json
{
  "run_id": "react_hotpotqa_v1__20260422T143022__3f8a1c2b",
  "task_id": "hotpotqa_5abc123",
  "agent_id": "react__no_memory__gpt-4o",
  "trial_num": 0,
  "seed": 42,
  "config_hash": "3f8a1c2b...",   // SHA-256 del config completo
  "steps": [...],                  // Lista de pasos
  "termination": "success",        // "success" | "max_steps" | "parse_error" | "llm_error"
  "final_answer": "William Shakespeare",
  "total_tokens": 1247,
  "total_latency_ms": 3421.5,
  "success": true,
  "score": 1.0
}
```

### `traces.jsonl`

Una línea JSON por **paso** del agente (más granular). Solo si `save_traces: true`. Útil para debugging y análisis de comportamiento.

### `metrics.json`

```json
{
  "run_id": "react_hotpotqa_v1__20260422T143022__3f8a1c2b",
  "agent_id": "react__no_memory__gpt-4o",
  "config": { ... },
  "metrics": {
    "success_rate": {
      "value": 0.62,
      "breakdown": {"easy": 0.85, "medium": 0.55, "hard": 0.30}
    },
    "pass_at_k": {
      "value": 0.71,
      "breakdown": {"pass@1": 0.62, "pass@3": 0.71, "pass@5": 0.78}
    },
    "step_count": {
      "value": 4.3,
      "breakdown": {"success": 3.1, "failure": 6.8}
    }
  }
}
```

### `report.md`

Tabla Markdown lista para GitHub. Ejemplo:

```markdown
# Experiment: react_hotpotqa_gpt4o_v1

**Strategy:** react | **Model:** gpt-4o | **Memory:** no_memory

## Metrics

| Metric | Value | Breakdown |
|--------|-------|-----------|
| success_rate | 0.6200 | easy=0.850, medium=0.550, hard=0.300 |
| pass_at_k | 0.7100 | pass@1=0.620, pass@3=0.710, pass@5=0.780 |
```

---

## 13. Cómo añadir cosas nuevas

### Añadir un nuevo dataset

1. Crea `src/tasks/loaders/mi_dataset.py`:

```python
from src.tasks.base import TaskLoader
from src.schema import TaskInstance

class MiDatasetLoader(TaskLoader):
    @property
    def name(self) -> str:
        return "mi_dataset"

    def load(self, split, n_samples, seed, filter_kwargs=None):
        # Cargar datos y devolver List[TaskInstance]
        ...
```

2. Registra en `src/tasks/loaders/__init__.py`:

```python
from src.tasks.loaders.mi_dataset import MiDatasetLoader
TASK_REGISTRY.register("mi_dataset", MiDatasetLoader)
```

3. Úsalo en el YAML: `dataset: "mi_dataset"`.

### Añadir una nueva estrategia

1. Crea `src/strategies/mi_estrategia.py` implementando `PlanningStrategy`:
   - `build_prompt(state, memory_context, tool_descriptions) -> str`
   - `parse_response(raw, state) -> Action` — **nunca debe lanzar excepciones**
   - `name` property

2. Registra en `src/strategies/factory.py`.

3. Añade el nombre al `Literal` en `src/config.py` → `AgentConfig.strategy`.

### Añadir una nueva herramienta

1. Crea `src/tools/mi_herramienta.py` heredando de `BaseTool`.

2. Registra en `src/tools/factory.py`.

3. Úsala en el YAML: `tools: [{name: "mi_herramienta"}]`.

### Añadir una nueva métrica

1. Crea `src/evaluation/metrics/mi_metrica.py` implementando `Metric`:
   - `name` property
   - `compute(trajectories, tasks) -> MetricResult`

2. Registra en `src/evaluation/metrics/__init__.py`:

```python
from src.evaluation.metrics.mi_metrica import MiMetrica
METRIC_REGISTRY.register(MiMetrica)
```

3. Úsala en el YAML: `metrics: ["mi_metrica"]`.

---

## 14. Tests

El suite tiene 152 tests (sin contar los tuyos propios). Se pueden correr todos offline — ningún test hace llamadas reales a APIs.

```bash
python3 -m pytest                    # todos
python3 -m pytest tests/unit/        # solo unitarios
python3 -m pytest tests/integration/ # solo integración
python3 -m pytest -v --tb=short      # verbose con traceback corto
```

La cobertura por módulo:

| Fichero de test | Qué prueba |
|---|---|
| `test_schema.py` | Serialización/deserialización de todos los modelos Pydantic |
| `test_config.py` | load_config, deep merge, config_hash determinista |
| `test_agent.py` | BaseAgent: act(), observe(), memory wiring, agent_id |
| `test_react_strategy.py` | Parser de Thought/Action/Action Input, casos borde |
| `test_tools.py` | search (offline fixture), calculator (allowlist), finish |
| `test_tools_base.py` | ToolRegistry: validate_and_execute, herramienta desconocida |
| `test_execution_engine.py` | Bucle de pasos, terminación por max_steps/abort/success |
| `test_trace_logger.py` | JSONL output, crash-safety, round-trip load |
| `test_evaluation.py` | Validators (exact/fuzzy), PassAtK, SuccessRate, EvaluationModule |
| `test_metrics_phase5.py` | StepCount, ToolAccuracy, FailureRecovery, Latency |
| `test_llm_judge.py` | Parser de scores, mitigación de sesgo, robustez ante fallos |
| `test_react_smoke.py` | Pipeline completo con LLM mockeado (end-to-end) |

---

## 15. Papers de referencia

Cada decisión de diseño está justificada por investigación publicada:

| Paper | Cita | Relevancia en el código |
|---|---|---|
| ReAct | Yao et al., ICLR 2023. arXiv:2210.03629 | `src/strategies/react.py` |
| Reflexion | Shinn et al., NeurIPS 2023. arXiv:2303.11366 | `src/memory/episodic.py`, `EpisodicMemory.reset()` no-op |
| Plan-and-Solve | Wang et al., ACL 2023. arXiv:2305.04091 | `src/strategies/` (pendiente) |
| Tree of Thoughts | Yao et al., NeurIPS 2023. arXiv:2305.10601 | `src/strategies/` (pendiente) |
| Toolformer | Schick et al., NeurIPS 2023. arXiv:2302.04761 | `src/tools/base.py`, ToolRegistry |
| HotPotQA | Yang et al., EMNLP 2018 | `src/tasks/loaders/hotpotqa.py` |
| SQuAD token F1 | Rajpurkar et al., 2016 | `src/evaluation/validators/fuzzy_match.py` |
| Pass@k formula | Chen et al., 2021 / Shinn et al., 2023 | `src/evaluation/metrics/pass_at_k.py` |
| HELM | Liang et al., 2022. arXiv:2211.09110 | Multi-metric design, `EvaluationModule` |
| AgentBench | Liu et al., ICLR 2024. arXiv:2308.03688 | Arquitectura general del benchmark |
| WebArena | Zhou et al., 2023. arXiv:2307.13854 | Reproducibility-first: config_hash, seed |
| MT-Bench / LLM Judge | Zheng et al., NeurIPS 2023. arXiv:2306.05685 | `src/evaluation/validators/llm_judge.py` |
| Positional bias | Wang et al., 2023. arXiv:2309.03882 | Prompt ordering en LLMJudgeValidator |
| Verbosity bias | Dubois et al., 2024. arXiv:2404.04475 | Templates del juez LLM |
