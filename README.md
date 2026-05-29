# Redes Bayesianas

Proyecto académico que implementa dos **redes bayesianas discretas** con `pgmpy` para resolver problemas de inferencia probabilística en dominios distintos:

1. **Predicción de lluvia en Xalapa, Veracruz** a partir de datos climáticos históricos.
2. **Diagnóstico médico** de la relación entre cáncer y fatiga, considerando factores de riesgo.

Autor: **Capistran Ortiz Diego**

---

## Contenido del repositorio

```
Redes Bayesianas/
│
├── ProyectoClimaXalapa.py        # Red bayesiana para prediccion de lluvia
├── DiagnosticoCancerFatiga.py    # Red bayesiana para diagnostico medico
│
├── DataSets/                      # Datos climaticos historicos (CSV)
│   ├── XOBVC.csv                  # Estacion Observatorio Xalapa (principal)
│   ├── C30135.csv, C30192.csv, C30488.csv
│   ├── VERVC.csv, XCEVC.csv, XCFVC.csv, XCRVC.csv, XOLVC.csv, XSUVC.csv
│   └── DatasetsRed_Diego.zip      # Datasets comprimidos
│
├── red_bayesiana.png              # DAG generado: diagnostico de cancer
├── red_bayesiana_dag.png          # DAG generado: prediccion de lluvia
├── requirements.txt               # Dependencias de Python
└── README.md
```

---

## Requisitos

- **Python 3.10 o superior**
- Sistema operativo: Windows, Linux o macOS

### Librerías principales

| Librería | Uso |
|----------|-----|
| `pgmpy` | Definición de la red bayesiana, CPDs e inferencia |
| `networkx` | Representación del DAG |
| `matplotlib` | Visualización gráfica de la red |
| `numpy` | Operaciones numéricas |
| `pandas` | Lectura y procesamiento de los CSV |

### Instalación

```bash
# 1. Clonar o descargar el proyecto
cd "Redes Bayesianas"

# 2. (Opcional pero recomendado) crear entorno virtual
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 1. Proyecto Clima Xalapa

### Objetivo
Estimar la **probabilidad de lluvia** en la ciudad de Xalapa, Veracruz, a partir de la temporada del año y variables meteorológicas observables.

### Fuente de datos
- **Estación XOBVC** — Observatorio de Xalapa
- **Periodo:** 2015 – 2024 (n = 3,423 registros)
- **Origen:** Sistema de Información Hidrológica (SIH)

### Estructura del DAG (5 nodos, 7 aristas)

```
                Temporada
              /     |     \
             v      v      v
        TMax     Evaporacion  Lluvia
         |  \       ^         ^  ^
         v   \______|         |  |
       Amplitud_Termica ------+  |
                              |  |
                Evaporacion --+--+
```

| Nodo | Estados | Tipo |
|------|---------|------|
| `Temporada` | Seca (sequía), Lluviosa | Raíz |
| `Temperatura_Maxima` | Fresco, Cálido, Caluroso | Intermedio |
| `Amplitud_Termica` | Baja, Media, Alta | Intermedio |
| `Evaporacion` | Baja, Media, Alta | Intermedio |
| `Lluvia` | Sí, No | Objetivo |

### Discretización por terciles

| Variable | Umbrales |
|----------|----------|
| Temporada | Seca (Nov–Abr) / Lluviosa (May–Oct) |
| Temp. máxima | Fresco ≤ 24.4 °C / Cálido ≤ 27.4 °C / Caluroso > 27.4 °C |
| Amplitud térmica | Baja ≤ 10.1 °C / Media ≤ 12.6 °C / Alta > 12.6 °C |
| Evaporación | Baja ≤ 1.8 mm / Media ≤ 3.15 mm / Alta > 3.15 mm |
| Lluvia | No (0 mm) / Sí (> 0 mm) |

### Cómo ejecutar

```bash
python ProyectoClimaXalapa.py
```

El script:
1. Define la red y sus CPDs (calculadas desde el dataset con suavizado de Laplace).
2. Valida el modelo con `check_model()`.
3. Genera el DAG y lo guarda como `red_bayesiana_dag.png`.
4. Ejecuta consultas de inferencia con `VariableElimination`.
5. Lanza un **menú interactivo** que solicita al usuario los valores y devuelve la probabilidad de lluvia.

### Función principal de predicción

```python
predecir_lluvia(temporada=None,
                temperatura_maxima=None,
                amplitud=None,
                evaporacion=None)
```

Acepta entre **0 y 4 parámetros**. Si no se proporciona ninguno, devuelve la **probabilidad marginal** P(Lluvia). Con uno o más, devuelve la probabilidad **condicional**.

### Tipo de red
- **Estructuralmente predictiva** (las aristas van de causa → efecto, terminando en `Lluvia`).
- **Operativamente flexible**: soporta tanto inferencia **predictiva** (Temporada → Lluvia) como **diagnóstica** (Lluvia → Temporada) e **intercausal** (*explaining away* entre variables del mismo nivel).

---

## 2. Diagnóstico Cáncer – Fatiga

### Objetivo
Construir un mini-sistema experto que dado un síntoma (**fatiga**) o factores de riesgo (**fumar**, **contaminación ambiental**) estime la probabilidad de tener **cáncer**.

### Estructura del DAG (4 nodos, 3 aristas)

```
   Fumador ─────┐
                ├──▶ Cancer ──▶ Fatiga
   Contaminacion ┘
```

| Nodo | Estados |
|------|---------|
| `Fumador` | Sí, No |
| `Contaminacion` | Alta, Baja |
| `Cancer` | Sí, No |
| `Fatiga` | Sí, No |

### Probabilidades a priori
- P(Fumador = Sí) = 0.30
- P(Contaminacion = Alta) = 0.20
- P(Cáncer | Fumador, Contaminación): valores plausibles para cada combinación.
- P(Fatiga | Cáncer = Sí) = 0.80, P(Fatiga | Cáncer = No) = 0.10.

### Cómo ejecutar

```bash
python DiagnosticoCancerFatiga.py
```

El script:
1. Construye la red con sus 4 CPDs.
2. Ejecuta **6 consultas** que ilustran razonamiento diagnóstico e *intercausal* (efecto **explaining away**).
3. Visualiza el DAG y lo guarda como `red_bayesiana.png`.

### Consultas incluidas

| # | Consulta | Tipo |
|---|----------|------|
| 1 | P(Cáncer \| Fatiga = Sí) | Diagnóstica |
| 2 | P(Contaminación \| Cáncer = Sí) | Diagnóstica |
| 3 | P(Contaminación \| Cáncer = Sí, Fumador = Sí) | Intercausal |
| 4 | P(Cáncer \| Fatiga = No) | Diagnóstica |
| 5 | P(Contaminación \| Cáncer = No) | Diagnóstica |
| 6 | P(Contaminación \| Cáncer = Sí, Fumador = No) | Intercausal |

---

## Conceptos clave aplicados

- **Red bayesiana discreta**: DAG donde cada nodo tiene una CPD (Tabla de Probabilidad Condicional).
- **Inferencia por eliminación de variables (`VariableElimination`)**: algoritmo exacto para calcular probabilidades marginales y condicionales.
- **Tipos de razonamiento**:
  - *Predictivo / causal:* P(Efecto | Causa)
  - *Diagnóstico:* P(Causa | Efecto)
  - *Intercausal o explaining away:* dos causas independientes se vuelven dependientes al observar el efecto.
- **Suavizado de Laplace** para evitar probabilidades cero en celdas con pocos datos.
- **Discretización por terciles** para variables continuas.

---

## Salidas esperadas

Al ejecutar cualquiera de los dos scripts se generan:

1. **Salida por consola** con tablas de probabilidades, validación del modelo y resultados de las consultas.
2. **Imagen PNG** con el DAG (`red_bayesiana_dag.png` o `red_bayesiana.png`).
3. En el caso de Xalapa, **menú interactivo** que pide datos al usuario y devuelve la probabilidad de lluvia.

---

## Referencias

- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques.* MIT Press.
- Documentación de `pgmpy`: https://pgmpy.org
- Sistema de Información Hidrológica (SIH) — CONAGUA.

---

## Licencia y uso

Proyecto desarrollado con fines **académicos**. El uso de los datasets y la reproducción del código se permite citando al autor.
