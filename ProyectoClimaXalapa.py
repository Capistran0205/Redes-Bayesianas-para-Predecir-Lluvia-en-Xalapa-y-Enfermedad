#===========================================================================
#    RED BAYESIANA PARA PREDICCIÓN DE LLUVIA EN XALAPA, VERACRUZ
#    Implementación con pgmpy
#    Autor: Capistran Ortiz Diego
#    Apoyo: Claude    
#    Programa de Red Bayesiana para Predicción de Lluvia en Xalapa, Veracrurz.
#    Considerando la fuente de datos, Sistema de Información Hidrológica (SIH)
#    de la Estación XOBVC — Observatorio de Xalapa (2015–2024).
#    Fecha: 21 de Mayo del 2026
# ===========================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import sys

# =========================================================================
# 1. DEFINICIÓN DE LA ESTRUCTURA DE LA RED BAYESIANAS
# =========================================================================
# Aristas del DAG según el modelo propuesto:
#   Temporada → Temperatura_Maxima
#   Temporada → Amplitud_Termica
#   Temporada → Evaporacion
#   Temporada → Lluvia
#   Temperatura_Maxima → Evaporacion
#   Temperatura_Maxima → Lluvia
#   Amplitud_Termica → Lluvia
#   Evaporacion → Lluvia

modelo = DiscreteBayesianNetwork([
    ('Temporada', 'Temperatura_Maxima'),
    ('Temporada', 'Evaporacion'),
    ('Temporada', 'Lluvia'),
    ('Temperatura_Maxima', 'Amplitud_Termica'),
    ('Temperatura_Maxima', 'Evaporacion'),
    ('Amplitud_Termica', 'Lluvia'),
    ('Evaporacion', 'Lluvia')
])

# =========================================================================
# 2. TABLAS DE PROBABILIDAD CONDICIONAL (CPTs)
#    Calculadas a partir de datos históricos XOBVC 2015–2024 (n=3,423)
#    Discretización por terciles + suavizado de Laplace para datos escasos
# =========================================================================
# Consideraciones Previas
# ---- Umbrales de discretización (mediante terciles, 33.33%) ----
# Temporada:           Seca (Nov–Abr)  | Lluviosa (May–Oct)
# Temperatura Máxima:  Fresco ≤ 24.4°C | Cálido ≤ 27.4°C | Caluroso > 27.4°C
# Amplitud Térmica:    Baja ≤ 10.1°C   | Media ≤ 12.6°C   | Alta > 12.6°C
# Evaporación:         Baja ≤ 1.8mm    | Media ≤ 3.15mm   | Alta > 3.15mm
# Lluvia:              No (0mm)        | Sí (>0mm)

# =========================================================================
# TABLA 1: P(Temporada) — PROBABILIDAD SIMPLE (MARGINAL)
# Nodo raíz, sin padres en el DAG.
# =========================================================================
# Estados: [Seca (sequía), Lluviosa]
#   P(Seca)     = 0.4806
#   P(Lluviosa) = 0.5194

cpd_temporada = TabularCPD(
    variable='Temporada',
    variable_card=2,
    values=[[0.4806],   # Seca (sequía)
            [0.5194]],  # Lluviosa
    state_names={'Temporada': ['Seca (sequia)', 'Lluviosa']}
)

# =========================================================================
# TABLA 2: P(Temperatura_Maxima | Temporada) — PROBABILIDAD CONDICIONAL
# 1 padre: Temporada.
# =========================================================================
# Estados TMax: [Fresco, Cálido, Caluroso]
#
#                  | Seca (sequía) | Lluviosa
# Fresco           |   0.5623      |  0.1305
# Cálido           |   0.2207      |  0.4409
# Caluroso         |   0.2170      |  0.4286

cpd_tmax = TabularCPD(
    variable='Temperatura_Maxima',
    variable_card=3,
    values=[[0.5623, 0.1305],   # Fresco
            [0.2207, 0.4409],   # Cálido
            [0.2170, 0.4286]],  # Caluroso
    evidence=['Temporada'],
    evidence_card=[2],
    state_names={
        'Temperatura_Maxima': ['Fresco', 'Calido', 'Caluroso'],
        'Temporada': ['Seca (sequia)', 'Lluviosa']
    }
)

# =========================================================================
# TABLA 3: P(Amplitud_Termica | Temperatura_Maxima) — PROB. CONDICIONAL
# 1 padre: Temperatura_Maxima.
# La amplitud termica (TMax - TMin) depende directamente de la
# temperatura maxima. La temporada la influye solo de manera
# indirecta a traves de TMax.
# =========================================================================
# Estados Amplitud: [Baja, Media, Alta]
#
#                  | Fresco  | Calido  | Caluroso
# Baja             |  0.6690 |  0.2807 |  0.0411
# Media            |  0.2187 |  0.4656 |  0.3208
# Alta             |  0.1124 |  0.2537 |  0.6381

cpd_amplitud = TabularCPD(
    variable='Amplitud_Termica',
    variable_card=3,
    values=[[0.6690, 0.2807, 0.0411],   # Baja
            [0.2187, 0.4656, 0.3208],   # Media
            [0.1124, 0.2537, 0.6381]],  # Alta
    evidence=['Temperatura_Maxima'],
    evidence_card=[3],
    state_names={
        'Amplitud_Termica': ['Baja', 'Media', 'Alta'],
        'Temperatura_Maxima': ['Fresco', 'Calido', 'Caluroso']
    }
)

# =========================================================================
# TABLA 4: P(Evaporacion | Temporada, Temperatura_Maxima)
# — PROBABILIDAD CONDICIONAL
# 2 padres: Temporada, Temperatura_Maxima.
# =========================================================================
# Orden de columnas: Temporada × TMax
# (Seca,Fresco) (Seca,Cálido) (Seca,Caluroso) (Lluv,Fresco) (Lluv,Cálido) (Lluv,Caluroso)

cpd_evaporacion = TabularCPD(
    variable='Evaporacion',
    variable_card=3,
    values=[
        # Evaporación = Baja
        [0.5903, 0.2590, 0.0672, 0.6638, 0.3265, 0.0932],
        # Evaporación = Media
        [0.2627, 0.4077, 0.3277, 0.2457, 0.4235, 0.3570],
        # Evaporación = Alta
        [0.1470, 0.3333, 0.6050, 0.0905, 0.2500, 0.5499]
    ],
    evidence=['Temporada', 'Temperatura_Maxima'],
    evidence_card=[2, 3],
    state_names={
        'Evaporacion': ['Baja', 'Media', 'Alta'],
        'Temporada': ['Seca (sequia)', 'Lluviosa'],
        'Temperatura_Maxima': ['Fresco', 'Calido', 'Caluroso']
    }
)

# =========================================================================
# TABLA 5: P(Lluvia | Temporada, Amplitud_Termica, Evaporacion)
# — PROBABILIDAD CONDICIONAL (BINARIA)
# 3 padres: Temporada, Amplitud_Termica, Evaporacion.
# 18 combinaciones (2 x 3 x 3).
# Temperatura_Maxima NO es padre directo de Lluvia; su influencia
# llega indirectamente a traves de Amplitud_Termica y Evaporacion.
# =========================================================================
# Orden de columnas: Temporada(2) x Amplitud(3) x Evaporacion(3)
#
# Temporada:   0=Seca, 1=Lluviosa
# Amplitud:    0=Baja, 1=Media, 2=Alta
# Evaporacion: 0=Baja, 1=Media, 2=Alta

cpt_data = {
    # ===== TEMPORADA SECA =====
    (0,0,0): (0.8131, 0.1869),  # Seca, Amp=Baja, Evap=Baja     (n=412)
    (0,0,1): (0.5865, 0.4135),  # Seca, Amp=Baja, Evap=Media    (n=133)
    (0,0,2): (0.6296, 0.3704),  # Seca, Amp=Baja, Evap=Alta     (n=54)
    (0,1,0): (0.3553, 0.6447),  # Seca, Amp=Media, Evap=Baja    (n=152)
    (0,1,1): (0.2941, 0.7059),  # Seca, Amp=Media, Evap=Media   (n=136)
    (0,1,2): (0.3750, 0.6250),  # Seca, Amp=Media, Evap=Alta    (n=112)
    (0,2,0): (0.1800, 0.8200),  # Seca, Amp=Alta, Evap=Baja     (n=100)
    (0,2,1): (0.1381, 0.8619),  # Seca, Amp=Alta, Evap=Media    (n=239)
    (0,2,2): (0.1270, 0.8730),  # Seca, Amp=Alta, Evap=Alta     (n=307)

    # ===== TEMPORADA LLUVIOSA =====
    (1,0,0): (0.8972, 0.1028),  # Lluviosa, Amp=Baja, Evap=Baja   (n=282)
    (1,0,1): (0.7836, 0.2164),  # Lluviosa, Amp=Baja, Evap=Media  (n=171)
    (1,0,2): (0.7222, 0.2778),  # Lluviosa, Amp=Baja, Evap=Alta   (n=90)
    (1,1,0): (0.7000, 0.3000),  # Lluviosa, Amp=Media, Evap=Baja  (n=160)
    (1,1,1): (0.5402, 0.4598),  # Lluviosa, Amp=Media, Evap=Media (n=311)
    (1,1,2): (0.4655, 0.5345),  # Lluviosa, Amp=Media, Evap=Alta  (n=275)
    (1,2,0): (0.6410, 0.3590),  # Lluviosa, Amp=Alta, Evap=Baja   (n=39)
    (1,2,1): (0.4022, 0.5978),  # Lluviosa, Amp=Alta, Evap=Media  (n=179)
    (1,2,2): (0.2620, 0.7380),  # Lluviosa, Amp=Alta, Evap=Alta   (n=271)
}

# Construir la matriz de valores para pgmpy
# Filas: estados de Lluvia [Si, No]
# Columnas: 18 combinaciones ordenadas
si_vals = []
no_vals = []
for i_temp in range(2):
    for i_amp in range(3):
        for i_evap in range(3):
            p_si, p_no = cpt_data[(i_temp, i_amp, i_evap)]
            si_vals.append(p_si)
            no_vals.append(p_no)

cpd_lluvia = TabularCPD(
    variable='Lluvia',
    variable_card=2,
    values=[si_vals, no_vals],
    evidence=['Temporada', 'Amplitud_Termica', 'Evaporacion'],
    evidence_card=[2, 3, 3],
    state_names={
        'Lluvia': ['Si', 'No'],
        'Temporada': ['Seca (sequia)', 'Lluviosa'],
        'Amplitud_Termica': ['Baja', 'Media', 'Alta'],
        'Evaporacion': ['Baja', 'Media', 'Alta']
    }
)

# =========================================================================
# 3. AGREGAR CPTs AL MODELO Y VERIFICAR QUE NO DE ERROR
# =========================================================================
modelo.add_cpds(cpd_temporada, cpd_tmax, cpd_amplitud, cpd_evaporacion, cpd_lluvia)

print("=" * 65)
print("  RED BAYESIANA — PREDICCIÓN DE LLUVIA EN XALAPA")
print("  Datos: XOBVC (Observatorio de Xalapa) · 2015–2024")
print("=" * 65)

# Verificar consistencia del modelo
if modelo.check_model():
    print("\n[OK] Modelo valido: la estructura y las CPTs son consistentes.")
else:
    print("\n[ERROR] El modelo no es consistente.")

# Información del modelo
print(f"\nNodos:   {modelo.nodes()}")
print(f"Aristas: {modelo.edges()}")
print(f"Total de parámetros: {sum(cpd.get_values().size for cpd in modelo.get_cpds())}")

# =========================================================================
# 4. VISUALIZACIÓN DEL GRAFO DIRIGIDO ACÍCLICO (DAG)
# =========================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')
fig.canvas.manager.set_window_title("Funciones de membresía - Pluma difusa")
# Crear grafo dirigido desde el modelo
G = nx.DiGraph()
G.add_edges_from(modelo.edges())

# Posiciones manuales para replicar la estructura del DAG original
pos = {
    'Temporada':          (0.50, 0.95),
    'Temperatura_Maxima': (0.15, 0.60),
    'Amplitud_Termica':   (0.85, 0.60),
    'Evaporacion':        (0.15, 0.25),
    'Lluvia':             (0.55, 0.10)
}

# Colores por tipo de probabilidad
node_colors = {
    'Temporada':          '#e74c3c',  # Rojo — nodo raiz (prob. simple)
    'Temperatura_Maxima': '#3498db',  # Azul — prob. condicional
    'Amplitud_Termica':   '#3498db',  # Azul — prob. condicional
    'Evaporacion':        '#3498db',  # Azul — prob. condicional
    'Lluvia':             '#f1c40f'   # Amarillo — nodo objetivo
}

# Etiquetas con formato legible (multilinea para que quepan en los circulos)
labels = {
    'Temporada':          'Temporada',
    'Temperatura_Maxima': 'Temperatura\nMaxima',
    'Amplitud_Termica':   'Amplitud\nTermica',
    'Evaporacion':        'Evaporacion',
    'Lluvia':             'Lluvia'
}

# Radio del circulo (en coordenadas de datos)
NODE_RADIUS = 0.11

# Dibujar aristas
nx.draw_networkx_edges(
    G, pos, ax=ax,
    edge_color='#000000',
    arrows=True,
    arrowstyle='-|>',
    arrowsize=18,
    node_size=1200,
    connectionstyle='arc3,rad=0.05',
    width=1.5,
    min_source_margin=45,
    min_target_margin=45
)

# Dibujar nodos como circulos
for node in G.nodes():
    x, y = pos[node]
    circle = Circle(
        (x, y), radius=NODE_RADIUS,
        facecolor=node_colors[node],
        edgecolor='#000000', linewidth=2,
        zorder=5
    )
    ax.add_patch(circle)
    ax.text(
        x, y, labels[node],
        ha='center', va='center',
        fontsize=8, fontweight='bold',
        color='#000000' if node != 'Lluvia' else '#1a1a2e',
        zorder=6
    )

# Leyenda
legend_elements = [
    mpatches.Patch(facecolor='#e74c3c', edgecolor='white', label='Prob. Simple (nodo raiz)'),
    mpatches.Patch(facecolor='#3498db', edgecolor='white', label='Prob. Condicional'),
    mpatches.Patch(facecolor='#f1c40f', edgecolor='white', label='Nodo objetivo (Lluvia)')
]
legend = ax.legend(
    handles=legend_elements, loc='lower left',
    fontsize=8, facecolor="#ffffff", edgecolor='#30363d',
    labelcolor="#000000", framealpha=0.95
)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.10)
ax.set_aspect('equal')
ax.axis('off')

ax.text(
    0.50, 1.08,
    'Red Bayesiana — Prediccion de Lluvia en Xalapa',
    ha='center', va='center', fontsize=14, fontweight='bold',
    color="#000000", transform=ax.transData
)

plt.tight_layout()
plt.savefig('red_bayesiana_dag.png', dpi=200, bbox_inches='tight',
            facecolor='#ffffff', edgecolor='none')
print("\n[OK] Grafo del DAG guardado en: red_bayesiana_dag.png")
plt.show()

# =========================================================================
# 5. MOTOR DE INFERENCIA
# =========================================================================
inferencia = VariableElimination(modelo)

print("\n" + "=" * 65)
print("  CONSULTAS DE INFERENCIA")
print("=" * 65)

# ---- Consulta 4: Escenario de máxima probabilidad de lluvia ----
print("\nConsulta 4: P(Lluvia | Lluviosa, Fresco, Amp=Baja, Evap=Baja)")
print("   (Escenario de alta probabilidad de lluvia)")
resultado = inferencia.query(
    variables=['Lluvia'],
    evidence={
        'Temporada': 'Lluviosa',
        'Temperatura_Maxima': 'Fresco',
        'Amplitud_Termica': 'Baja',
        'Evaporacion': 'Baja'
    }
)
print(resultado)


# =========================================================================
# 6. FUNCION DE PREDICCION INTERACTIVA
# =========================================================================
print("\n" + "=" * 65)
print("  SISTEMA DE PREDICCIÓN INTERACTIVO")
print("=" * 65)

def predecir_lluvia(temporada=None, temperatura_maxima=None, amplitud=None, evaporacion=None):
    """
    Predice la probabilidad de lluvia dados los valores observados.
    Acepta cualquier combinacion de 1, 2, 3 o 4 parametros. Si no se
    proporciona ninguno, retorna la probabilidad marginal P(Lluvia).

    Parametros (todos opcionales):
        temporada:          'Seca (sequia)' o 'Lluviosa'
        temperatura_maxima: 'Fresco', 'Calido' o 'Caluroso'
        amplitud:           'Baja', 'Media' o 'Alta'
        evaporacion:        'Baja', 'Media' o 'Alta'

    Retorna:
        dict con P(Si) y P(No)
    """
    # Mapeo nombre-parametro -> (nombre-nodo-en-red, etiqueta-impresion)
    mapeo = [
        ('Temporada',          'Temporada',        temporada),
        ('Temperatura_Maxima', 'Temperatura Max.', temperatura_maxima),
        ('Amplitud_Termica',   'Amplitud Termica', amplitud),
        ('Evaporacion',        'Evaporacion',      evaporacion),
    ]

    # Construir evidencia solo con los parametros que se pasaron
    evidencia = {nodo: valor for nodo, _, valor in mapeo if valor is not None}

    if evidencia:
        resultado = inferencia.query(variables=['Lluvia'], evidence=evidencia)
    else:
        # Sin evidencia: probabilidad marginal P(Lluvia)
        resultado = inferencia.query(variables=['Lluvia'])

    p_si = resultado.values[0]
    p_no = resultado.values[1]

    print(f"\n  {'─' * 50}")
    print(f"  Prediccion para Xalapa")
    print(f"  {'─' * 50}")
    if evidencia:
        for _, etiqueta, valor in mapeo:
            if valor is not None:
                print(f"  {etiqueta:<20} {valor}")
    else:
        print(f"  (sin evidencia — probabilidad marginal)")
    print(f"  {'─' * 50}")
    print(f"  P(Lluvia = Si) = {p_si:.4f} ({p_si*100:.1f}%)")
    print(f"  P(Lluvia = No) = {p_no:.4f} ({p_no*100:.1f}%)")
    print(f"  {'─' * 50}")

    if p_si >= 0.7:
        print(f"  --> PROBABILIDAD ALTA de lluvia")
    elif p_si >= 0.4:
        print(f"  --> PROBABILIDAD MODERADA de lluvia")
    else:
        print(f"  --> PROBABILIDAD BAJA de lluvia")

    return {'P(Si)': p_si, 'P(No)': p_no}

def determinar_temporada(m1, m2):
    if m1 == 'noviembre' and m2 == 'abril':
        return 'Seca (sequia)'
    elif m1 == 'mayo' and m2 == 'octubre':
        return 'Lluviosa'
    else:
        return None

def determinar_temp_maxima(c):
    if c > 0 and c <= 24.4:
        return 'Fresco'
    elif c > 24.4 and c <= 27.4:
        return 'Cálido'
    elif c > 27.4:
        return 'Caluroso'
    else:
        return None

def determinar_amplitud_term(c):
    if c > 0 and c <= 10.1:
        return 'Baja'
    elif c > 10.1 and c <= 12.6:
        return 'Media'
    elif c > 12.6:
        return 'Alta'
    else:
        return None
    
def determinar_evaporacion(c):
    if c > 0 and c <= 1.8:
        return 'Baja'
    elif c > 1.8 and c <= 3.15:
        return 'Media'
    elif c > 3.15:
        return 'Alta'
    else:
        return None

# ========================================
# 7. PRUEBA DEL SISTEMA
# =========================================

if __name__ ==  "__main__":

 while True:
        # Ejemplos de predicción
        primer_intervalo = ''
        segundo_intervalo = ''
        temporada = ''
        amplitud = ''
        temp_max = ''
        evaporacion = ''
        decision = ''   
        print('''
                Rango de Valores para cada variable:\n
                # Temporada:           Seca (Nov–Abr)  | Lluviosa (May–Oct)\n
                # Temperatura Máxima:  Fresco ≤ 24.4°C | Cálido ≤ 27.4°C | Caluroso > 27.4°C\n
                # Amplitud Térmica:    Baja ≤ 10.1°C   | Media ≤ 12.6°C   | Alta > 12.6°C\n
                # Evaporación:         Baja ≤ 1.8mm    | Media ≤ 3.15mm   | Alta > 3.15mm\n
        ''')
        
    
        primer_intervalo = input('\nMes inicial para Temporada: ')
        segundo_intervalo = input('\n Mes final para Temporada: ')
        temporada = determinar_temporada(primer_intervalo.lower(), segundo_intervalo.lower())
        
        c = float(input("\nIngrese la temperatura actual en °C para determinar la Amplitud Térmica: "))
        amplitud = determinar_amplitud_term(c)

        tm = float(input("\nIngrese la temperatura actual en °C para determinar la Temperatura Máxima: "))
        temp_max = determinar_temp_maxima(tm)

        m = float(input("\nIngrese la evaporación: "))
        evaporacion = determinar_evaporacion(m)

        print(f"\n>> Ejemplo dado los datos del usuario: ")
        predecir_lluvia(
            temporada=temporada,
            amplitud=amplitud,
            evaporacion=evaporacion
        )

        decision = input('¿Desea realizar otra inferencia probabilística? (s/n): ')
        decision = decision.lower()

        if decision == 'n':
            print('Ha salido vuelva pronto')
            sys.exit(0)
        else:
            print('Recargando sistema de inferencia difuso')