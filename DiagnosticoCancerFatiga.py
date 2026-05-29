#===========================================================================
#    RED BAYESIANA PARA PREDICCIÓN DE CAUSAS DE CÁNCER Y FATIGA SEGÚN FACTORES
#    Implementación con pgmpy
#    Autor: Capistran Ortiz Diego
#    Apoyo: Gemini  
#    Programa de Red Bayesiana para Predicción de cáncer dado factores como 
#    fumador, contaminación y sus síntomas asociados como la fatiga. Se realizan 
#    consultas de inferencia
#    Fecha: 21 de Mayo del 2026
# ===========================================================================

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# 1. Definir la estructura de la red (nodos y aristas)
model = DiscreteBayesianNetwork([
    ('Fumador', 'Cancer'), 
    ('Contaminacion', 'Cancer'),
    ('Cancer', 'Fatiga')
])

# 2. Definir las Tablas de Probabilidad Condicional (CPDs)

# P(Fumador) - Asumimos que el 30% de la población fuma
cpd_fumador = TabularCPD(variable='Fumador', variable_card=2,
                         values=[[0.3],   # Sí
                                 [0.7]],  # No
                         state_names={'Fumador': ['Sí', 'No']})

# P(Contaminacion) - Asumimos que el 20% vive en zonas de alta contaminación
cpd_contaminacion = TabularCPD(variable='Contaminacion', variable_card=2,
                               values=[[0.2],   # Alta
                                       [0.8]],  # Baja
                               state_names={'Contaminacion': ['Alta', 'Baja']})

# P(Cancer | Fumador, Contaminacion)
# Las columnas representan las combinaciones de (Fumador, Contaminacion)
# Orden de columnas: (Sí, Alta), (Sí, Baja), (No, Alta), (No, Baja)
cpd_cancer = TabularCPD(variable='Cancer', variable_card=2,
                        values=[[0.08, 0.05, 0.02, 0.001],  # Cáncer = Sí
                                [0.92, 0.95, 0.98, 0.999]], # Cáncer = No
                        evidence=['Fumador', 'Contaminacion'], 
                        evidence_card=[2, 2],
                        state_names={'Cancer': ['Sí', 'No'],
                                     'Fumador': ['Sí', 'No'],
                                     'Contaminacion': ['Alta', 'Baja']})

# P(Fatiga | Cancer)
cpd_fatiga = TabularCPD(variable='Fatiga', variable_card=2,
                        values=[[0.80, 0.10],   # Fatiga = Sí
                                [0.20, 0.90]],  # Fatiga = No
                        evidence=['Cancer'], evidence_card=[2],
                        state_names={'Fatiga': ['Sí', 'No'],
                                     'Cancer': ['Sí', 'No']})

# 3. Añadir las CPDs al modelo y validar
model.add_cpds(cpd_fumador, cpd_contaminacion, cpd_cancer, cpd_fatiga)
assert model.check_model()

# 4. Inferencia
infer = VariableElimination(model)

print("--- DIAGNÓSTICO MÉDICO BAYESIANO ---")

# Caso 1: Paciente llega con Fatiga. ¿Cuál es la probabilidad de que tenga cáncer?
print("\n1. P(Cáncer | Fatiga = Sí):")
res_1 = infer.query(variables=['Cancer'], evidence={'Fatiga': 'Sí'})
print(res_1)

# Caso 2: Sabemos que tiene cáncer. ¿Qué tan probable es que viva en alta contaminación?
print("\n2. P(Contaminación | Cáncer = Sí):")
res_2 = infer.query(variables=['Contaminacion'], evidence={'Cancer': 'Sí'})
print(res_2)

# Caso 3: Tiene cáncer, pero descubrimos que ES FUMADOR.
# ¿Qué pasa con la probabilidad de la contaminación?
print("\n3. P(Contaminación | Cáncer = Sí, Fumador = Sí):")
res_3 = infer.query(variables=['Contaminacion'], evidence={'Cancer': 'Sí', 'Fumador': 'Sí'})
print(res_3)

# Caso 1 Modificado: Paciente no llega con Fatiga. ¿Cuál es la probabilidad de que tenga cáncer?
print("\n1. P(Cáncer | Fatiga = No):")
res_1_mod = infer.query(variables=['Cancer'], evidence={'Fatiga': 'No'})
print(res_1_mod)

# Caso 2 Modificado: Sabemos que no tiene cáncer. ¿Qué tan probable es que viva en alta contaminación?
print("\n2. P(Contaminación | Cáncer = No):")
res_2_mod = infer.query(variables=['Contaminacion'], evidence={'Cancer': 'No'})
print(res_2_mod)

# Caso 3 Modificado: Tiene cáncer, pero descubrimos que NO ES FUMADOR.
print("\n3. P(Contaminación | Cáncer = Sí, Fumador = No):")
res_3_mod = infer.query(variables=['Contaminacion'], evidence={'Cancer': 'Sí', 'Fumador': 'No'})
print(res_3_mod)


# 5. Visualización de la red con networkx + matplotlib
pos = {
    'Fumador':       (0, 2),
    'Contaminacion': (2, 2),
    'Cancer':        (1, 1),
    'Fatiga':        (1, 0),
}
# 5.1 Se define la red con sus nodos
plt.figure(figsize=(7, 5))
nx.draw(
    model,
    pos=pos,
    with_labels=True,
    node_color="#fb0000",
    node_size=4500,
    font_size=8,
    font_weight='bold',
    arrows=True,
    arrowsize=20,
    edge_color='#4169E1',
)
plt.title("Red Bayesiana: Diagnóstico de Cáncer")
plt.tight_layout()
plt.savefig("red_bayesiana.png", dpi=150)
plt.show()