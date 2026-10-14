# Reporte Estadistico de Evaluacion RAGAS (Muestras Emparejadas)

**Benchmark:** NeurIPS 2024 CRAG (Comprehensive RAG Benchmark)

**Tamano Muestral Total ($N$):** 5 consultas evaluadas en diseno intra-sujeto.

## 1. Tabla Resumen de Hipotesis y Metricas Principales

| Metrica Evaluada | Control (Naive) | Tratamiento (Hybrid) | Diferencia Media ($\Delta$) | Normalidad ($p_{\text{Shapiro}}$) | Test Aplicado | $p$-valor ($H_1$) | Tamano Efecto | IC 95% ($\Delta$) | Significativo? |
|---|---|---|---|---|---|---|---|---|---|
| **Faithfulness (RAGAS)** | 0.400 $\pm$ 0.548 | 0.400 $\pm$ 0.548 | **+0.000** | Normal (p=0.325) | Paired Student's t-test | 0.5000 | Cohen's d: 0.000 | [-0.878, 0.878] | NO (p >= 0.05) |
| **Answer Relevance (RAGAS)** | 0.700 $\pm$ 0.447 | 0.800 $\pm$ 0.447 | **+0.100** | No-Normal (p=0.000) | Wilcoxon Signed-Rank Test | 0.5000 | Rank-Biserial r: 1.000 | [0.000, 0.300] | NO (p >= 0.05) |
| **Context Precision (RAGAS)** | 0.900 $\pm$ 0.224 | 0.900 $\pm$ 0.224 | **+0.000** | Constante (varianza=0) | N/A (Sin varianza) | 1.0000 | Cohen's d: 0.000 | [0.000, 0.000] | NO (sin varianza) |
| **Latencia Respuesta (ms)** | 13992.384 $\pm$ 4061.001 | 7619.042 $\pm$ 4738.680 | **-6373.342** | Normal (p=0.296) | Paired Student's t-test | **0.0018** | Cohen's d: -2.724 | [-9278.596, -3468.088] | **SI (p < 0.05)** |
| **Cross-Encoder Top-1 (Semantic)** | 6.973 $\pm$ 2.144 | 6.973 $\pm$ 2.144 | **+0.000** | Constante (varianza=0) | N/A (Sin varianza) | 1.0000 | Cohen's d: 0.000 | [0.000, 0.000] | NO (sin varianza) |

> [!NOTE]
> **Nota metodologica sobre varianza nula:** Las metricas con la indicacion 'Sin varianza' obtuvieron valores identicos entre las condiciones Control y Tratamiento en todas las observaciones ($s^2_\Delta = 0$, p. ej. precision 1.000 vs 1.000). Al carecer de dispersion muestral, el contraste de hipotesis no es aplicable y se reporta de forma determinista $p = 1.0000$ sin generar indeterminaciones (`NaN`).


## 2. Desglose por Dominio CRAG

| Dominio | Muestra ($n$) | $\Delta$ Faithfulness | $\Delta$ Answer Relevance | $\Delta$ Context Precision |
|---|---|---|---|---|
| **Music** | 2 | -0.500 | +0.000 | +0.000 |
| **Sports** | 2 | +0.000 | +0.000 | +0.000 |
| **Open** | 1 | +1.000 | +0.500 | +0.000 |

## 3. Desglose por Complejidad de Consulta

| Tipo de Consulta | Muestra ($n$) | $\Delta$ Faithfulness | $\Delta$ Answer Relevance | $\Delta$ Context Precision |
|---|---|---|---|---|
| **Simple_w_condition** | 1 | -1.000 | +0.000 | +0.000 |
| **Multi-hop** | 1 | +0.000 | +0.000 | +0.000 |
| **Comparison** | 1 | +0.000 | +0.000 | +0.000 |
| **Simple** | 2 | +0.500 | +0.250 | +0.000 |

## 4. Conclusiones Académicas y Discusión para el TFM

- **Evidencia no concluyente:** no se alcanzó significancia (Answer Relevance: p = 0.500; Cross-Encoder: p = 1.000). Con N = 5 la potencia para detectar efectos pequeños o medios es limitada, por lo que la ausencia de significancia no equivale a ausencia de efecto: se requiere ampliar N (p. ej. al menos 30-50 pares) y reportar el tamaño del efecto y su IC 95% antes de aceptar o rechazar la hipótesis central.
- **Tipo de consulta:** el mayor Δ de relevancia aparece en consultas *simple* (+0.250, n = 2) y el menor en *simple_w_condition* (+0.000, n = 1). Las consultas factuales cerradas se resuelven con un único fragmento y favorecen al baseline vectorial; la ventaja del grafo debería concentrarse en consultas multi-hop, cuya muestra por estrato es aún pequeña para inferir.
- **Precisión contextual:** sin varianza entre condiciones (ambos recuperadores sitúan el documento gold en las mismas posiciones).
- **Fidelidad:** Δ = +0.000, p = 0.5000, Cohen's d = 0.000. Las abstenciones puntúan como fieles por diseño, por lo que esta métrica debe leerse junto a Answer Relevance.
