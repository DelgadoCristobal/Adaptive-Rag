# Reporte Estadistico de Evaluacion RAGAS (Muestras Emparejadas)

**Benchmark:** NeurIPS 2024 CRAG (Comprehensive RAG Benchmark)

**Tamano Muestral Total ($N$):** 40 consultas evaluadas en diseno intra-sujeto.

## 1. Tabla Resumen de Hipotesis y Metricas Principales

| Metrica Evaluada | Control (Naive) | Tratamiento (Hybrid) | Diferencia Media ($\Delta$) | Normalidad ($p_{\text{Shapiro}}$) | Test Aplicado | $p$-valor ($H_1$) | Tamano Efecto | IC 95% ($\Delta$) | Significativo? |
|---|---|---|---|---|---|---|---|---|---|
| **Faithfulness (RAGAS)** | 0.812 $\pm$ 0.370 | 0.762 $\pm$ 0.392 | **-0.050** | No-Normal (p=0.000) | Wilcoxon Signed-Rank Test | 0.8330 | Rank-Biserial r: -0.600 | [-0.138, 0.037] | NO (p >= 0.05) |
| **Answer Relevance (RAGAS)** | 0.463 $\pm$ 0.485 | 0.438 $\pm$ 0.483 | **-0.025** | No-Normal (p=0.000) | Wilcoxon Signed-Rank Test | 0.6272 | Rank-Biserial r: -0.333 | [-0.138, 0.087] | NO (p >= 0.05) |
| **Context Precision (RAGAS)** | 0.938 $\pm$ 0.232 | 0.938 $\pm$ 0.232 | **+0.000** | Constante (varianza=0) | N/A (Sin varianza) | 1.0000 | Cohen's d: 0.000 | [0.000, 0.000] | NO (sin varianza) |
| **Latencia Respuesta (ms)** | 5401.917 $\pm$ 1264.276 | 5245.327 $\pm$ 215.675 | **-156.590** | No-Normal (p=0.000) | Wilcoxon Signed-Rank Test | 0.8264 | Rank-Biserial r: 0.200 | [-578.623, 78.712] | NO (p >= 0.05) |
| **Cross-Encoder Top-1 (Semantic)** | 4.715 $\pm$ 3.738 | 4.738 $\pm$ 3.727 | **+0.022** | No-Normal (p=0.000) | Wilcoxon Signed-Rank Test | 0.1587 | Rank-Biserial r: 1.000 | [0.000, 0.067] | NO (p >= 0.05) |

> [!NOTE]
> **Nota metodologica sobre varianza nula:** Las metricas con la indicacion 'Sin varianza' obtuvieron valores identicos entre las condiciones Control y Tratamiento en todas las observaciones ($s^2_\Delta = 0$, p. ej. precision 1.000 vs 1.000). Al carecer de dispersion muestral, el contraste de hipotesis no es aplicable y se reporta de forma determinista $p = 1.0000$ sin generar indeterminaciones (`NaN`).


## 2. Desglose por Dominio CRAG

| Dominio | Muestra ($n$) | $\Delta$ Faithfulness | $\Delta$ Answer Relevance | $\Delta$ Context Precision |
|---|---|---|---|---|
| **Finance** | 2 | +0.500 | -0.500 | +0.000 |
| **Movie** | 9 | -0.056 | +0.111 | +0.000 |
| **Music** | 7 | +0.000 | -0.071 | +0.000 |
| **Open** | 9 | +0.000 | -0.167 | +0.000 |
| **Sports** | 13 | -0.192 | +0.077 | +0.000 |

## 3. Desglose por Complejidad de Consulta

| Tipo de Consulta | Muestra ($n$) | $\Delta$ Faithfulness | $\Delta$ Answer Relevance | $\Delta$ Context Precision |
|---|---|---|---|---|
| **Simple** | 14 | -0.143 | -0.107 | +0.000 |
| **Multi-hop** | 7 | +0.000 | +0.000 | +0.000 |
| **Aggregation** | 4 | +0.250 | -0.375 | +0.000 |
| **Comparison** | 8 | -0.125 | +0.250 | +0.000 |
| **Simple_w_condition** | 7 | +0.000 | +0.000 | +0.000 |

## 4. Conclusiones Académicas y Discusión para el TFM

- **Evidencia no concluyente:** no se alcanzó significancia (Answer Relevance: p = 0.627; Cross-Encoder: p = 0.159). Con N = 40 la potencia para detectar efectos pequeños o medios es limitada, por lo que la ausencia de significancia no equivale a ausencia de efecto: se requiere ampliar N (p. ej. al menos 30-50 pares) y reportar el tamaño del efecto y su IC 95% antes de aceptar o rechazar la hipótesis central.
- **Tipo de consulta:** el mayor Δ de relevancia aparece en consultas *comparison* (+0.250, n = 8) y el menor en *aggregation* (-0.375, n = 4). Las consultas factuales cerradas se resuelven con un único fragmento y favorecen al baseline vectorial; la ventaja del grafo debería concentrarse en consultas multi-hop, cuya muestra por estrato es aún pequeña para inferir.
- **Precisión contextual:** sin varianza entre condiciones (ambos recuperadores sitúan el documento gold en las mismas posiciones).
- **Fidelidad:** Δ = -0.050, p = 0.8330, Rank-Biserial r = -0.600. Las abstenciones puntúan como fieles por diseño, por lo que esta métrica debe leerse junto a Answer Relevance.
