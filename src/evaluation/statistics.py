from typing import List, Dict, Any
import logging
import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)

# (result key, display name) for the five paired metrics; latency is the only "lower is better" one.
METRICS = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevance", "Answer Relevance"),
    ("context_precision", "Context Precision"),
    ("latency_ms", "Latency (ms)"),
    ("top_rerank_score", "Cross-Encoder Top-1"),
]
_REPORT_KEYS = {"top_rerank_score": "rerank_score"}
_ALT_DESC = {"greater": "H1: Hybrid > Naive", "less": "H1: Hybrid < Naive", "two-sided": "Two-Sided"}


class StatisticalAnalyzer:
    """Performs formal inferential statistical analysis on paired RAG evaluation metrics."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def analyze_metric(
        self,
        naive_scores: List[float],
        hybrid_scores: List[float],
        metric_name: str,
        alternative: str = "auto"
    ) -> Dict[str, Any]:
        """Runs Shapiro-Wilk on the paired differences, then a paired t-test or Wilcoxon test plus effect size and 95% CI."""
        naive_arr = np.asarray(naive_scores, dtype=float)
        hybrid_arr = np.asarray(hybrid_scores, dtype=float)
        deltas = hybrid_arr - naive_arr
        n = len(deltas)

        if n < 3:
            return {"metric": metric_name, "error": "Sample size too small (n < 3)"}

        if alternative == "auto":
            # Latency/time/cost: the treatment should be lower; quality metrics: higher.
            lower_is_better = any(k in metric_name.lower() for k in ["latency", "time", "cost", "ms", "latencia", "tiempo"])
            alt = "less" if lower_is_better else "greater"
        else:
            alt = alternative

        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1))

        variance_note = None
        if std_delta <= 1e-9:
            # s^2 = 0: no test is defined; report p = 1 deterministically instead of NaN.
            shapiro_stat, shapiro_p, is_normal = 1.0, 1.0, True
            test_name, test_stat, p_val = "N/A (Sin varianza)", 0.0, 1.0
            effect_size_name, effect_size_val = "Cohen's d", 0.0
            ci_lower = ci_upper = mean_delta
            variance_note = (
                "Sin varianza entre condiciones (diferencias nulas)"
                if abs(mean_delta) < 1e-9
                else f"Sin varianza entre condiciones (Δ constante = {mean_delta:.3f})"
            )
        else:
            shapiro_stat, shapiro_p = stats.shapiro(deltas)
            is_normal = bool(shapiro_p > self.alpha)
            alt_desc = _ALT_DESC.get(alt, alt)

            if is_normal:
                test_name = f"Paired Student's t-test ({alt_desc})"
                res = stats.ttest_rel(hybrid_arr, naive_arr, alternative=alt)
                test_stat, p_val = float(res.statistic), float(res.pvalue)
                effect_size_name, effect_size_val = "Cohen's d", mean_delta / std_delta  # paired d = mean(Δ)/sd(Δ)

                half_width = stats.t.ppf(1.0 - self.alpha / 2.0, df=n - 1) * std_delta / np.sqrt(n)
                ci_lower, ci_upper = mean_delta - half_width, mean_delta + half_width
            else:
                test_name = f"Wilcoxon Signed-Rank Test ({alt_desc})"
                try:
                    w_res = stats.wilcoxon(hybrid_arr, naive_arr, alternative=alt, zero_method="wilcox")
                    test_stat, p_val = float(w_res.statistic), float(w_res.pvalue)
                except ValueError as e:
                    logger.warning("Wilcoxon test failed for %s (%s); reporting p = 1.0", metric_name, e)
                    test_stat, p_val = 0.0, 1.0

                pos, neg = int(np.sum(deltas > 0)), int(np.sum(deltas < 0))
                effect_size_name = "Rank-Biserial r"
                effect_size_val = (pos - neg) / (pos + neg) if pos + neg else 0.0

                # Percentile bootstrap CI of the mean difference (seeded for reproducible reports).
                boot_means = np.random.default_rng(0).choice(deltas, size=(1000, n), replace=True).mean(axis=1)
                ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

            test_stat = 0.0 if np.isnan(test_stat) else test_stat
            p_val = 1.0 if np.isnan(p_val) else p_val

        return {
            "metric": metric_name,
            "n": n,
            "mean_naive": round(float(np.mean(naive_arr)), 4),
            "std_naive": round(float(np.std(naive_arr, ddof=1)), 4),
            "mean_hybrid": round(float(np.mean(hybrid_arr)), 4),
            "std_hybrid": round(float(np.std(hybrid_arr, ddof=1)), 4),
            "mean_delta": round(mean_delta, 4),
            "std_delta": round(std_delta, 4),
            "median_delta": round(float(np.median(deltas)), 4),
            "shapiro_stat": round(float(shapiro_stat), 4),
            "shapiro_p": round(float(shapiro_p), 5),
            "is_normal": is_normal,
            "test_name": test_name,
            "test_stat": round(test_stat, 4),
            "p_value": round(p_val, 5),
            "is_significant": bool(p_val < self.alpha),
            "effect_size_name": effect_size_name,
            "effect_size_val": round(float(effect_size_val), 4),
            "ci_95": (round(float(ci_lower), 4), round(float(ci_upper), 4)),
            "alternative": alt,
            "variance_note": variance_note
        }

    @staticmethod
    def _breakdown(results: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
        """Mean paired deltas per value of `field` (e.g. question type or domain)."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            groups.setdefault(r.get(field, "general"), []).append(r)
        return {
            key: {
                "count": len(rows),
                "faithfulness_delta": round(float(np.mean([r["delta"]["faithfulness"] for r in rows])), 4),
                "relevance_delta": round(float(np.mean([r["delta"]["answer_relevance"] for r in rows])), 4),
                "precision_delta": round(float(np.mean([r["delta"]["context_precision"] for r in rows])), 4),
            }
            for key, rows in groups.items()
        }

    def generate_full_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates a comprehensive evaluation report from paired evaluation rows."""
        overall = {
            _REPORT_KEYS.get(key, key): self.analyze_metric(
                [r["naive"].get(key, 0.0) for r in results],
                [r["hybrid"].get(key, 0.0) for r in results],
                name,
            )
            for key, name in METRICS
        }
        return {
            "overall": overall,
            "by_category": self._breakdown(results, "type"),
            "by_domain": self._breakdown(results, "domain"),
            "sample_size": len(results)
        }

    def _conclusions(self, report: Dict[str, Any]) -> List[str]:
        """Discussion bullets derived from the observed statistics (never asserts an unsupported effect)."""
        ov, n = report["overall"], report["sample_size"]

        def usable(key: str):
            d = ov.get(key, {})
            return None if "error" in d else d

        def wins(d) -> bool:
            return bool(d) and not d["variance_note"] and d["is_significant"] and d["mean_delta"] > 0

        def fmt(d) -> str:
            return f"Δ = {d['mean_delta']:+.3f}, p = {d['p_value']:.4f}, {d['effect_size_name']} = {d['effect_size_val']:.3f}"

        rel, rerank, prec, faith = (usable(k) for k in ("answer_relevance", "rerank_score", "context_precision", "faithfulness"))
        if rel is None:
            return [f"- **Potencia insuficiente:** con N = {n} (< 3 pares) no es posible aplicar contrastes de hipótesis; amplíe la muestra."]

        out = []
        if wins(rel) or wins(rerank):
            evidence = [f"{name} ({fmt(d)})" for name, d in (("Answer Relevance", rel), ("Cross-Encoder Top-1", rerank)) if wins(d)]
            out.append(
                "- **Superioridad de la topología de grafo:** la recuperación híbrida (grafo KùzuDB + vector + Cross-Encoder) supera "
                "al control puramente vectorial con significancia estadística en " + " y ".join(evidence) + ". "
                "Las rutas de relaciones aportan evidencia que el vecindario vectorial no recupera, y el generador la convierte en "
                "respuestas más pertinentes."
            )
        else:
            if rel["variance_note"] and (rerank is None or rerank["variance_note"]):
                reason = "las diferencias pareadas son nulas (varianza cero), de modo que el contraste no es informativo"
            else:
                reason = (
                    f"no se alcanzó significancia (Answer Relevance: p = {rel['p_value']:.3f}"
                    + (f"; Cross-Encoder: p = {rerank['p_value']:.3f}" if rerank else "") + ")"
                )
            out.append(
                f"- **Evidencia no concluyente:** {reason}. Con N = {n} la potencia para detectar efectos pequeños o "
                "medios es limitada, por lo que la ausencia de significancia no equivale a ausencia de efecto: se requiere ampliar "
                "N (p. ej. al menos 30-50 pares) y reportar el tamaño del efecto y su IC 95% antes de aceptar o rechazar la hipótesis central."
            )
            if max(rel["mean_naive"], rel["mean_hybrid"]) < 0.2:
                out.append(
                    f"- **Moderación del generador (efecto suelo):** la relevancia media es {rel['mean_naive']:.3f} (control) y "
                    f"{rel['mean_hybrid']:.3f} (tratamiento). Con un generador pequeño y ventana de contexto reducida, abundan las "
                    "abstenciones, lo que comprime la métrica hacia 0 y enmascara la contribución del recuperador; conviene repetir el "
                    "experimento con un generador de mayor capacidad o mayor `num_ctx`."
                )
            cats = report.get("by_category", {})
            if len(cats) > 1:
                (lo, lo_d), (hi, hi_d) = (lambda r: (r[0], r[-1]))(sorted(cats.items(), key=lambda kv: kv[1]["relevance_delta"]))
                if hi_d["relevance_delta"] != lo_d["relevance_delta"]:
                    out.append(
                    f"- **Tipo de consulta:** el mayor Δ de relevancia aparece en consultas *{hi}* ({hi_d['relevance_delta']:+.3f}, "
                    f"n = {hi_d['count']}) y el menor en *{lo}* ({lo_d['relevance_delta']:+.3f}, n = {lo_d['count']}). Las consultas "
                    "factuales cerradas se resuelven con un único fragmento y favorecen al baseline vectorial; la ventaja del grafo "
                    "debería concentrarse en consultas multi-hop, cuya muestra por estrato es aún pequeña para inferir."
                )

        if prec:
            out.append(
                "- **Precisión contextual:** "
                + ("sin varianza entre condiciones (ambos recuperadores sitúan el documento gold en las mismas posiciones)."
                   if prec["variance_note"] else
                   ("mejora significativa" if wins(prec) else "diferencia no significativa") + f" ({fmt(prec)}).")
            )
        if faith:
            out.append(
                "- **Fidelidad:** " + ("sin varianza entre condiciones." if faith["variance_note"] else f"{fmt(faith)}.")
                + " Las abstenciones puntúan como fieles por diseño, por lo que esta métrica debe leerse junto a Answer Relevance."
            )
        return out

    def format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Renders formatted Markdown report suitable for TFM thesis documentation."""
        ov = report["overall"]
        lines = [
            "# Reporte Estadistico de Evaluacion RAGAS (Muestras Emparejadas)\n",
            f"**Benchmark:** NeurIPS 2024 CRAG (Comprehensive RAG Benchmark)\n",
            f"**Tamano Muestral Total ($N$):** {report['sample_size']} consultas evaluadas en diseno intra-sujeto.\n",
            "## 1. Tabla Resumen de Hipotesis y Metricas Principales\n",
            "| Metrica Evaluada | Control (Naive) | Tratamiento (Hybrid) | Diferencia Media ($\\Delta$) | Normalidad ($p_{\\text{Shapiro}}$) | Test Aplicado | $p$-valor ($H_1$) | Tamano Efecto | IC 95% ($\\Delta$) | Significativo? |",
            "|---|---|---|---|---|---|---|---|---|---|"
        ]

        metrics_list = [
            ("faithfulness", "Faithfulness (RAGAS)"),
            ("answer_relevance", "Answer Relevance (RAGAS)"),
            ("context_precision", "Context Precision (RAGAS)"),
            ("latency_ms", "Latencia Respuesta (ms)"),
            ("rerank_score", "Cross-Encoder Top-1 (Semantic)"),
        ]

        for m_key, m_name in metrics_list:
            if m_key not in ov or "error" in ov[m_key]:
                continue
            data = ov[m_key]
            if data.get("variance_note"):
                norm_str = "Constante (varianza=0)"
                sig_str = "NO (sin varianza)"
                test_col = "N/A (Sin varianza)"
                p_val_str = "1.0000"
            else:
                norm_str = f"Normal (p={data['shapiro_p']:.3f})" if data['is_normal'] else f"No-Normal (p={data['shapiro_p']:.3f})"
                sig_str = "**SI (p < 0.05)**" if data['is_significant'] else "NO (p >= 0.05)"
                test_col = data['test_name'].split(' (')[0]
                p_val_str = f"**{data['p_value']:.4f}**" if data['is_significant'] else f"{data['p_value']:.4f}"
            lines.append(
                f"| **{m_name}** | {data['mean_naive']:.3f} $\\pm$ {data['std_naive']:.3f} | "
                f"{data['mean_hybrid']:.3f} $\\pm$ {data['std_hybrid']:.3f} | "
                f"**{data['mean_delta']:+.3f}** | {norm_str} | {test_col} | "
                f"{p_val_str} | {data['effect_size_name']}: {data['effect_size_val']:.3f} | "
                f"[{data['ci_95'][0]:.3f}, {data['ci_95'][1]:.3f}] | {sig_str} |"
            )

        has_zero_var = any(ov[m_key].get("variance_note") for m_key, _ in metrics_list if m_key in ov and "error" not in ov[m_key])
        if has_zero_var:
            lines.append("\n> [!NOTE]\n> **Nota metodologica sobre varianza nula:** Las metricas con la indicacion 'Sin varianza' obtuvieron valores identicos entre las condiciones Control y Tratamiento en todas las observaciones ($s^2_\\Delta = 0$, p. ej. precision 1.000 vs 1.000). Al carecer de dispersion muestral, el contraste de hipotesis no es aplicable y se reporta de forma determinista $p = 1.0000$ sin generar indeterminaciones (`NaN`).\n")

        lines.append("\n## 2. Desglose por Dominio CRAG\n")
        lines.append("| Dominio | Muestra ($n$) | $\\Delta$ Faithfulness | $\\Delta$ Answer Relevance | $\\Delta$ Context Precision |")
        lines.append("|---|---|---|---|---|")
        for dom, d_data in report.get("by_domain", {}).items():
            lines.append(
                f"| **{dom.capitalize()}** | {d_data['count']} | "
                f"{d_data['faithfulness_delta']:+.3f} | {d_data['relevance_delta']:+.3f} | {d_data['precision_delta']:+.3f} |"
            )

        lines.append("\n## 3. Desglose por Complejidad de Consulta\n")
        lines.append("| Tipo de Consulta | Muestra ($n$) | $\\Delta$ Faithfulness | $\\Delta$ Answer Relevance | $\\Delta$ Context Precision |")
        lines.append("|---|---|---|---|---|")
        for cat, c_data in report["by_category"].items():
            lines.append(
                f"| **{cat.capitalize()}** | {c_data['count']} | "
                f"{c_data['faithfulness_delta']:+.3f} | {c_data['relevance_delta']:+.3f} | {c_data['precision_delta']:+.3f} |"
            )

        lines.append("\n## 4. Conclusiones Académicas y Discusión para el TFM\n")
        lines.extend(self._conclusions(report))
        lines.append("")

        return "\n".join(lines)
