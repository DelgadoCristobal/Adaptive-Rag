from typing import List, Dict, Any, Optional
import logging
import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)

# (result key, display name) for the five paired metrics.
METRICS = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevance", "Answer Relevance"),
    ("context_precision", "Context Precision"),
    ("latency_ms", "Latency (ms)"),
    ("top_rerank_score", "Cross-Encoder Top-1"),
]
_REPORT_KEYS = {"top_rerank_score": "rerank_score"}

# Pre-declared direction per metric ("higher" or "lower" is better). It never changes the test (always
# two-sided); it only decides which arm is better. Unknown metrics log a warning and default to "higher".
METRIC_DIRECTION = {
    "faithfulness": "higher",
    "answer_relevance": "higher",
    "context_precision": "higher",
    "latency_ms": "lower",
    "top_rerank_score": "higher",
}
_DISPLAY_TO_KEY = {name: key for key, name in METRICS}

# Diagnostic metrics are reported but excluded from the Holm family and from the conclusions.
DIAGNOSTIC_METRICS = {"top_rerank_score"}
_DIAGNOSTIC_NOTE = (
    "Diagnostic only: the hybrid candidate pool is a superset of the naive pool, so the top-1 "
    "cross-encoder score favours hybrid by construction."
)


def metric_direction(metric: str) -> str:
    """Direction for a metric given its result key or display name; unknown -> warning + "higher"."""
    key = _DISPLAY_TO_KEY.get(metric, metric)
    if key not in METRIC_DIRECTION:
        logger.warning("No pre-declared direction for metric %r; defaulting to 'higher'", metric)
        return "higher"
    return METRIC_DIRECTION[key]


def rank_biserial_kerby(deltas: np.ndarray) -> float:
    """Matched-pairs rank-biserial r (Kerby): (W+ - W-) / (W+ + W-) over the ranks of non-zero |delta|."""
    d = np.asarray(deltas, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))  # average ranks for ties
    w_pos, w_neg = float(ranks[d > 0].sum()), float(ranks[d < 0].sum())
    return (w_pos - w_neg) / (w_pos + w_neg)


def holm_adjust(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values, returned in input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


class StatisticalAnalyzer:
    """Performs formal inferential statistical analysis on paired RAG evaluation metrics."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def analyze_metric(
        self,
        naive_scores: List[float],
        hybrid_scores: List[float],
        metric_name: str,
        direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Two-sided Wilcoxon signed-rank test, Kerby rank-biserial r and a two-sided bootstrap CI of mean delta.

        `is_significant` uses the raw p-value here; `generate_full_report` re-decides it on the
        Holm-adjusted p-value across the outcome family.
        """
        naive_arr = np.asarray(naive_scores, dtype=float)
        hybrid_arr = np.asarray(hybrid_scores, dtype=float)
        deltas = hybrid_arr - naive_arr
        n = len(deltas)

        if n < 3:
            return {"metric": metric_name, "error": "Sample size too small (n < 3)"}

        direction = direction or metric_direction(metric_name)
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1))
        effect_size_val = rank_biserial_kerby(deltas)

        variance_note = None
        if np.all(np.abs(deltas) <= 1e-9):
            # Every pair is identical: there is nothing to test, so report p = 1 instead of NaN.
            # A constant NON-zero Δ is not this case: it is a consistent win and goes to Wilcoxon.
            test_name, test_stat, p_val = "N/A (Sin varianza)", 0.0, 1.0
            ci_lower = ci_upper = mean_delta
            variance_note = "Sin varianza entre condiciones (diferencias nulas)"
        else:
            test_name = "Wilcoxon Signed-Rank Test (Two-Sided)"
            try:
                w_res = stats.wilcoxon(hybrid_arr, naive_arr, alternative="two-sided", zero_method="wilcox")
                test_stat, p_val = float(w_res.statistic), float(w_res.pvalue)
            except ValueError as e:
                logger.warning("Wilcoxon test failed for %s (%s); reporting p = 1.0", metric_name, e)
                test_stat, p_val = 0.0, 1.0
            test_stat = 0.0 if np.isnan(test_stat) else test_stat
            p_val = 1.0 if np.isnan(p_val) else p_val

            # Two-sided percentile bootstrap CI of the mean difference (seeded for reproducible reports).
            boot_means = np.random.default_rng(0).choice(deltas, size=(1000, n), replace=True).mean(axis=1)
            ci_lower, ci_upper = np.percentile(boot_means, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])

        result = {
            "metric": metric_name,
            "n": n,
            "mean_naive": round(float(np.mean(naive_arr)), 4),
            "std_naive": round(float(np.std(naive_arr, ddof=1)), 4),
            "mean_hybrid": round(float(np.mean(hybrid_arr)), 4),
            "std_hybrid": round(float(np.std(hybrid_arr, ddof=1)), 4),
            "mean_delta": round(mean_delta, 4),
            "std_delta": round(std_delta, 4),
            "median_delta": round(float(np.median(deltas)), 4),
            "test_name": test_name,
            "test_stat": round(test_stat, 4),
            "p_value": round(p_val, 5),
            "p_value_adjusted": round(p_val, 5),
            "is_significant": bool(p_val < self.alpha),
            "effect_size_name": "Rank-Biserial r",
            "effect_size_val": round(float(effect_size_val), 4),
            "ci_95": (round(float(ci_lower), 4), round(float(ci_upper), 4)),
            "alternative": "two-sided",
            "direction": direction,
            "diagnostic": False,
            "variance_note": variance_note,
        }
        self._interpret(result)
        return result

    @staticmethod
    def _interpret(d: Dict[str, Any]) -> None:
        """Sets `better_arm` and `improved` from the pre-declared direction and the significance decision."""
        signed = d["mean_delta"] if d["direction"] == "higher" else -d["mean_delta"]
        if not d["is_significant"] or d["variance_note"] or signed == 0:
            d["better_arm"] = "tie"
        else:
            d["better_arm"] = "hybrid" if signed > 0 else "naive"
        d["improved"] = d["better_arm"] == "hybrid"

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
        for key, _ in METRICS:
            d = overall[_REPORT_KEYS.get(key, key)]
            if key in DIAGNOSTIC_METRICS and "error" not in d:
                d["diagnostic"] = True
                d["diagnostic_note"] = _DIAGNOSTIC_NOTE

        # Holm-Bonferroni across the outcome family (diagnostics excluded); significance uses the adjusted p.
        family = [
            overall[_REPORT_KEYS.get(key, key)] for key, _ in METRICS
            if key not in DIAGNOSTIC_METRICS and "error" not in overall[_REPORT_KEYS.get(key, key)]
        ]
        for d, adj in zip(family, holm_adjust([d["p_value"] for d in family])):
            d["p_value_adjusted"] = round(adj, 5)
            d["is_significant"] = bool(adj < self.alpha)
            self._interpret(d)
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
            # Significant (Holm-adjusted) and in the pre-declared favourable direction for hybrid.
            return bool(d) and bool(d.get("improved"))

        def fmt(d) -> str:
            return (
                f"Δ = {d['mean_delta']:+.3f}, p = {d['p_value']:.4f}, p Holm = {d['p_value_adjusted']:.4f}, "
                f"{d['effect_size_name']} = {d['effect_size_val']:.3f}"
            )

        rel, prec, faith = (usable(k) for k in ("answer_relevance", "context_precision", "faithfulness"))
        if rel is None:
            return [f"- **Potencia insuficiente:** con N = {n} (< 3 pares) no es posible aplicar contrastes de hipótesis; amplíe la muestra."]

        out = []
        if wins(rel):
            out.append(
                "- **Superioridad de la recuperación híbrida:** la recuperación híbrida supera al control puramente vectorial "
                f"con significancia estadística (corrección de Holm) en Answer Relevance ({fmt(rel)})."
            )
        else:
            if rel["variance_note"]:
                reason = "las diferencias pareadas son nulas (varianza cero), de modo que el contraste no es informativo"
            elif rel["better_arm"] == "naive":
                reason = f"el control vectorial obtiene mejor Answer Relevance de forma significativa ({fmt(rel)})"
            else:
                reason = f"no se alcanzó significancia tras la corrección de Holm (Answer Relevance: {fmt(rel)})"
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
                    f"n = {hi_d['count']}) y el menor en *{lo}* ({lo_d['relevance_delta']:+.3f}, n = {lo_d['count']}). "
                    "Desglose descriptivo sin contraste por estrato: no permite inferir diferencias entre tipos de consulta."
                )

        if prec:
            out.append(
                "- **Precisión contextual:** "
                + ("sin varianza entre condiciones (ambos recuperadores sitúan el documento gold en las mismas posiciones)."
                   if prec["variance_note"] else
                   ("mejora significativa" if wins(prec)
                    else "empeora significativamente" if prec["better_arm"] == "naive"
                    else "diferencia no significativa") + f" ({fmt(prec)}).")
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
            "Contrastes bilaterales de Wilcoxon (rangos con signo); $p$ ajustado por Holm-Bonferroni sobre la familia de "
            "metricas de resultado. La direccion favorable de cada metrica esta pre-declarada (latencia: menor es mejor).\n",
            "| Metrica Evaluada | Control (Naive) | Tratamiento (Hybrid) | Diferencia Media ($\\Delta$) | Test Aplicado | $p$-valor (bilateral) | $p$ Holm | Tamano Efecto | IC 95% ($\\Delta$) | Significativo (Holm)? | Mejor Brazo |",
            "|---|---|---|---|---|---|---|---|---|---|---|"
        ]

        metrics_list = [
            ("faithfulness", "Faithfulness (RAGAS)"),
            ("answer_relevance", "Answer Relevance (RAGAS)"),
            ("context_precision", "Context Precision (RAGAS)"),
            ("latency_ms", "Latencia Respuesta (ms)"),
            ("rerank_score", "Cross-Encoder Top-1 (Semantic) [diagnostico]"),
        ]

        for m_key, m_name in metrics_list:
            if m_key not in ov or "error" in ov[m_key]:
                continue
            data = ov[m_key]
            if data.get("variance_note"):
                sig_str = "NO (sin varianza)"
                test_col = "N/A (Sin varianza)"
                p_val_str = p_adj_str = "1.0000"
            else:
                test_col = data['test_name'].split(' (')[0]
                p_val_str = f"{data['p_value']:.4f}"
                if data.get("diagnostic"):
                    sig_str, p_adj_str = "N/A (diagnostico)", "N/A"
                else:
                    sig_str = "**SI (p Holm < 0.05)**" if data['is_significant'] else "NO (p Holm >= 0.05)"
                    p_adj_str = f"**{data['p_value_adjusted']:.4f}**" if data['is_significant'] else f"{data['p_value_adjusted']:.4f}"
            lines.append(
                f"| **{m_name}** | {data['mean_naive']:.3f} $\\pm$ {data['std_naive']:.3f} | "
                f"{data['mean_hybrid']:.3f} $\\pm$ {data['std_hybrid']:.3f} | "
                f"**{data['mean_delta']:+.3f}** | {test_col} | "
                f"{p_val_str} | {p_adj_str} | {data['effect_size_name']}: {data['effect_size_val']:.3f} | "
                f"[{data['ci_95'][0]:.3f}, {data['ci_95'][1]:.3f}] | {sig_str} | "
                f"{'N/A' if data.get('diagnostic') else data.get('better_arm', 'tie')} |"
            )

        diag = ov.get("rerank_score", {})
        if diag.get("diagnostic"):
            lines.append(
                "\n> **Metrica diagnostica:** Cross-Encoder Top-1 se reporta solo como diagnostico y queda fuera de la "
                "familia de Holm y de las conclusiones: el pool de candidatos hibrido contiene al pool naive, por lo que "
                "favorece al brazo hibrido por construccion.\n"
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
