"""Evaluation package: RAGAS metrics computation, benchmark loaders, and inferential statistics."""
from .evaluator import RAGEvaluator
from .statistics import StatisticalAnalyzer
from .crag_loader import CRAGBenchmarkLoader

__all__ = ["RAGEvaluator", "StatisticalAnalyzer", "CRAGBenchmarkLoader"]
