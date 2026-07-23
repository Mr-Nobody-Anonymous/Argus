"""
Self-Referential Logic Mutation Engine

Sandboxed code mutation engine that dynamically writes, compiles, tests,
and hot-swaps small Python filter functions in memory.

Security:
  - AST validation rejects unsafe node types and dunder access
  - Isolated global scope with whitelisted builtins only
  - No signal-based timers (thread-safe duration tracking instead)
  - Zero-cost fallback to static pipeline on any error
"""

import ast
import math
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

from config.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class CodeVariant:
    """A single compiled heuristic filter variant."""
    variant_id: str
    code_string: str
    compiled_func: Optional[Callable]
    fitness_score: float = 0.0
    frame_count: int = 0
    error_count: int = 0
    is_active: bool = False
    created_at: float = field(default_factory=time.time)


class LogicMutator:
    """Sandboxed code mutation engine."""

    DEFAULT_FILTER_CODE = '''
def filter_detections(detections):
    """Default pass-through filter."""
    return detections
'''

    FORBIDDEN_AST_NODES = {
        ast.Import, ast.ImportFrom,
        ast.Attribute, ast.Lambda, ast.ClassDef,
        ast.With, ast.AsyncWith,
        ast.Try, ast.TryExcept, ast.TryFinally,
        ast.Global, ast.Nonlocal, ast.Delete,
        ast.Yield, ast.YieldFrom,
        ast.Await, ast.AsyncFunctionDef,
        ast.GeneratorExp,
    }

    def __init__(self):
        self.config = get_config()
        self.lm_config = self.config.logic_mutation
        self.enabled = self.lm_config.enabled
        self.max_code_lines = self.lm_config.max_code_line_length
        self.stability_threshold = self.lm_config.stability_threshold
        self.max_variants = self.lm_config.max_variants
        self.test_frame_count = self.lm_config.test_frame_count
        self.allowed_builtins = set(self.lm_config.allowed_builtin_overrides)
        self.allowed_imports = set(self.lm_config.allowed_imports)
        self.sandbox_timeout_ms = self.lm_config.sandbox_timeout_ms
        self._variants: Dict[str, CodeVariant] = {}
        self._active_variant_id: Optional[str] = None
        self._lock = threading.RLock()
        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._safe_globals = self._build_safe_globals()
        self._default_func: Callable = self._compile_default()
        logger.info(f"LogicMutator initialized (enabled={self.enabled})")

    async def start(self):
        if not self.enabled:
            return
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._stop_event.clear()
        self._loop_thread = threading.Thread(target=self._mutation_loop, daemon=True, name="logic-mutator")
        self._loop_thread.start()
        logger.info("LogicMutator started")

    async def stop(self):
        self._stop_event.set()
        if self._loop_thread:
            self._loop_thread.join(timeout=3.0)
            self._loop_thread = None
        logger.info("LogicMutator stopped")

    # ── Core API ──

    def compile_heuristic_filter(self, code_string: str) -> Callable:
        """Compile a code string into a callable Python function.
        Uses AST validation + isolated global scope.
        Raises ValueError on syntax/AST errors."""
        try:
            tree = ast.parse(code_string)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in filter code: {e}")

        self._validate_ast(tree)

        lines = code_string.strip().split('\n')
        if len(lines) > self.max_code_lines:
            raise ValueError(f"Filter code exceeds max line length ({len(lines)} > {self.max_code_lines})")

        try:
            compiled = compile(tree, '<logic_mutator>', 'exec')
            local_ns: Dict[str, Any] = {}
            exec(compiled, self._safe_globals, local_ns)
            if 'filter_detections' not in local_ns:
                raise ValueError("Compiled code must define 'filter_detections' function")
            filter_func = local_ns['filter_detections']
            if not callable(filter_func):
                raise ValueError("'filter_detections' is not callable")
            return filter_func
        except Exception as e:
            raise ValueError(f"Compilation error: {e}")

    def generate_mutation(self, base_code: str) -> str:
        """Generate a mutated version of a base filter function.
        Uses AST-based mutation: swap comparison operators,
        adjust numeric constants, swap logical operators."""
        try:
            tree = ast.parse(base_code)
        except SyntaxError:
            return base_code

        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and node.ops:
                for i, op in enumerate(node.ops):
                    if isinstance(op, ast.Lt): node.ops[i] = ast.LtE()
                    elif isinstance(op, ast.LtE): node.ops[i] = ast.Lt()
                    elif isinstance(op, ast.Gt): node.ops[i] = ast.GtE()
                    elif isinstance(op, ast.GtE): node.ops[i] = ast.Gt()
                    elif isinstance(op, ast.Eq): node.ops[i] = ast.NotEq()
                    elif isinstance(op, ast.NotEq): node.ops[i] = ast.Eq()

            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value > 0:
                    node.value = node.value + (node.value * 0.1)

            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And): node.op = ast.Or()
                elif isinstance(node.op, ast.Or): node.op = ast.And()

        try:
            return ast.unparse(tree)
        except Exception:
            return base_code

    def evaluate_variant(self, variant: CodeVariant, test_data: List) -> float:
        """Run a compiled variant against test data and return fitness score."""
        if variant.compiled_func is None:
            return 0.0
        try:
            start = time.time()
            result = variant.compiled_func(test_data)
            elapsed = (time.time() - start) * 1000
            speed_fitness = max(0.0, 1.0 - (elapsed / 50.0))
            if isinstance(result, list):
                quality_fitness = min(1.0, len(result) / max(len(test_data), 1))
            else:
                quality_fitness = 0.0
            return max(0.0, min(1.0, 0.5 * speed_fitness + 0.5 * quality_fitness))
        except Exception:
            variant.error_count += 1
            return 0.0

    def get_active_filter(self) -> Callable:
        """Get the current best compiled filter function."""
        with self._lock:
            if self._active_variant_id and self._active_variant_id in self._variants:
                variant = self._variants[self._active_variant_id]
                if variant.compiled_func is not None:
                    return variant.compiled_func
            return self._default_func

    def apply_filter(self, detections: List[Dict]) -> List[Dict]:
        """Run detections through the active compiled filter.
        Synchronous, side-effect free, thread-safe.
        Falls back to original detections on any error."""
        if not self.enabled or not detections:
            return detections
        try:
            filter_func = self.get_active_filter()
            result = filter_func(detections)
            if isinstance(result, list):
                return result
            return detections
        except Exception as e:
            logger.warning(f"Logic mutator filter error (sandboxed fallback): {e}")
            with self._lock:
                if self._active_variant_id:
                    variant = self._variants.get(self._active_variant_id)
                    if variant:
                        variant.error_count += 1
                        if variant.error_count > 3:
                            variant.is_active = False
                            self._active_variant_id = None
            return detections

    def reset_to_baseline(self):
        """Wipe all variants, reset to default filter. Called during recovery."""
        with self._lock:
            self._variants.clear()
            self._active_variant_id = None
            self._default_func = self._compile_default()
        logger.info("LogicMutator reset to baseline (all variants cleared)")

    def get_status(self) -> Dict:
        """Return full mutation engine status."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "active_variant": self._active_variant_id,
                "total_variants": len(self._variants),
                "max_variants": self.max_variants,
                "loop_active": self._loop_thread is not None and self._loop_thread.is_alive(),
                "variants": {
                    vid: {"fitness": round(v.fitness_score, 3), "errors": v.error_count,
                          "is_active": v.is_active, "frame_count": v.frame_count}
                    for vid, v in self._variants.items()
                },
            }

    # ── Internal ──

    def _build_safe_globals(self) -> Dict[str, Any]:
        """Build isolated global scope with whitelisted builtins only."""
        safe_builtins = {}
        for name in self.allowed_builtins:
            if hasattr(__builtins__, name):
                safe_builtins[name] = getattr(__builtins__, name)
            elif name in __builtins__:
                safe_builtins[name] = __builtins__[name]
        safe_builtins.update({
            "True": True, "False": False, "None": None,
            "float": float, "int": int, "str": str, "bool": bool,
            "list": list, "dict": dict, "tuple": tuple, "set": set,
            "range": range, "enumerate": enumerate, "zip": zip,
            "abs": abs, "min": min, "max": max, "round": round,
            "sum": sum, "len": len, "any": any, "all": all,
            "sorted": sorted, "filter": filter, "map": map,
        })
        safe_globals = {"__builtins__": safe_builtins}
        for module_name in self.allowed_imports:
            try:
                if module_name == "math":
                    safe_globals["math"] = math
            except Exception:
                logger.debug(f"Could not add import {module_name}")
        return safe_globals

    def _compile_default(self) -> Callable:
        """Compile the default baseline filter."""
        try:
            return self.compile_heuristic_filter(self.DEFAULT_FILTER_CODE)
        except Exception as e:
            logger.error(f"Could not compile default filter: {e}")
            return lambda detections: detections

    def _validate_ast(self, tree: ast.AST):
        """Validate AST — reject forbidden nodes and dunder access."""
        for node in ast.walk(tree):
            for forbidden_type in self.FORBIDDEN_AST_NODES:
                if isinstance(node, forbidden_type):
                    raise ValueError(f"Forbidden AST node type: {type(node).__name__}")
            if isinstance(node, ast.Name) and "__" in node.id:
                raise ValueError(f"Dunder access blocked in variable name: {node.id}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "__" in node.name:
                raise ValueError(f"Dunder access blocked in function name: {node.name}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
                raise ValueError(f"Dunder access blocked in string constant: {node.value}")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if "__" in func_name:
                        raise ValueError(f"Dunder access blocked in function call: {func_name}")
                    if func_name not in self.allowed_builtins and func_name != "filter_detections":
                        raise ValueError(f"Function call not in whitelist: {func_name}")
                else:
                    raise ValueError(f"Non-name function call not allowed: {type(node.func).__name__}")

    def _mutation_loop(self):
        """Background loop that generates, tests, and adopts code mutations."""
        while not self._stop_event.is_set():
            try:
                self._evaluate_and_adopt_mutations()
            except Exception as e:
                logger.warning(f"Logic mutation loop error: {e}")
            for _ in range(50):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

    def _evaluate_and_adopt_mutations(self):
        """Generate, test, and adopt new code mutations."""
        with self._lock:
            if len(self._variants) >= self.max_variants:
                self._prune_variants()
            if self._active_variant_id and self._active_variant_id in self._variants:
                base_code = self._variants[self._active_variant_id].code_string
            else:
                base_code = self.DEFAULT_FILTER_CODE
            mutated_code = self.generate_mutation(base_code)
            try:
                compiled_func = self.compile_heuristic_filter(mutated_code)
            except ValueError:
                return
            variant_id = f"mut_{int(time.time() * 1000)}"
            variant = CodeVariant(variant_id=variant_id, code_string=mutated_code, compiled_func=compiled_func)
            test_data = [
                {"class_name": "person", "confidence": 0.8, "bbox": [10, 10, 50, 50]},
                {"class_name": "car", "confidence": 0.9, "bbox": [100, 100, 200, 200]},
                {"class_name": "person", "confidence": 0.3, "bbox": [5, 5, 20, 20]},
            ]
            fitness = self.evaluate_variant(variant, test_data)
            variant.fitness_score = fitness
            variant.frame_count = len(test_data)
            active_fitness = 0.0
            if self._active_variant_id and self._active_variant_id in self._variants:
                active_fitness = self._variants[self._active_variant_id].fitness_score
            improvement = fitness - active_fitness
            if improvement > self.stability_threshold and fitness > 0.5:
                if self._active_variant_id and self._active_variant_id in self._variants:
                    self._variants[self._active_variant_id].is_active = False
                variant.is_active = True
                self._variants[variant_id] = variant
                self._active_variant_id = variant_id
                logger.info(f"Adopted new logic mutation (fitness={fitness:.3f}, improvement={improvement:.3f})")
            else:
                self._variants[variant_id] = variant

    def _prune_variants(self):
        """Prune lowest-fitness variants to stay under max_variants."""
        sorted_variants = sorted(self._variants.values(), key=lambda v: v.fitness_score, reverse=True)
        keep_ids = {v.variant_id for v in sorted_variants[:self.max_variants]}
        self._variants = {vid: v for vid, v in self._variants.items() if vid in keep_ids or v.is_active}


_logic_mutator: Optional[LogicMutator] = None


def get_logic_mutator() -> LogicMutator:
    global _logic_mutator
    if _logic_mutator is None:
        _logic_mutator = LogicMutator()
    return _logic_mutator