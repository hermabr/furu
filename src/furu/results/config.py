from __future__ import annotations

from dataclasses import dataclass

from .codecs import default_result_registry
from .markers import ResultRule, ResultSpec, merge_specs
from .paths import LogicalPath
from .registry import ResultRegistry


@dataclass(slots=True)
class ResultConfig:
    registry: ResultRegistry
    rules: tuple[ResultRule, ...] = ()

    def rule_for_path(self, logical_path: LogicalPath) -> ResultSpec | None:
        spec: ResultSpec | None = None
        for rule in self.rules:
            if rule.matches(logical_path):
                spec = merge_specs(spec, rule.spec)
        return spec


def default_result_config() -> ResultConfig:
    return ResultConfig(registry=default_result_registry())
