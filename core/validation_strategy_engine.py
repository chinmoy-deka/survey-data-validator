from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from core.models import TabulationTable


class ValidationStrategy(str, Enum):
    DIRECT = "DIRECT"
    GRID_SINGLE_SELECT = "GRID_SINGLE_SELECT"
    GRID_MULTI_SELECT = "GRID_MULTI_SELECT"
    RANK_EXACT = "RANK_EXACT"
    RANK_CUMULATIVE = "RANK_CUMULATIVE"
    TOP_2_BOX = "TOP_2_BOX"
    BOTTOM_2_BOX = "BOTTOM_2_BOX"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class StrategyDecision:
    strategy: ValidationStrategy
    confidence: float
    reason: str
    target_rank: int | None = None


class ValidationStrategyEngine:
    """Choose a validation algorithm from tabulation and raw naming evidence."""

    RANK_PATTERN = re.compile(r"rank\s*[-_:]?\s*(\d+(?:\s*\+\s*\d+)*)", re.I)

    def classify_group(
        self,
        tables: list[TabulationTable],
        raw_columns: list[str],
    ) -> dict[int, StrategyDecision]:
        if not tables:
            return {}

        qid = self._normal_id(tables[0].question_id)
        decisions: dict[int, StrategyDecision] = {}

        # Explicit rank labels are stronger than repeated-ID/grid evidence.
        rank_matches = []
        for table in tables:
            text = f"{table.table_title} {table.question_text}"
            match = self.RANK_PATTERN.search(text)
            rank_matches.append(match)

        if all(match is not None for match in rank_matches):
            for table, match in zip(tables, rank_matches):
                parts = [int(x) for x in re.findall(r"\d+", match.group(1))]
                cumulative = len(parts) > 1
                target = max(parts)
                decisions[table.table_number] = StrategyDecision(
                    strategy=(
                        ValidationStrategy.RANK_CUMULATIVE
                        if cumulative
                        else ValidationStrategy.RANK_EXACT
                    ),
                    confidence=1.0,
                    reason=(
                        f"Explicit cumulative rank label detected ({match.group(0)})."
                        if cumulative
                        else f"Explicit exact rank label detected ({match.group(0)})."
                    ),
                    target_rank=target,
                )
            return decisions

        # Derived box summaries.
        for table in tables:
            title = str(table.table_title or "").upper()
            if "TOP 2 BOX" in title:
                decisions[table.table_number] = StrategyDecision(
                    ValidationStrategy.TOP_2_BOX,
                    1.0,
                    "The table title explicitly identifies a Top 2 Box summary.",
                )
            elif "BOTTOM 2 BOX" in title:
                decisions[table.table_number] = StrategyDecision(
                    ValidationStrategy.BOTTOM_2_BOX,
                    1.0,
                    "The table title explicitly identifies a Bottom 2 Box summary.",
                )

        if len(decisions) == len(tables):
            return decisions

        repeated = len(tables) > 1
        single_grid_pattern = re.compile(rf"^{re.escape(qid)}R(\d+)$", re.I)
        multi_grid_pattern = re.compile(rf"^{re.escape(qid)}R(\d+)C(\d+)$", re.I)
        has_single_grid = any(single_grid_pattern.match(self._normal_id(c)) for c in raw_columns)
        has_multi_grid = any(multi_grid_pattern.match(self._normal_id(c)) for c in raw_columns)

        if repeated and has_multi_grid:
            for table in tables:
                decisions.setdefault(
                    table.table_number,
                    StrategyDecision(
                        ValidationStrategy.GRID_MULTI_SELECT,
                        0.99,
                        "Repeated tables align with raw variables named QID_rN_cM.",
                    ),
                )
            return decisions

        if repeated and has_single_grid:
            for table in tables:
                decisions.setdefault(
                    table.table_number,
                    StrategyDecision(
                        ValidationStrategy.GRID_SINGLE_SELECT,
                        0.99,
                        "Repeated tables align with raw variables named QID_rN.",
                    ),
                )
            return decisions

        for table in tables:
            decisions.setdefault(
                table.table_number,
                StrategyDecision(
                    ValidationStrategy.DIRECT,
                    0.75,
                    "No specialised derived, ranking, or repeated-grid pattern was detected.",
                ),
            )
        return decisions

    @staticmethod
    def _normal_id(value: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
