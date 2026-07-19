from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.models import QuestionType, TabulationTable


@dataclass
class TabulationClassification:
    question_id: str
    final_type: QuestionType | None
    confidence: float
    scores: dict[str, int] = field(default_factory=dict)
    reason: str = ""
    status: str = "unresolved"
    rank_numbers: list[int] = field(default_factory=list)


class TabulationClassificationEngine:
    """Classify a group of tabulation tables sharing one question ID."""

    RANK_PATTERN = re.compile(
        r"(?:^|\b)rank\s*[-_:]?\s*(\d+)\b",
        re.IGNORECASE,
    )
    GRID_WORDS = (
        "attribute",
        "statement",
        "brand",
        "supplier",
        "distributor",
        "company",
        "product",
        "service",
        "item",
    )

    def classify(
        self,
        question_id: str,
        tables: list[TabulationTable],
    ) -> TabulationClassification:
        scores = {
            "ranking": 0,
            "grid": 0,
            "single_select": 0,
            "multi_select": 0,
        }
        reasons: list[str] = []

        if not tables:
            return TabulationClassification(
                question_id=question_id,
                final_type=None,
                confidence=0.0,
                scores=scores,
                reason="No tabulation tables were supplied.",
            )

        repeated = len(tables) > 1
        if repeated:
            scores["ranking"] += 10
            scores["grid"] += 20
            reasons.append(
                f"The question ID is repeated across {len(tables)} tables."
            )
        else:
            scores["single_select"] += 20

        rank_numbers: list[int] = []
        all_have_rank = True
        for table in tables:
            searchable = " ".join(
                part
                for part in (
                    table.table_title,
                    table.question_text,
                )
                if part
            )
            match = self.RANK_PATTERN.search(searchable)
            if match is None:
                all_have_rank = False
            else:
                rank_numbers.append(int(match.group(1)))

        if rank_numbers:
            scores["ranking"] += 45
            reasons.append(
                "Rank labels were detected in the tabulation titles or question text."
            )

        if all_have_rank and len(rank_numbers) == len(tables):
            scores["ranking"] += 25
            reasons.append("Every repeated table contains an explicit rank number.")

        if rank_numbers and len(set(rank_numbers)) == len(rank_numbers):
            scores["ranking"] += 10
            reasons.append("The detected rank numbers are unique.")

        if rank_numbers:
            ordered = sorted(rank_numbers)
            expected = list(range(1, len(rank_numbers) + 1))
            if ordered == expected:
                scores["ranking"] += 10
                reasons.append(
                    "The rank numbers form a complete sequence beginning at 1."
                )

        titles = [self._normalize(table.table_title) for table in tables]
        distinct_titles = {title for title in titles if title}
        if repeated and len(distinct_titles) == len(tables):
            scores["grid"] += 15

        if repeated and not rank_numbers:
            scores["grid"] += 25
            reasons.append(
                "Repeated tables have no rank labels, which is more consistent with a grid."
            )

        joined_titles = " ".join(titles)
        if repeated and any(word in joined_titles for word in self.GRID_WORDS):
            scores["grid"] += 5

        ranked_scores = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )
        best_name, best_score = ranked_scores[0]
        second_score = ranked_scores[1][1]
        margin = best_score - second_score

        if best_score < 50 or margin < 15:
            return TabulationClassification(
                question_id=question_id,
                final_type=None,
                confidence=min(best_score / 100, 0.99),
                scores=scores,
                reason=" ".join(reasons)
                or "The tabulation evidence was insufficient for a reliable classification.",
                status="unresolved",
                rank_numbers=rank_numbers,
            )

        type_map = {
            "ranking": QuestionType.RANKING,
            "grid": QuestionType.MATRIX,
            "single_select": QuestionType.SINGLE_SELECT,
            "multi_select": QuestionType.MULTI_SELECT,
        }
        return TabulationClassification(
            question_id=question_id,
            final_type=type_map[best_name],
            confidence=min(best_score / 100, 1.0),
            scores=scores,
            reason=" ".join(reasons),
            status="resolved",
            rank_numbers=rank_numbers,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
