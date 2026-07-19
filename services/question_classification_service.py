from __future__ import annotations

import re
from collections import defaultdict

from core.models import QuestionFamily, QuestionType, SurveyProject
from core.tabulation_classification_engine import (
    TabulationClassificationEngine,
)


class QuestionClassificationService:
    """Resolve raw-data classifications using imported tabulation evidence."""

    GRID_ROW_PATTERN = re.compile(
        r"^(?P<qid>.+?)_r(?P<row>\d+)$", re.IGNORECASE
    )

    def __init__(self) -> None:
        self.engine = TabulationClassificationEngine()

    def classify_project(self, project: SurveyProject) -> SurveyProject:
        if project.tabulation_workbook is None:
            return project

        table_groups = defaultdict(list)
        for table in project.tabulation_workbook.tables:
            normalized = self._normalize_id(table.question_id)
            if normalized:
                table_groups[normalized].append(table)

        exact_index: dict[str, list[QuestionFamily]] = defaultdict(list)
        grid_index: dict[str, list[QuestionFamily]] = defaultdict(list)

        for family in project.question_families:
            normalized_name = self._normalize_id(family.name)
            if normalized_name:
                exact_index[normalized_name].append(family)

            match = self.GRID_ROW_PATTERN.match(family.name)
            if match and family.structural_type in {
                "single_select_grid_row",
                "multi_select_grid_row",
            }:
                grid_index[self._normalize_id(match.group("qid"))].append(family)

            family.metadata.setdefault(
                "preliminary_type", family.detected_type.value
            )

        for normalized_id, tables in table_groups.items():
            classification = self.engine.classify(
                question_id=tables[0].question_id,
                tables=tables,
            )

            exact_families = exact_index.get(normalized_id, [])
            grid_families = grid_index.get(normalized_id, [])

            targets: list[QuestionFamily] = []
            if classification.final_type == QuestionType.RANKING:
                targets = exact_families
            elif classification.final_type == QuestionType.MATRIX:
                targets = grid_families or exact_families
            elif len(exact_families) == 1:
                targets = exact_families

            for family in targets:
                self._apply_classification(family, classification)

        return project

    @staticmethod
    def _apply_classification(family, classification) -> None:
        family.metadata["tabulation_type"] = (
            classification.final_type.value
            if classification.final_type is not None
            else None
        )
        family.metadata["classification_scores"] = classification.scores
        family.metadata["classification_confidence"] = classification.confidence
        family.metadata["classification_reason"] = classification.reason
        family.metadata["classification_status"] = classification.status
        family.metadata["rank_numbers"] = classification.rank_numbers

        if classification.final_type is not None:
            family.confirmed_type = classification.final_type
            family.confidence = classification.confidence
            family.reason = classification.reason

    @staticmethod
    def _normalize_id(value: str) -> str:
        text = str(value or "").strip().upper()
        return re.sub(r"[^A-Z0-9]", "", text)
