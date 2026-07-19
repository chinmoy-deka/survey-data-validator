from __future__ import annotations

import re
from collections import Counter, defaultdict

from core.models import (
    QuestionFamily,
    SurveyProject,
    TableValidationResult,
)
from core.validation_engine import (
    TabulationValidationEngine,
    ValidationEngineError,
)


class ValidationService:
    """Coordinates validation for the current survey project."""

    GRID_ROW_PATTERN = re.compile(
        r"^(?P<qid>.+?)_r(?P<row>\d+)$",
        re.IGNORECASE,
    )

    @staticmethod
    def _get_raw_dataframe(project: SurveyProject):
        """Return the raw-data DataFrame across model versions."""
        return getattr(
            project,
            "raw_dataframe",
            getattr(project, "raw_data", None),
        )

    def validate_all(
        self,
        project: SurveyProject,
        percentage_tolerance: float = 0.5,
        respondent_tolerance: int = 0,
    ) -> list[TableValidationResult]:
        """Validate every imported tabulation table."""
        raw_dataframe = self._get_raw_dataframe(project)
        self._validate_project_state(project, raw_dataframe)

        engine = TabulationValidationEngine(
            percentage_tolerance=percentage_tolerance,
            respondent_tolerance=respondent_tolerance,
        )

        family_index = self._build_family_index(
            project.question_families
        )

        tables = project.tabulation_workbook.tables

        table_id_counts = Counter(
            self._normalize_question_id(table.question_id)
            for table in tables
            if self._normalize_question_id(table.question_id)
        )

        occurrence_by_id: dict[str, int] = defaultdict(int)
        results: list[TableValidationResult] = []

        for table in tables:
            normalized_id = self._normalize_question_id(
                table.question_id
            )
            occurrence_index = occurrence_by_id[normalized_id]
            occurrence_by_id[normalized_id] += 1

            family = self._find_matching_family(
                question_id=table.question_id,
                family_index=family_index,
                occurrence_index=occurrence_index,
                total_occurrences=table_id_counts.get(
                    normalized_id,
                    1,
                ),
            )

            result = engine.validate_table(
                dataframe=raw_dataframe,
                table=table,
                family=family,
            )

            results.append(result)

        project.validation_results = results
        return results

    def validate_one(
        self,
        project: SurveyProject,
        table_number: int,
        percentage_tolerance: float = 0.5,
        respondent_tolerance: int = 0,
    ) -> TableValidationResult:
        """Validate one imported tabulation table."""
        raw_dataframe = self._get_raw_dataframe(project)
        self._validate_project_state(project, raw_dataframe)

        tables = project.tabulation_workbook.tables
        table_position = next(
            (
                index
                for index, item in enumerate(tables)
                if item.table_number == table_number
            ),
            None,
        )

        if table_position is None:
            raise ValidationEngineError(
                f"Table {table_number} was not found."
            )

        table = tables[table_position]
        normalized_id = self._normalize_question_id(
            table.question_id
        )

        matching_positions = [
            index
            for index, item in enumerate(tables)
            if self._normalize_question_id(item.question_id)
            == normalized_id
        ]

        occurrence_index = matching_positions.index(
            table_position
        )
        total_occurrences = len(matching_positions)

        family_index = self._build_family_index(
            project.question_families
        )

        family = self._find_matching_family(
            question_id=table.question_id,
            family_index=family_index,
            occurrence_index=occurrence_index,
            total_occurrences=total_occurrences,
        )

        engine = TabulationValidationEngine(
            percentage_tolerance=percentage_tolerance,
            respondent_tolerance=respondent_tolerance,
        )

        return engine.validate_table(
            dataframe=raw_dataframe,
            table=table,
            family=family,
        )

    @staticmethod
    def _validate_project_state(
        project: SurveyProject,
        raw_dataframe,
    ) -> None:
        if raw_dataframe is None:
            raise ValidationEngineError(
                "Upload the raw-data workbook first."
            )

        if project.tabulation_workbook is None:
            raise ValidationEngineError(
                "Import the tabulation workbook first."
            )

        if not project.question_families:
            raise ValidationEngineError(
                "No raw-data question families are available."
            )

    def _build_family_index(
        self,
        families: list[QuestionFamily],
    ) -> dict[str, object]:
        """
        Build two indexes:

        exact:
            Final refined family name -> family or families.

        grid_rows:
            Base question ID -> ordered QID_r1, QID_r2, ... families.
        """
        exact: dict[str, list[QuestionFamily]] = {}
        grid_rows: dict[str, list[tuple[int, QuestionFamily]]] = {}

        for family in families:
            family_name = self._get_family_name(family)
            normalized_name = self._normalize_question_id(
                family_name
            )

            if normalized_name:
                exact.setdefault(normalized_name, []).append(
                    family
                )

            grid_details = self._get_grid_row_details(
                family
            )

            if grid_details is None:
                continue

            base_id, row_number = grid_details
            normalized_base_id = self._normalize_question_id(
                base_id
            )

            if not normalized_base_id:
                continue

            grid_rows.setdefault(
                normalized_base_id,
                [],
            ).append((row_number, family))

        ordered_grid_rows: dict[
            str,
            list[QuestionFamily],
        ] = {}

        for base_id, row_families in grid_rows.items():
            row_families.sort(key=lambda item: item[0])
            ordered_grid_rows[base_id] = [
                family
                for _, family in row_families
            ]

        return {
            "exact": exact,
            "grid_rows": ordered_grid_rows,
        }

    def _find_matching_family(
        self,
        question_id: str,
        family_index: dict[str, object],
        occurrence_index: int = 0,
        total_occurrences: int = 1,
    ) -> QuestionFamily | None:
        """
        Match a tabulation table to one refined family.

        Rules:
        1. A table ID such as QID_r2 matches QID_r2 exactly.
        2. Repeated tables carrying only QID map sequentially to
           QID_r1, QID_r2, ... in workbook order.
        3. A non-grid question uses an exact normalized match.
        4. Ambiguous or out-of-range matches remain unmatched.
        """
        normalized_question_id = self._normalize_question_id(
            question_id
        )

        if not normalized_question_id:
            return None

        exact_index = family_index.get("exact", {})
        grid_index = family_index.get("grid_rows", {})

        exact_matches = exact_index.get(
            normalized_question_id,
            [],
        )

        question_is_explicit_grid_row = bool(
            re.search(
                r"R\d+$",
                normalized_question_id,
                re.IGNORECASE,
            )
        )

        # An explicit QID_rN reference should always use its exact row.
        if question_is_explicit_grid_row:
            if len(exact_matches) == 1:
                return exact_matches[0]
            return None

        row_candidates = grid_index.get(
            normalized_question_id,
            [],
        )

        # Repeated base IDs represent separate grid rows in table order.
        if total_occurrences > 1 and row_candidates:
            if occurrence_index < len(row_candidates):
                return row_candidates[occurrence_index]
            return None

        # A single base-ID table may map to the only available row.
        if len(row_candidates) == 1 and not exact_matches:
            return row_candidates[0]

        if len(exact_matches) == 1:
            return exact_matches[0]

        return None

    def _get_grid_row_details(
        self,
        family: QuestionFamily,
    ) -> tuple[str, int] | None:
        """Return the base QID and row number for a refined grid row."""
        family_name = self._get_family_name(family)
        match = self.GRID_ROW_PATTERN.match(family_name)

        if match is None:
            return None

        structural_type = str(
            getattr(family, "structural_type", "") or ""
        ).strip().lower()

        if structural_type not in {
            "single_select_grid_row",
            "multi_select_grid_row",
        }:
            return None

        return (
            match.group("qid"),
            int(match.group("row")),
        )

    @staticmethod
    def _get_family_name(family: QuestionFamily) -> str:
        for attribute_name in (
            "name",
            "family_name",
            "question_id",
        ):
            value = getattr(family, attribute_name, None)
            if value:
                return str(value)

        return ""

    @staticmethod
    def _normalize_question_id(value: str) -> str:
        """Normalize separators while preserving letters and numbers."""
        text = str(value or "").strip().upper()
        return re.sub(r"[^A-Z0-9]", "", text)
