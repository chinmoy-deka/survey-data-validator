from __future__ import annotations

import copy
import re
from collections import defaultdict

from core.models import (
    ColumnInfo,
    QuestionFamily,
    QuestionType,
)


class FamilyRefiner:
    """
    Applies Stage-2 grouping based on the naming conventions
    supplied by the quality analyst.

    Stage 1 remains responsible for broad generic grouping.
    Stage 2 refines only clearly recognised structures.
    """

    MULTI_SELECT_GRID_PATTERN = re.compile(
        r"^(?P<qid>.+?)_r(?P<row>\d+)_c(?P<column>\d+)$",
        re.IGNORECASE,
    )

    SINGLE_SELECT_GRID_PATTERN = re.compile(
        r"^(?P<qid>.+?)_r(?P<row>\d+)$",
        re.IGNORECASE,
    )

    MULTI_SELECT_PATTERN = re.compile(
        r"^(?P<qid>.+?)_(?P<option>\d+)$",
        re.IGNORECASE,
    )

    def refine(
        self,
        families: list[QuestionFamily],
    ) -> list[QuestionFamily]:
        """
        Refine all Stage-1 families.

        Multi-select-grid columns may have been left in separate
        Stage-1 families, so all physical columns are inspected
        together before the remaining families are refined.
        """

        all_columns = [
            column
            for family in families
            for column in family.columns
        ]

        grid_column_names = {
            column.name
            for column in all_columns
            if self.MULTI_SELECT_GRID_PATTERN.match(
                column.name
            )
        }

        refined_families: list[QuestionFamily] = []

        # First create multi-select-grid row families globally.
        refined_families.extend(
            self._build_multi_select_grid_families(
                all_columns
            )
        )

        # Then refine all remaining Stage-1 families.
        for family in families:
            remaining_columns = [
                column
                for column in family.columns
                if column.name not in grid_column_names
            ]

            if not remaining_columns:
                continue

            temporary_family = copy.deepcopy(family)
            temporary_family.columns = remaining_columns

            refined_families.extend(
                self._refine_standard_family(
                    temporary_family
                )
            )

        return sorted(
            refined_families,
            key=lambda family: self._natural_sort_key(
                family.name
            ),
        )

    def _refine_standard_family(
        self,
        family: QuestionFamily,
    ) -> list[QuestionFamily]:
        columns = family.columns

        if not columns:
            return []

        if self._is_single_select_grid(columns):
            return [
                self._create_family(
                    source_family=family,
                    name=column.name,
                    columns=[column],
                    structural_type=(
                        "single_select_grid_row"
                    ),
                    detected_type=(
                        QuestionType.SINGLE_SELECT
                    ),
                )
                for column in self._sort_grid_rows(
                    columns
                )
            ]

        if self._is_normal_multi_select(columns):
            sorted_columns = (
                self._sort_multi_select_options(
                    columns
                )
            )

            qid = self._extract_multi_select_qid(
                sorted_columns[0].name
            )

            return [
                self._create_family(
                    source_family=family,
                    name=qid,
                    columns=sorted_columns,
                    structural_type="multi_select",
                    detected_type=(
                        QuestionType.MULTI_SELECT
                    ),
                )
            ]

        if len(columns) == 1:
            column = columns[0]

            return [
                self._create_family(
                    source_family=family,
                    name=column.name,
                    columns=[column],
                    structural_type="single_select",
                    detected_type=(
                        QuestionType.SINGLE_SELECT
                    ),
                )
            ]

        # Unrecognised structures remain as Stage-1 groups.
        retained = copy.deepcopy(family)
        retained.grouping_method = "generic"
        retained.structural_type = "unresolved"
        retained.source_family_name = family.name

        return [retained]

    def _build_multi_select_grid_families(
        self,
        columns: list[ColumnInfo],
    ) -> list[QuestionFamily]:
        """
        Group QID_rN_cN variables by QID_rN.

        Example:
            Q5_r1_c1
            Q5_r1_c2
            Q5_r2_c1

        becomes:
            Q5_r1 -> c1, c2
            Q5_r2 -> c1
        """

        grouped: dict[
            str,
            list[tuple[int, ColumnInfo]],
        ] = defaultdict(list)

        source_qids: dict[str, str] = {}

        for column in columns:
            match = (
                self.MULTI_SELECT_GRID_PATTERN.match(
                    column.name
                )
            )

            if match is None:
                continue

            qid = match.group("qid")
            row_number = int(match.group("row"))
            column_number = int(
                match.group("column")
            )

            row_family_name = (
                f"{qid}_r{row_number}"
            )

            grouped[row_family_name].append(
                (column_number, column)
            )

            source_qids[row_family_name] = qid

        refined_families: list[
            QuestionFamily
        ] = []

        for family_name, grouped_columns in (
            grouped.items()
        ):
            grouped_columns.sort(
                key=lambda item: item[0]
            )

            physical_columns = [
                copy.deepcopy(column)
                for _, column in grouped_columns
            ]

            for column in physical_columns:
                column.family = family_name

            refined_families.append(
                QuestionFamily(
                    name=family_name,
                    columns=physical_columns,
                    detected_type=(
                        QuestionType.MULTI_SELECT
                    ),
                    confidence=1.0,
                    reason=(
                        "Detected using analyst-defined "
                        "multi-select grid naming pattern."
                    ),
                    grouping_method="analyst_rule",
                    structural_type=(
                        "multi_select_grid_row"
                    ),
                    source_family_name=(
                        source_qids[family_name]
                    ),
                )
            )

        return refined_families

    def _create_family(
        self,
        source_family: QuestionFamily,
        name: str,
        columns: list[ColumnInfo],
        structural_type: str,
        detected_type: QuestionType,
    ) -> QuestionFamily:
        """
        Create a refined family while preserving ColumnInfo
        metadata and relevant Stage-1 information.
        """

        cloned_columns = copy.deepcopy(columns)

        for column in cloned_columns:
            column.family = name

        return QuestionFamily(
            name=name,
            columns=cloned_columns,
            detected_type=detected_type,
            confirmed_type=None,
            confidence=1.0,
            reason=(
                "Detected using analyst-defined naming "
                "convention."
            ),
            grouping_method="analyst_rule",
            structural_type=structural_type,
            source_family_name=source_family.name,
            metadata=copy.deepcopy(
                source_family.metadata
            ),
        )

    def _is_single_select_grid(
        self,
        columns: list[ColumnInfo],
    ) -> bool:
        if not columns:
            return False

        question_ids = set()

        for column in columns:
            match = (
                self.SINGLE_SELECT_GRID_PATTERN.match(
                    column.name
                )
            )

            if match is None:
                return False

            question_ids.add(
                match.group("qid").lower()
            )

        return len(question_ids) == 1

    def _is_normal_multi_select(
        self,
        columns: list[ColumnInfo],
    ) -> bool:
        if len(columns) < 2:
            return False

        question_ids = set()

        for column in columns:
            match = self.MULTI_SELECT_PATTERN.match(
                column.name
            )

            if match is None:
                return False

            question_ids.add(
                match.group("qid").lower()
            )

        return len(question_ids) == 1

    def _sort_grid_rows(
        self,
        columns: list[ColumnInfo],
    ) -> list[ColumnInfo]:
        return sorted(
            columns,
            key=lambda column: int(
                self.SINGLE_SELECT_GRID_PATTERN.match(
                    column.name
                ).group("row")
            ),
        )

    def _sort_multi_select_options(
        self,
        columns: list[ColumnInfo],
    ) -> list[ColumnInfo]:
        return sorted(
            columns,
            key=lambda column: int(
                self.MULTI_SELECT_PATTERN.match(
                    column.name
                ).group("option")
            ),
        )

    def _extract_multi_select_qid(
        self,
        column_name: str,
    ) -> str:
        match = self.MULTI_SELECT_PATTERN.match(
            column_name
        )

        if match is None:
            return column_name

        return match.group("qid")

    @staticmethod
    def _natural_sort_key(
        value: str,
    ) -> list[str | int]:
        parts = re.split(
            r"(\d+)",
            str(value).lower(),
        )

        return [
            int(part) if part.isdigit() else part
            for part in parts
        ]