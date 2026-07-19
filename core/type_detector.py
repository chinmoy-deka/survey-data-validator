from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.models import QuestionFamily, QuestionType


class QuestionTypeDetector:
    """Infers the likely survey question type for a question family."""

    def detect(
        self,
        dataframe: pd.DataFrame,
        family: QuestionFamily,
    ) -> QuestionFamily:
        family_dataframe = dataframe[
            family.column_names
        ].copy()

        if family.name == "SYSTEM":
            return self._set_result(
                family,
                QuestionType.SYSTEM_METADATA,
                1.0,
                (
                    "Columns beginning with 'sys_' were "
                    "grouped as system metadata."
                ),
            )

        if (
            family_dataframe.empty
            or family_dataframe.isna().all().all()
        ):
            return self._set_result(
                family,
                QuestionType.EMPTY,
                1.0,
                (
                    "All values in this question family "
                    "are blank."
                ),
            )

        # Stage 2 analyst-defined structures take priority
        # over generic value-based type detection.
        analyst_result = self._detect_analyst_structure(
            dataframe=family_dataframe,
            family=family,
        )

        if analyst_result is not None:
            return analyst_result

        # Retain generic matrix detection only for families
        # not resolved by the Stage-2 FamilyRefiner.
        if self._is_matrix_family(family):
            return self._set_result(
                family,
                QuestionType.MATRIX,
                0.95,
                (
                    "Column names use matrix row suffixes "
                    "such as _r1, _r2 and _r3."
                ),
            )

        if family.number_of_columns > 1:
            return self._detect_multi_column_type(
                dataframe=family_dataframe,
                family=family,
            )

        return self._detect_single_column_type(
            series=family_dataframe.iloc[:, 0],
            family=family,
        )

    def _detect_analyst_structure(
        self,
        dataframe: pd.DataFrame,
        family: QuestionFamily,
    ) -> QuestionFamily | None:
        """
        Apply structural question types produced by the
        Stage-2 FamilyRefiner.

        The naming structure determines the question type.
        Data values verify whether the expected raw-data
        format has been followed.
        """

        if family.grouping_method != "analyst_rule":
            return None

        structural_type = family.structural_type

        if structural_type in {
            "single_select",
            "single_select_grid_row",
        }:
            valid_format, message = (
                self._validate_single_select_format(
                    dataframe
                )
            )

            if valid_format:
                return self._set_result(
                    family,
                    QuestionType.SINGLE_SELECT,
                    1.0,
                    (
                        "Classified as Single Select using "
                        "the analyst-defined naming pattern. "
                        f"{message}"
                    ),
                )

            family.metadata[
                "structural_warning"
            ] = message

            return self._set_result(
                family,
                QuestionType.SINGLE_SELECT,
                0.85,
                (
                    "Classified as Single Select using the "
                    "analyst-defined naming pattern, but the "
                    f"data format requires review. {message}"
                ),
            )

        if structural_type in {
            "multi_select",
            "multi_select_grid_row",
        }:
            valid_format, message = (
                self._validate_multi_select_format(
                    dataframe
                )
            )

            if valid_format:
                return self._set_result(
                    family,
                    QuestionType.MULTI_SELECT,
                    1.0,
                    (
                        "Classified as Multi Select using "
                        "the analyst-defined naming pattern. "
                        f"{message}"
                    ),
                )

            family.metadata[
                "structural_warning"
            ] = message

            return self._set_result(
                family,
                QuestionType.MULTI_SELECT,
                0.85,
                (
                    "Classified as Multi Select using the "
                    "analyst-defined naming pattern, but the "
                    f"data format requires review. {message}"
                ),
            )

        return None

    @staticmethod
    def _validate_single_select_format(
        dataframe: pd.DataFrame,
    ) -> tuple[bool, str]:
        """
        Validate an analyst-defined single-select question.

        Expected structure:
        - exactly one physical column;
        - nonblank responses must be numeric integers;
        - valid response codes must begin at 1.
        """

        if dataframe.shape[1] != 1:
            return (
                False,
                (
                    "A single-select question should contain "
                    "exactly one raw-data column."
                ),
            )

        series = dataframe.iloc[:, 0]
        non_missing = series.dropna()

        if non_missing.empty:
            return (
                True,
                "The column currently contains no responses.",
            )

        non_missing = non_missing[
            non_missing.astype(str).str.strip() != ""
        ]

        if non_missing.empty:
            return (
                True,
                "The column currently contains no responses.",
            )

        numeric = pd.to_numeric(
            non_missing,
            errors="coerce",
        )

        invalid_numeric_count = int(
            numeric.isna().sum()
        )

        if invalid_numeric_count > 0:
            return (
                False,
                (
                    f"{invalid_numeric_count} response(s) "
                    "are not numeric codes."
                ),
            )

        non_integer_mask = ~np.isclose(
            numeric,
            np.round(numeric),
        )

        non_integer_count = int(
            non_integer_mask.sum()
        )

        if non_integer_count > 0:
            return (
                False,
                (
                    f"{non_integer_count} response(s) "
                    "are not integer codes."
                ),
            )

        integer_values = numeric.astype(int)

        invalid_range_count = int(
            (integer_values < 1).sum()
        )

        if invalid_range_count > 0:
            return (
                False,
                (
                    f"{invalid_range_count} response(s) "
                    "fall below the minimum allowed code 1."
                ),
            )

        minimum_code = int(
            integer_values.min()
        )
        maximum_code = int(
            integer_values.max()
        )

        return (
            True,
            (
                "Observed integer response codes range from "
                f"{minimum_code} to {maximum_code}."
            ),
        )

    @staticmethod
    def _validate_multi_select_format(
        dataframe: pd.DataFrame,
    ) -> tuple[bool, str]:
        """
        Validate an analyst-defined multi-select question.

        Expected values:
        - 0 = not selected;
        - 1 = selected;
        - blank = missing, where applicable.
        """

        invalid_details: list[str] = []

        for column_name in dataframe.columns:
            series = dataframe[column_name]

            non_missing = series.dropna()

            non_missing = non_missing[
                non_missing.astype(str).str.strip() != ""
            ]

            if non_missing.empty:
                continue

            numeric = pd.to_numeric(
                non_missing,
                errors="coerce",
            )

            non_numeric_count = int(
                numeric.isna().sum()
            )

            if non_numeric_count > 0:
                invalid_details.append(
                    (
                        f"{column_name}: "
                        f"{non_numeric_count} non-numeric "
                        "response(s)"
                    )
                )
                continue

            invalid_binary_count = int(
                (~numeric.isin([0, 1])).sum()
            )

            if invalid_binary_count > 0:
                invalid_details.append(
                    (
                        f"{column_name}: "
                        f"{invalid_binary_count} value(s) "
                        "outside 0/1"
                    )
                )

        if invalid_details:
            preview = "; ".join(
                invalid_details[:5]
            )

            if len(invalid_details) > 5:
                preview += (
                    f"; and {len(invalid_details) - 5} "
                    "more column issue(s)"
                )

            return (
                False,
                (
                    "The family contains values that do not "
                    f"follow the required 0/1 format: {preview}."
                ),
            )

        return (
            True,
            (
                "All populated responses follow the required "
                "0/1 multi-select format."
            ),
        )

    def _detect_multi_column_type(
        self,
        dataframe: pd.DataFrame,
        family: QuestionFamily,
    ) -> QuestionFamily:
        if self._contains_only_binary_values(dataframe):
            return self._set_result(
                family,
                QuestionType.MULTI_SELECT,
                0.98,
                (
                    "Multiple related columns contain only "
                    "binary values such as 0 and 1."
                ),
            )

        if self._rows_sum_to_approximately_100(dataframe):
            return self._set_result(
                family,
                QuestionType.PERCENTAGE_ALLOCATION,
                0.95,
                (
                    "Most completed rows total "
                    "approximately 100."
                ),
            )

        if self._looks_like_ranking(dataframe):
            return self._set_result(
                family,
                QuestionType.RANKING,
                0.90,
                (
                    "Rows contain unique sequential rank "
                    "values across related columns."
                ),
            )

        if self._all_columns_numeric(dataframe):
            return self._set_result(
                family,
                QuestionType.MANUAL_REVIEW,
                0.55,
                (
                    "Multiple numeric columns were detected, "
                    "but they do not clearly match binary, "
                    "ranking or percentage-allocation rules."
                ),
            )

        return self._set_result(
            family,
            QuestionType.MANUAL_REVIEW,
            0.40,
            (
                "The family contains multiple columns with "
                "mixed or unclear data patterns."
            ),
        )

    def _detect_single_column_type(
        self,
        series: pd.Series,
        family: QuestionFamily,
    ) -> QuestionFamily:
        non_missing = series.dropna()

        if non_missing.empty:
            return self._set_result(
                family,
                QuestionType.EMPTY,
                1.0,
                "The column contains no responses.",
            )

        if self._series_is_text(non_missing):
            unique_count = (
                non_missing.astype(str).nunique()
            )

            if unique_count > 20:
                return self._set_result(
                    family,
                    QuestionType.OPEN_ENDED,
                    0.90,
                    (
                        "The column contains text with many "
                        "distinct responses."
                    ),
                )

            return self._set_result(
                family,
                QuestionType.SINGLE_SELECT,
                0.65,
                (
                    "The column contains a limited number "
                    "of repeated text categories."
                ),
            )

        numeric_series = pd.to_numeric(
            non_missing,
            errors="coerce",
        )
        numeric_series = numeric_series.dropna()

        if numeric_series.empty:
            return self._set_result(
                family,
                QuestionType.OPEN_ENDED,
                0.75,
                (
                    "Responses are non-numeric and appear "
                    "to be free text."
                ),
            )

        unique_values = sorted(
            numeric_series.unique().tolist()
        )
        unique_count = len(unique_values)

        if set(unique_values).issubset({0, 1}):
            return self._set_result(
                family,
                QuestionType.BINARY_FLAG,
                0.95,
                "The column contains only 0 and 1.",
            )

        if self._looks_like_rating_scale(
            numeric_series
        ):
            return self._set_result(
                family,
                QuestionType.RATING_SCALE,
                0.80,
                (
                    "The column contains a compact ordered "
                    "numeric scale."
                ),
            )

        if unique_count <= 20:
            return self._set_result(
                family,
                QuestionType.SINGLE_SELECT,
                0.70,
                (
                    "The column contains a limited number "
                    "of coded numeric categories."
                ),
            )

        return self._set_result(
            family,
            QuestionType.NUMERIC_ENTRY,
            0.80,
            (
                "The column contains many distinct "
                "numeric values."
            ),
        )

    @staticmethod
    def _set_result(
        family: QuestionFamily,
        question_type: QuestionType,
        confidence: float,
        reason: str,
    ) -> QuestionFamily:
        family.detected_type = question_type
        family.confidence = confidence
        family.reason = reason

        return family

    @staticmethod
    def _is_matrix_family(
        family: QuestionFamily,
    ) -> bool:
        return any(
            column.suffix
            and column.suffix.lower().startswith("r")
            and column.suffix[1:].isdigit()
            for column in family.columns
        )

    @staticmethod
    def _contains_only_binary_values(
        dataframe: pd.DataFrame,
    ) -> bool:
        values = pd.to_numeric(
            dataframe.stack(future_stack=True),
            errors="coerce",
        ).dropna()

        if values.empty:
            return False

        return set(
            values.unique()
        ).issubset({0, 1})

    @staticmethod
    def _rows_sum_to_approximately_100(
        dataframe: pd.DataFrame,
    ) -> bool:
        numeric = dataframe.apply(
            pd.to_numeric,
            errors="coerce",
        )

        completed_rows = numeric.dropna(
            how="all"
        )

        if completed_rows.empty:
            return False

        row_sums = completed_rows.sum(
            axis=1,
            min_count=1,
        )

        valid_rows = row_sums.between(
            99,
            101,
        )

        return valid_rows.mean() >= 0.80

    @staticmethod
    def _looks_like_ranking(
        dataframe: pd.DataFrame,
    ) -> bool:
        numeric = dataframe.apply(
            pd.to_numeric,
            errors="coerce",
        )

        completed_rows = numeric.dropna(
            how="all"
        )

        if completed_rows.empty:
            return False

        valid_row_count = 0

        for _, row in completed_rows.iterrows():
            values = row.dropna().tolist()

            if len(values) < 2:
                continue

            if any(
                not float(value).is_integer()
                for value in values
            ):
                continue

            integer_values = [
                int(value)
                for value in values
            ]

            if (
                len(integer_values)
                != len(set(integer_values))
            ):
                continue

            minimum = min(integer_values)
            maximum = max(integer_values)

            if (
                minimum >= 1
                and maximum <= dataframe.shape[1]
            ):
                valid_row_count += 1

        return (
            valid_row_count
            / len(completed_rows)
            >= 0.80
        )

    @staticmethod
    def _all_columns_numeric(
        dataframe: pd.DataFrame,
    ) -> bool:
        for column in dataframe.columns:
            converted = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            original_non_missing = (
                dataframe[column].notna().sum()
            )

            converted_non_missing = (
                converted.notna().sum()
            )

            if original_non_missing == 0:
                continue

            if (
                converted_non_missing
                / original_non_missing
                < 0.90
            ):
                return False

        return True

    @staticmethod
    def _series_is_text(
        series: pd.Series,
    ) -> bool:
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        numeric_ratio = (
            numeric.notna().mean()
        )

        return numeric_ratio < 0.80

    @staticmethod
    def _looks_like_rating_scale(
        series: pd.Series,
    ) -> bool:
        unique_values = sorted(
            series.unique().tolist()
        )

        if (
            len(unique_values) < 3
            or len(unique_values) > 11
        ):
            return False

        if any(
            not math.isclose(
                value,
                round(value),
            )
            for value in unique_values
        ):
            return False

        integer_values = [
            int(round(value))
            for value in unique_values
        ]

        return (
            min(integer_values) >= 0
            and max(integer_values) <= 10
            and (
                max(integer_values)
                - min(integer_values)
            ) <= 10
        )