from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from core.models import (
    OptionValidationResult,
    QuestionFamily,
    TableValidationResult,
    TabulationTable,
    ValidationStatus,
)


class ValidationEngineError(Exception):
    """Raised when tabulation validation cannot be completed."""


class TabulationValidationEngine:
    """Compare parsed tabulation tables against uploaded raw data."""

    MULTI_SELECT_STRUCTURES = {
        "multi_select",
        "multi_select_grid_row",
    }
    SINGLE_SELECT_STRUCTURES = {
        "single_select",
        "single_select_grid_row",
    }
    MULTI_SELECT_TYPES = {
        "multi select",
        "multi-select",
        "multiple response",
        "multiple select",
    }
    SINGLE_SELECT_TYPES = {
        "single select",
        "single-select",
    }

    def __init__(
        self,
        percentage_tolerance: float = 0.5,
        respondent_tolerance: int = 0,
    ) -> None:
        self.percentage_tolerance = percentage_tolerance
        self.respondent_tolerance = respondent_tolerance

    def validate_table(
        self,
        dataframe: pd.DataFrame,
        table: TabulationTable,
        family: QuestionFamily | None,
    ) -> TableValidationResult:
        if dataframe is None or dataframe.empty:
            raise ValidationEngineError(
                "Raw-data worksheet is empty."
            )

        if family is None:
            return TableValidationResult(
                table_number=table.table_number,
                question_id=table.question_id,
                table_title=table.table_title,
                matched_family_name=None,
                question_type="Unmatched",
                reported_respondents=table.total_respondents,
                calculated_respondents=None,
                respondent_difference=None,
                respondent_status=ValidationStatus.UNMATCHED,
                overall_status=ValidationStatus.UNMATCHED,
                messages=[
                    "No exact matching raw-data question family was "
                    "found. Grid rows must match IDs such as QID_r1, "
                    "QID_r2, and so on."
                ],
            )

        columns = self._get_family_columns(family)
        existing_columns = [
            column
            for column in columns
            if column in dataframe.columns
        ]
        family_name = self._get_family_name(family)
        question_type = self._get_question_type(family)

        if not existing_columns:
            return TableValidationResult(
                table_number=table.table_number,
                question_id=table.question_id,
                table_title=table.table_title,
                matched_family_name=family_name,
                question_type=question_type,
                reported_respondents=table.total_respondents,
                calculated_respondents=None,
                respondent_difference=None,
                respondent_status=ValidationStatus.FAIL,
                overall_status=ValidationStatus.FAIL,
                messages=[
                    "The matching family was found, but none of its "
                    "variables exist in the raw data."
                ],
            )

        family_dataframe = dataframe[existing_columns].copy()
        calculated_respondents = self._calculate_respondent_base(
            family_dataframe
        )
        respondent_difference, respondent_status = (
            self._validate_respondent_count(
                reported=table.total_respondents,
                calculated=calculated_respondents,
            )
        )

        structural_type = self._normalize_key(
            getattr(family, "structural_type", "")
        )
        normalized_type = self._normalize_text(question_type)
        messages = list(getattr(table, "warnings", []) or [])

        if (
            structural_type in self.MULTI_SELECT_STRUCTURES
            or normalized_type in self.MULTI_SELECT_TYPES
        ):
            option_results, validation_messages = (
                self._validate_multi_select(
                    dataframe=family_dataframe,
                    table=table,
                    respondent_base=calculated_respondents,
                )
            )
        elif (
            structural_type in self.SINGLE_SELECT_STRUCTURES
            or normalized_type in self.SINGLE_SELECT_TYPES
        ):
            if len(existing_columns) != 1:
                option_results = []
                validation_messages = [
                    "A single-select family must contain exactly one "
                    "raw-data variable."
                ]
                respondent_status = ValidationStatus.FAIL
            else:
                option_results, validation_messages = (
                    self._validate_single_select(
                        series=family_dataframe[existing_columns[0]],
                        raw_variable=existing_columns[0],
                        table=table,
                        respondent_base=calculated_respondents,
                    )
                )
        else:
            option_results = []
            validation_messages = [
                "This question type is not automatically validated. "
                "Multiple columns were not assumed to be multi-select."
            ]

        messages.extend(validation_messages)
        overall_status = self._get_overall_status(
            respondent_status=respondent_status,
            option_results=option_results,
            has_unvalidated_type=not option_results
            and bool(validation_messages),
        )

        return TableValidationResult(
            table_number=table.table_number,
            question_id=table.question_id,
            table_title=table.table_title,
            matched_family_name=family_name,
            question_type=question_type,
            reported_respondents=table.total_respondents,
            calculated_respondents=calculated_respondents,
            respondent_difference=respondent_difference,
            respondent_status=respondent_status,
            option_results=option_results,
            overall_status=overall_status,
            messages=messages,
        )

    def _validate_multi_select(
        self,
        dataframe: pd.DataFrame,
        table: TabulationTable,
        respondent_base: int,
    ) -> tuple[list[OptionValidationResult], list[str]]:
        """
        Map raw variables to displayed options by numeric column order.

        Example: CQ_1 -> first option, CQ_2 -> second option.
        Only 0, 1, and blank values are valid.
        """
        messages = [
            "Multi-select variables were mapped to tabulation options "
            "by numeric column order."
        ]
        options = list(table.options)
        columns = self._sort_columns_by_option_number(
            list(dataframe.columns)
        )

        if len(options) != len(columns):
            messages.append(
                f"The tabulation contains {len(options)} options, "
                f"while the raw-data family contains {len(columns)} "
                "variables."
            )

        option_results: list[OptionValidationResult] = []
        pair_count = max(len(options), len(columns))

        for index in range(pair_count):
            option = options[index] if index < len(options) else None
            column = columns[index] if index < len(columns) else None

            if option is None and column is not None:
                selected_count, invalid_count = (
                    self._count_strict_binary_values(dataframe[column])
                )
                calculated_percentage = self._calculate_percentage(
                    selected_count,
                    respondent_base,
                )
                option_results.append(
                    OptionValidationResult(
                        option_label="[Missing tabulation option]",
                        raw_variable=column,
                        reported_value=None,
                        reported_percentage=None,
                        calculated_count=selected_count,
                        calculated_percentage=calculated_percentage,
                        difference=None,
                        status=ValidationStatus.UNMATCHED,
                        message=(
                            "Raw-data variable has no corresponding "
                            "tabulation option."
                            + (
                                f" It also contains {invalid_count} "
                                "invalid nonblank value(s)."
                                if invalid_count
                                else ""
                            )
                        ),
                    )
                )
                continue

            if option is not None and column is None:
                option_results.append(
                    OptionValidationResult(
                        option_label=option.label,
                        raw_variable=None,
                        reported_value=option.reported_value,
                        reported_percentage=option.reported_percentage,
                        calculated_count=None,
                        calculated_percentage=None,
                        difference=None,
                        status=ValidationStatus.UNMATCHED,
                        message=(
                            "No raw-data variable exists at this option "
                            "position."
                        ),
                    )
                )
                continue

            selected_count, invalid_count = (
                self._count_strict_binary_values(dataframe[column])
            )
            calculated_percentage = self._calculate_percentage(
                selected_count,
                respondent_base,
            )
            difference, status = self._compare_percentage(
                reported_percentage=option.reported_percentage,
                calculated_percentage=calculated_percentage,
            )

            message = (
                f"Mapped by position: {column} is option {index + 1}."
            )
            if invalid_count:
                status = ValidationStatus.FAIL
                message += (
                    f" {invalid_count} nonblank value(s) are outside "
                    "the permitted 0/1 format."
                )

            option_results.append(
                OptionValidationResult(
                    option_label=option.label,
                    raw_variable=column,
                    reported_value=option.reported_value,
                    reported_percentage=option.reported_percentage,
                    calculated_count=selected_count,
                    calculated_percentage=calculated_percentage,
                    difference=difference,
                    status=status,
                    message=message,
                )
            )

        return option_results, messages

    def _validate_single_select(
        self,
        series: pd.Series,
        raw_variable: str,
        table: TabulationTable,
        respondent_base: int,
    ) -> tuple[list[OptionValidationResult], list[str]]:
        """
        Map code 1 to the first displayed option, code 2 to the second,
        and so on. Valid populated codes are integer values from 1 to n.
        """
        options = list(table.options)
        number_of_options = len(options)
        messages = [
            "Single-select response codes were mapped by displayed "
            "option order: 1 = first option, 2 = second option, etc."
        ]

        valid_codes, invalid_values, invalid_count = (
            self._analyse_single_select_codes(
                series=series,
                maximum_code=number_of_options,
            )
        )
        frequencies = valid_codes.value_counts().to_dict()

        if invalid_count:
            preview = ", ".join(
                f"{value!r} ({count})"
                for value, count in list(invalid_values.items())[:5]
            )
            messages.append(
                f"{invalid_count} invalid single-select response(s) "
                f"were found. Invalid values: {preview}. Valid codes "
                f"are integers from 1 to {number_of_options}."
            )

        option_results: list[OptionValidationResult] = []
        for code, option in enumerate(options, start=1):
            calculated_count = int(frequencies.get(code, 0))
            calculated_percentage = self._calculate_percentage(
                calculated_count,
                respondent_base,
            )
            difference, status = self._compare_percentage(
                reported_percentage=option.reported_percentage,
                calculated_percentage=calculated_percentage,
            )

            message = (
                f"Mapped by code order: raw code {code} is displayed "
                f"option {code}."
            )
            if invalid_count:
                status = ValidationStatus.FAIL
                message += (
                    " The question contains invalid populated response "
                    "codes, so option validation cannot fully pass."
                )

            option_results.append(
                OptionValidationResult(
                    option_label=option.label,
                    raw_variable=raw_variable,
                    raw_value=code,
                    reported_value=option.reported_value,
                    reported_percentage=option.reported_percentage,
                    calculated_count=calculated_count,
                    calculated_percentage=calculated_percentage,
                    difference=difference,
                    status=status,
                    message=message,
                )
            )

        if not options:
            messages.append(
                "No displayed tabulation options were available, so the "
                "valid single-select range could not be established."
            )

        return option_results, messages

    def _analyse_single_select_codes(
        self,
        series: pd.Series,
        maximum_code: int,
    ) -> tuple[pd.Series, dict[str, int], int]:
        valid_codes: list[int] = []
        invalid_values: dict[str, int] = {}

        for value in series.tolist():
            if self._is_blank(value):
                continue

            numeric = self._to_finite_number(value)
            valid = (
                numeric is not None
                and float(numeric).is_integer()
                and 1 <= int(numeric) <= maximum_code
            )
            if valid:
                valid_codes.append(int(numeric))
            else:
                key = str(value)
                invalid_values[key] = invalid_values.get(key, 0) + 1

        invalid_total = sum(invalid_values.values())
        return (
            pd.Series(valid_codes, dtype="int64"),
            invalid_values,
            invalid_total,
        )

    def _calculate_respondent_base(
        self,
        dataframe: pd.DataFrame,
    ) -> int:
        nonblank_mask = pd.DataFrame(
            {
                column: self._nonblank_mask(dataframe[column])
                for column in dataframe.columns
            },
            index=dataframe.index,
        )
        return int(nonblank_mask.any(axis=1).sum())

    def _validate_respondent_count(
        self,
        reported: int | None,
        calculated: int,
    ) -> tuple[int | None, ValidationStatus]:
        if reported is None:
            return None, ValidationStatus.WARNING

        difference = calculated - reported
        if abs(difference) <= self.respondent_tolerance:
            return difference, ValidationStatus.PASS
        return difference, ValidationStatus.FAIL

    def _count_strict_binary_values(
        self,
        series: pd.Series,
    ) -> tuple[int, int]:
        selected_count = 0
        invalid_count = 0

        for value in series.tolist():
            if self._is_blank(value):
                continue

            numeric = self._to_finite_number(value)
            if numeric == 1:
                selected_count += 1
            elif numeric == 0:
                continue
            else:
                invalid_count += 1

        return selected_count, invalid_count

    def _compare_percentage(
        self,
        reported_percentage: float | None,
        calculated_percentage: float | None,
    ) -> tuple[float | None, ValidationStatus]:
        if (
            reported_percentage is None
            or calculated_percentage is None
        ):
            return None, ValidationStatus.WARNING

        difference = calculated_percentage - reported_percentage
        if abs(difference) <= self.percentage_tolerance:
            return difference, ValidationStatus.PASS
        return difference, ValidationStatus.FAIL

    @staticmethod
    def _calculate_percentage(
        numerator: int,
        denominator: int,
    ) -> float | None:
        # When both the response count and respondent base are zero,
        # treat the operational survey percentage as 0.00%. This lets
        # genuinely empty/unused questions validate against tabulations
        # that correctly report a zero base and zero for every option.
        if denominator == 0 and numerator == 0:
            return 0.0

        if denominator <= 0:
            return None

        return numerator / denominator * 100

    @staticmethod
    def _get_overall_status(
        respondent_status: ValidationStatus,
        option_results: list[OptionValidationResult],
        has_unvalidated_type: bool = False,
    ) -> ValidationStatus:
        statuses = [
            respondent_status,
            *[result.status for result in option_results],
        ]

        if ValidationStatus.FAIL in statuses:
            return ValidationStatus.FAIL
        if ValidationStatus.UNMATCHED in statuses:
            return ValidationStatus.WARNING
        if ValidationStatus.WARNING in statuses:
            return ValidationStatus.WARNING
        if has_unvalidated_type:
            return ValidationStatus.NOT_VALIDATED
        if statuses and all(
            status == ValidationStatus.PASS for status in statuses
        ):
            return ValidationStatus.PASS
        return ValidationStatus.NOT_VALIDATED

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
    def _get_family_columns(family: QuestionFamily) -> list[str]:
        for attribute_name in (
            "columns",
            "variables",
            "column_names",
        ):
            value = getattr(family, attribute_name, None)
            if not value:
                continue
            if isinstance(value, str):
                return [value]

            resolved_columns: list[str] = []
            for item in value:
                if isinstance(item, str):
                    resolved_columns.append(item)
                    continue
                column_name = getattr(
                    item,
                    "name",
                    getattr(item, "column_name", None),
                )
                if column_name:
                    resolved_columns.append(str(column_name))
            return resolved_columns
        return []

    @staticmethod
    def _get_question_type(family: QuestionFamily) -> str:
        resolved_type = (
            getattr(family, "confirmed_type", None)
            or getattr(family, "detected_type", None)
        )
        if resolved_type is None:
            return "Unknown"
        if hasattr(resolved_type, "value"):
            return str(resolved_type.value)
        return str(resolved_type)

    @staticmethod
    def _sort_columns_by_option_number(
        columns: list[str],
    ) -> list[str]:
        def sort_key(column: str):
            text = str(column)
            grid_match = re.search(
                r"_c(\d+)$", text, re.IGNORECASE
            )
            if grid_match:
                return (0, int(grid_match.group(1)), text.lower())

            option_match = re.search(r"_(\d+)$", text)
            if option_match:
                return (0, int(option_match.group(1)), text.lower())

            return (1, math.inf, text.lower())

        return sorted(columns, key=sort_key)

    def _nonblank_mask(self, series: pd.Series) -> pd.Series:
        return series.apply(lambda value: not self._is_blank(value))

    @staticmethod
    def _is_blank(value: Any) -> bool:
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass
        return isinstance(value, str) and not value.strip()

    @staticmethod
    def _to_finite_number(value: Any) -> float | None:
        if isinstance(value, (bool, np.bool_)):
            return float(int(value))
        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        text = re.sub(r"[_\-/]+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[\s\-]+", "_", text)
        return re.sub(r"_+", "_", text)
