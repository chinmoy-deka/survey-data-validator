from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from core.models import (
    TabulationOption,
    TabulationTable,
)


class TabulationParserError(Exception):
    """Raised when a tabulation workbook cannot be parsed."""


class TabulationParser:
    """Parses survey tabulation tables from a worksheet."""

    TABLE_HEADER_PATTERN = re.compile(
        r"^\s*Table\s+(\d+)\s*---\s*"
        r"([^-]+?)\s*---\s*(.*?)\s*$",
        re.IGNORECASE,
    )

    TOTAL_RESPONDENTS_LABELS = {
        "total respondents",
        "total respondent",
        "respondents",
        "base",
        "total base",
    }

    FOOTNOTE_PREFIXES = (
        "the percentage in gray",
        "the percentages in gray",
        "note:",
        "source:",
    )

    def parse(
        self,
        dataframe: pd.DataFrame,
        sheet_name: str,
    ) -> list[TabulationTable]:
        if dataframe is None or dataframe.empty:
            raise TabulationParserError(
                "The selected tabulation worksheet is empty."
            )

        normalized_dataframe = dataframe.copy()
        normalized_dataframe = normalized_dataframe.reset_index(
            drop=True
        )

        table_start_rows = self._find_table_start_rows(
            normalized_dataframe
        )

        if not table_start_rows:
            raise TabulationParserError(
                "No tabulation tables were detected. "
                "Expected headings such as "
                "'Table 1 --- S1 --- Table Title'."
            )

        tables: list[TabulationTable] = []

        for position, start_index in enumerate(table_start_rows):
            if position + 1 < len(table_start_rows):
                end_index = table_start_rows[position + 1] - 1
            else:
                end_index = len(normalized_dataframe) - 1

            table = self._parse_table(
                dataframe=normalized_dataframe,
                start_index=start_index,
                end_index=end_index,
                sheet_name=sheet_name,
            )

            tables.append(table)

        return tables

    def _find_table_start_rows(
        self,
        dataframe: pd.DataFrame,
    ) -> list[int]:
        first_column = dataframe.iloc[:, 0]

        start_rows: list[int] = []

        for row_index, value in first_column.items():
            text = self._clean_text(value)

            if not text:
                continue

            if self.TABLE_HEADER_PATTERN.match(text):
                start_rows.append(int(row_index))

        return start_rows

    def _parse_table(
        self,
        dataframe: pd.DataFrame,
        start_index: int,
        end_index: int,
        sheet_name: str,
    ) -> TabulationTable:
        header_text = self._clean_text(
            dataframe.iloc[start_index, 0]
        )

        header_match = self.TABLE_HEADER_PATTERN.match(
            header_text
        )

        if header_match is None:
            raise TabulationParserError(
                f"Invalid table heading at Excel row "
                f"{start_index + 1}."
            )

        table_number = int(header_match.group(1))
        question_id = header_match.group(2).strip()
        table_title = header_match.group(3).strip()

        question_text = self._extract_question_text(
            dataframe=dataframe,
            start_index=start_index,
            end_index=end_index,
        )

        total_row_index = self._find_total_respondents_row(
            dataframe=dataframe,
            start_index=start_index,
            end_index=end_index,
        )

        total_respondents: int | None = None
        options_start_index: int | None = None
        warnings: list[str] = []

        if total_row_index is not None:
            total_value = self._find_numeric_value_in_row(
                dataframe.iloc[total_row_index]
            )

            if total_value is not None:
                total_respondents = int(round(total_value))
            else:
                warnings.append(
                    "Total Respondents row was found, but "
                    "its value could not be read."
                )

            options_start_index = total_row_index + 1
        else:
            warnings.append(
                "Total Respondents row was not found."
            )

        base_label, column_code = self._extract_base_details(
            dataframe=dataframe,
            start_index=start_index,
            total_row_index=total_row_index,
        )

        options: list[TabulationOption] = []

        if options_start_index is not None:
            options = self._extract_options(
                dataframe=dataframe,
                start_index=options_start_index,
                end_index=end_index,
            )

        if not options:
            warnings.append(
                "No option rows were detected."
            )

        return TabulationTable(
            table_number=table_number,
            question_id=question_id,
            table_title=table_title,
            question_text=question_text,
            total_respondents=total_respondents,
            options=options,
            sheet_name=sheet_name,
            start_row=start_index + 1,
            end_row=end_index + 1,
            base_label=base_label,
            column_code=column_code,
            warnings=warnings,
        )

    def _extract_question_text(
        self,
        dataframe: pd.DataFrame,
        start_index: int,
        end_index: int,
    ) -> str:
        search_end = min(start_index + 5, end_index)

        for row_index in range(
            start_index + 1,
            search_end + 1,
        ):
            value = self._clean_text(
                dataframe.iloc[row_index, 0]
            )

            if not value:
                continue

            if value.lower() in self.TOTAL_RESPONDENTS_LABELS:
                continue

            if self.TABLE_HEADER_PATTERN.match(value):
                continue

            return value

        return ""

    def _find_total_respondents_row(
        self,
        dataframe: pd.DataFrame,
        start_index: int,
        end_index: int,
    ) -> int | None:
        for row_index in range(
            start_index + 1,
            end_index + 1,
        ):
            label = self._clean_text(
                dataframe.iloc[row_index, 0]
            ).lower()

            if label in self.TOTAL_RESPONDENTS_LABELS:
                return row_index

        return None

    def _extract_base_details(
        self,
        dataframe: pd.DataFrame,
        start_index: int,
        total_row_index: int | None,
    ) -> tuple[str | None, str | None]:
        if total_row_index is None:
            return None, None

        base_label: str | None = None
        column_code: str | None = None

        for row_index in range(
            start_index + 1,
            total_row_index,
        ):
            row = dataframe.iloc[row_index]

            values = [
                self._clean_text(value)
                for value in row.tolist()
            ]

            non_blank_values = [
                value
                for value in values
                if value
            ]

            if not non_blank_values:
                continue

            for value in non_blank_values:
                if value.lower() == "total":
                    base_label = value
                elif (
                    len(value) <= 4
                    and value.upper() == value
                    and value.lower() != "total"
                ):
                    column_code = value

        return base_label, column_code

    def _extract_options(
        self,
        dataframe: pd.DataFrame,
        start_index: int,
        end_index: int,
    ) -> list[TabulationOption]:
        options: list[TabulationOption] = []

        for row_index in range(
            start_index,
            end_index + 1,
        ):
            row = dataframe.iloc[row_index]

            label = self._clean_text(row.iloc[0])

            if not label:
                continue

            if self.TABLE_HEADER_PATTERN.match(label):
                break

            if self._is_footnote(label):
                continue

            if label.lower() in self.TOTAL_RESPONDENTS_LABELS:
                continue

            reported_value = self._find_numeric_value_in_row(
                row
            )

            if reported_value is None:
                continue

            options.append(
                TabulationOption(
                    label=label,
                    reported_value=reported_value,
                    source_row=row_index + 1,
                )
            )

        return options

    def _find_numeric_value_in_row(
        self,
        row: pd.Series,
    ) -> float | None:
        # Column A normally contains the option label.
        # Search remaining columns for the first usable number.
        values = row.iloc[1:].tolist()

        for value in values:
            numeric_value = self._to_number(value)

            if numeric_value is not None:
                return numeric_value

        return None

    def _is_footnote(self, text: str) -> bool:
        normalized_text = text.strip().lower()

        return normalized_text.startswith(
            self.FOOTNOTE_PREFIXES
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, float) and math.isnan(value):
            return ""

        return str(value).strip()

    @staticmethod
    def _to_number(value: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            numeric_value = float(value)

            if math.isnan(numeric_value):
                return None

            return numeric_value

        text = str(value).strip()

        if not text:
            return None

        text = text.replace(",", "")

        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100
            except ValueError:
                return None

        try:
            return float(text)
        except ValueError:
            return None