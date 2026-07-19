from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class QuestionType(str, Enum):
    SYSTEM_METADATA = "System Metadata"
    SINGLE_SELECT = "Single Select"
    MULTI_SELECT = "Multi Select"
    MATRIX = "Matrix / Grid"
    RATING_SCALE = "Rating Scale"
    RANKING = "Ranking"
    PERCENTAGE_ALLOCATION = "Percentage Allocation"
    NUMERIC_ENTRY = "Numeric Entry"
    OPEN_ENDED = "Open Ended"
    BINARY_FLAG = "Binary Flag"
    EMPTY = "Empty / Unused"
    MANUAL_REVIEW = "Manual Review"


@dataclass
class ColumnInfo:
    """Represents one physical column in the raw dataset."""

    index: int
    name: str
    dtype: str
    family: str
    suffix: str | None = None


@dataclass
class QuestionFamily:
    """Represents one logical survey question."""

    name: str
    columns: list[ColumnInfo] = field(default_factory=list)

    detected_type: QuestionType = QuestionType.MANUAL_REVIEW
    confirmed_type: QuestionType | None = None

    confidence: float = 0.0
    reason: str = ""
    grouping_method: str = "generic"
    structural_type: str = "unresolved"
    source_family_name: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    @property
    def number_of_columns(self) -> int:
        return len(self.columns)

    @property
    def final_type(self) -> QuestionType:
        return self.confirmed_type or self.detected_type


@dataclass
class WorkbookMetadata:
    """Stores information about the uploaded workbook."""

    file_name: str
    file_size_bytes: int
    sheet_names: list[str]
    selected_sheet: str
    header_row: int




@dataclass
class TabulationOption:
    """One reported option within a tabulation table."""

    label: str
    reported_value: float | None
    source_row: int

    @property
    def reported_percentage(self) -> float | None:
        if self.reported_value is None:
            return None

        # The supplied workbook stores percentages as decimals:
        # 0.75 means 75%.
        if 0 <= self.reported_value <= 1:
            return self.reported_value * 100

        return self.reported_value


@dataclass
class TabulationTable:
    """One table extracted from the tabulation workbook."""

    table_number: int
    question_id: str
    table_title: str
    question_text: str

    total_respondents: int | None = None
    options: list[TabulationOption] = field(default_factory=list)

    sheet_name: str = ""
    start_row: int = 0
    end_row: int = 0

    base_label: str | None = None
    column_code: str | None = None

    warnings: list[str] = field(default_factory=list)

    @property
    def option_count(self) -> int:
        return len(self.options)


@dataclass
class TabulationWorkbook:
    """Parsed representation of a tabulation workbook."""

    file_name: str
    selected_sheet: str
    tables: list[TabulationTable] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def question_ids(self) -> list[str]:
        return sorted(
            {
                table.question_id
                for table in self.tables
                if table.question_id
            }
        )
    

class ValidationStatus(str, Enum):
    PASS = "Pass"
    FAIL = "Fail"
    WARNING = "Warning"
    UNMATCHED = "Unmatched"
    NOT_VALIDATED = "Not Validated"


@dataclass
class OptionValidationResult:
    """Validation result for one tabulated option."""

    option_label: str
    raw_variable: str | None

    reported_value: float | None
    reported_percentage: float | None

    calculated_count: int | None
    calculated_percentage: float | None

    difference: float | None
    status: ValidationStatus

    raw_value: str | int | float | None = None
    message: str = ""


@dataclass
class TableValidationResult:
    """Validation result for one tabulation table."""

    table_number: int
    question_id: str
    table_title: str

    matched_family_name: str | None
    question_type: str

    reported_respondents: int | None
    calculated_respondents: int | None
    respondent_difference: int | None
    respondent_status: ValidationStatus

    option_results: list[OptionValidationResult] = field(
        default_factory=list
    )

    overall_status: ValidationStatus = (
        ValidationStatus.NOT_VALIDATED
    )

    messages: list[str] = field(default_factory=list)

    @property
    def passed_options(self) -> int:
        return sum(
            result.status == ValidationStatus.PASS
            for result in self.option_results
        )

    @property
    def failed_options(self) -> int:
        return sum(
            result.status == ValidationStatus.FAIL
            for result in self.option_results
        )

    @property
    def warning_options(self) -> int:
        return sum(
            result.status == ValidationStatus.WARNING
            for result in self.option_results
        )

    @property
    def option_count(self) -> int:
        return len(self.option_results)

@dataclass
class SurveyProject:
    """Represents one survey validation project."""

    project_name: str = "Untitled Survey Project"

    workbook: WorkbookMetadata | None = None
    raw_dataframe: pd.DataFrame | None = None
    question_families: list[QuestionFamily] = field(default_factory=list)
    tabulation_workbook: TabulationWorkbook | None = None

    dictionary_saved: bool = False
    validation_rules: dict[str, Any] = field(default_factory=dict)
    validation_results: list[TableValidationResult] = field(
    default_factory=list
    )
    report_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def has_raw_data(self) -> bool:
        return (
            self.raw_dataframe is not None
            and not self.raw_dataframe.empty
        )

    @property
    def row_count(self) -> int:
        if self.raw_dataframe is None:
            return 0

        return len(self.raw_dataframe)

    @property
    def column_count(self) -> int:
        if self.raw_dataframe is None:
            return 0

        return len(self.raw_dataframe.columns)

    @property
    def question_family_count(self) -> int:
        return len(self.question_families)

    def clear_analysis(self) -> None:
        """Clear analysis generated from the current raw dataset."""

        self.question_families = []
        self.dictionary_saved = False
        self.validation_rules = {}
        self.validation_results = []
        self.tabulation_workbook = None