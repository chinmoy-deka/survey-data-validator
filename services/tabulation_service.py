from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import pandas as pd

from core.models import (
    SurveyProject,
    TabulationWorkbook,
)
from core.tabulation_parser import (
    TabulationParser,
    TabulationParserError,
)


class TabulationService:
    """Loads and parses a survey tabulation workbook."""

    def __init__(self) -> None:
        self.parser = TabulationParser()

    def get_sheet_names(
        self,
        file: BinaryIO,
    ) -> list[str]:
        excel_source = self._get_excel_source(file)

        try:
            workbook = pd.ExcelFile(
                excel_source,
                engine="openpyxl",
            )

            return workbook.sheet_names

        except Exception as error:
            raise TabulationParserError(
                f"Unable to read the tabulation workbook: "
                f"{error}"
            ) from error

    def import_tabulation(
        self,
        project: SurveyProject,
        file: BinaryIO,
        file_name: str,
        sheet_name: str,
    ) -> SurveyProject:
        excel_source = self._get_excel_source(file)

        try:
            # header=None is essential because the workbook does
            # not contain one conventional table header row.
            dataframe = pd.read_excel(
                excel_source,
                sheet_name=sheet_name,
                header=None,
                engine="openpyxl",
            )

        except Exception as error:
            raise TabulationParserError(
                f"Unable to read worksheet "
                f"'{sheet_name}': {error}"
            ) from error

        tables = self.parser.parse(
            dataframe=dataframe,
            sheet_name=sheet_name,
        )

        project.tabulation_workbook = TabulationWorkbook(
            file_name=file_name,
            selected_sheet=sheet_name,
            tables=tables,
        )

        project.validation_results = []

        return project

    @staticmethod
    def _get_excel_source(
        file: BinaryIO,
    ) -> BytesIO:
        try:
            file.seek(0)
        except (AttributeError, OSError):
            pass

        content = file.read()

        try:
            file.seek(0)
        except (AttributeError, OSError):
            pass

        return BytesIO(content)