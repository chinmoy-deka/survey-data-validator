from __future__ import annotations

import pandas as pd

from core.models import SurveyProject, WorkbookMetadata
from services.raw_dictionary_service import RawDictionaryService


class SurveyProjectService:
    """Creates and updates survey projects."""

    def __init__(self) -> None:
        self.dictionary_service = RawDictionaryService()

    def load_raw_data(
        self,
        project: SurveyProject,
        dataframe: pd.DataFrame,
        file_name: str,
        file_size_bytes: int,
        sheet_names: list[str],
        selected_sheet: str,
        header_row: int,
    ) -> SurveyProject:
        if dataframe is None or dataframe.empty:
            raise ValueError("The raw dataset is empty.")

        project.raw_dataframe = dataframe

        project.workbook = WorkbookMetadata(
            file_name=file_name,
            file_size_bytes=file_size_bytes,
            sheet_names=sheet_names,
            selected_sheet=selected_sheet,
            header_row=header_row,
        )

        project.clear_analysis()

        project.question_families = self.dictionary_service.generate(
            dataframe
        )

        return project

    @staticmethod
    def clear_project(project: SurveyProject) -> SurveyProject:
        return SurveyProject(
            project_name=project.project_name
        )