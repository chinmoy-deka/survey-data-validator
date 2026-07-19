from __future__ import annotations

import pandas as pd

from core.column_grouper import ColumnGrouper
from core.models import QuestionFamily
from core.type_detector import QuestionTypeDetector


class RawDictionaryService:
    """Builds the raw-data question dictionary."""

    def __init__(self) -> None:
        self.grouper = ColumnGrouper()
        self.detector = QuestionTypeDetector()

    def generate(self, dataframe: pd.DataFrame) -> list[QuestionFamily]:
        families = self.grouper.group(dataframe)

        detected_families: list[QuestionFamily] = []

        for family in families:
            detected_family = self.detector.detect(
                dataframe=dataframe,
                family=family,
            )
            detected_families.append(detected_family)

        return detected_families