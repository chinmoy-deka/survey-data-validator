from __future__ import annotations

import pandas as pd

from core.column_grouper import ColumnGrouper
from core.family_refiner import FamilyRefiner
from core.models import QuestionFamily
from core.type_detector import QuestionTypeDetector


class RawDictionaryService:
    """Builds the raw-data question dictionary."""

    def __init__(self) -> None:
        self.grouper = ColumnGrouper()
        self.refiner = FamilyRefiner()
        self.detector = QuestionTypeDetector()

    def generate(
        self,
        dataframe: pd.DataFrame,
    ) -> list[QuestionFamily]:
        # Stage 1: Existing generic grouping.
        initial_families = self.grouper.group(
            dataframe
        )

        # Stage 2: Analyst-rule structural refinement.
        refined_families = self.refiner.refine(
            initial_families
        )

        detected_families: list[
            QuestionFamily
        ] = []

        for family in refined_families:
            detected_family = self.detector.detect(
                dataframe=dataframe,
                family=family,
            )

            detected_families.append(
                detected_family
            )

        return detected_families