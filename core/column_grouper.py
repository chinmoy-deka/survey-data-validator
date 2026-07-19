import re

from core.models import ColumnInfo
from core.models import QuestionFamily


class ColumnGrouper:

    def __init__(self):

        self.families = {}

    def group(self, dataframe):

        self.families = {}

        for index, column in enumerate(dataframe.columns):

            family, suffix = self.extract_family(column)

            info = ColumnInfo(
                index=index,
                name=column,
                dtype=str(dataframe[column].dtype),
                family=family,
                suffix=suffix,
            )

            if family not in self.families:

                self.families[family] = QuestionFamily(name=family)

            self.families[family].columns.append(info)

        return list(self.families.values())

    def extract_family(self, column):

        column = str(column).strip()

        if column.lower().startswith("sys_"):

            return "SYSTEM", column

        matrix = re.match(r"^(.*?)_r(\d+)$", column, re.IGNORECASE)

        if matrix:

            return matrix.group(1), "r" + matrix.group(2)

        numbered = re.match(r"^(.*?)_(\d+)$", column)

        if numbered:

            return numbered.group(1), numbered.group(2)

        return column, None