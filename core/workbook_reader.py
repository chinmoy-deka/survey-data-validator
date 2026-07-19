from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd


class WorkbookReaderError(Exception):
    """Raised when an Excel workbook cannot be read."""


@dataclass(frozen=True)
class WorkbookInfo:
    file_name: str
    sheet_names: list[str]
    file_size_bytes: int | None = None


def _reset_file_pointer(file: BinaryIO) -> None:
    """Return an uploaded file to its starting position."""
    if hasattr(file, "seek"):
        file.seek(0)


def _get_excel_source(file: BinaryIO | str | Path) -> BinaryIO | str | Path:
    """
    Prepare an Excel source for pandas.

    Streamlit UploadedFile objects are converted to BytesIO so they can be
    read repeatedly when the user changes the selected worksheet.
    """
    if isinstance(file, (str, Path)):
        return file

    _reset_file_pointer(file)

    if hasattr(file, "getvalue"):
        return BytesIO(file.getvalue())

    return file


def get_workbook_info(
    file: BinaryIO | str | Path,
    file_name: str | None = None,
) -> WorkbookInfo:
    """Return basic workbook information without loading all worksheet data."""
    try:
        source = _get_excel_source(file)

        with pd.ExcelFile(source, engine="openpyxl") as excel_file:
            sheet_names = excel_file.sheet_names

        resolved_name = file_name

        if resolved_name is None:
            if isinstance(file, (str, Path)):
                resolved_name = Path(file).name
            else:
                resolved_name = getattr(file, "name", "Uploaded workbook")

        file_size = getattr(file, "size", None)

        return WorkbookInfo(
            file_name=resolved_name,
            sheet_names=sheet_names,
            file_size_bytes=file_size,
        )

    except FileNotFoundError as exc:
        raise WorkbookReaderError("The selected Excel file was not found.") from exc
    except ValueError as exc:
        raise WorkbookReaderError(
            "The uploaded file is not a valid Excel workbook."
        ) from exc
    except Exception as exc:
        raise WorkbookReaderError(
            f"Unable to open the Excel workbook: {exc}"
        ) from exc


def read_worksheet(
    file: BinaryIO | str | Path,
    sheet_name: str,
    header_row: int = 0,
) -> pd.DataFrame:
    """
    Read one worksheet into a pandas DataFrame.

    header_row=0 means the first Excel row is used as column headers.
    """
    if header_row < 0:
        raise WorkbookReaderError("Header row cannot be negative.")

    try:
        source = _get_excel_source(file)

        dataframe = pd.read_excel(
            source,
            sheet_name=sheet_name,
            header=header_row,
            engine="openpyxl",
        )

        dataframe.columns = [
            str(column).strip() if pd.notna(column) else ""
            for column in dataframe.columns
        ]

        return dataframe

    except ValueError as exc:
        raise WorkbookReaderError(
            f"Worksheet '{sheet_name}' could not be found or read."
        ) from exc
    except Exception as exc:
        raise WorkbookReaderError(
            f"Unable to read worksheet '{sheet_name}': {exc}"
        ) from exc


def get_duplicate_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return duplicated column names."""
    return dataframe.columns[dataframe.columns.duplicated()].tolist()


def get_unnamed_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return blank or automatically generated unnamed columns."""
    unnamed_columns: list[str] = []

    for column in dataframe.columns:
        cleaned_column = str(column).strip()

        if not cleaned_column or cleaned_column.lower().startswith("unnamed:"):
            unnamed_columns.append(str(column))

    return unnamed_columns