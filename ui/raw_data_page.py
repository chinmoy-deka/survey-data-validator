from __future__ import annotations

import pandas as pd
import streamlit as st

from core.models import SurveyProject
from core.workbook_reader import (
    WorkbookReaderError,
    get_duplicate_columns,
    get_unnamed_columns,
    get_workbook_info,
    read_worksheet,
)
from services.project_service import SurveyProjectService


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"

    return f"{size_bytes / (1024**2):.1f} MB"


def _get_project() -> SurveyProject:
    project = st.session_state.get("project")

    if not isinstance(project, SurveyProject):
        project = SurveyProject()
        st.session_state["project"] = project

    return project


def _display_dataframe_summary(
    dataframe: pd.DataFrame,
) -> None:
    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Rows",
        f"{len(dataframe):,}",
    )

    metric_columns[1].metric(
        "Columns",
        f"{len(dataframe.columns):,}",
    )

    metric_columns[2].metric(
        "Empty Cells",
        f"{int(dataframe.isna().sum().sum()):,}",
    )

    metric_columns[3].metric(
        "Duplicate Rows",
        f"{int(dataframe.duplicated().sum()):,}",
    )


def _display_column_warnings(
    dataframe: pd.DataFrame,
) -> None:
    duplicate_columns = get_duplicate_columns(dataframe)
    unnamed_columns = get_unnamed_columns(dataframe)

    if duplicate_columns:
        st.warning(
            "Duplicate column names detected: "
            + ", ".join(duplicate_columns)
        )

    if unnamed_columns:
        st.warning(
            "Unnamed columns detected: "
            + ", ".join(unnamed_columns)
        )


def _display_column_list(
    dataframe: pd.DataFrame,
) -> None:
    with st.expander("View all columns"):
        rows = []

        for index, column in enumerate(dataframe.columns):
            series = dataframe[column]

            rows.append(
                {
                    "Column Number": index + 1,
                    "Column Name": str(column),
                    "Data Type": str(series.dtype),
                    "Non-Blank Values": int(
                        series.notna().sum()
                    ),
                    "Unique Values": int(
                        series.nunique(dropna=True)
                    ),
                }
            )

        column_table = pd.DataFrame(rows)

        st.dataframe(
            column_table,
            use_container_width=True,
            hide_index=True,
        )


def render_raw_data_page() -> None:
    st.title("Upload Raw Data")

    st.write(
        "Upload the survey raw-data workbook, select the "
        "worksheet and confirm that the dataset has loaded "
        "correctly."
    )

    project = _get_project()

    uploaded_file = st.file_uploader(
        "Upload Excel workbook",
        type=["xlsx", "xlsm"],
        key="raw_data_uploader",
    )

    if uploaded_file is None:
        if project.has_raw_data:
            st.success(
                "A dataset is already loaded. Open "
                "'Raw Data Dictionary' from the sidebar "
                "to review it."
            )
        else:
            st.info("Upload an Excel workbook to begin.")

        return

    try:
        workbook_info = get_workbook_info(uploaded_file)
    except WorkbookReaderError as error:
        st.error(str(error))
        return

    file_size = getattr(
        workbook_info,
        "file_size",
        getattr(workbook_info, "size_bytes", 0),
    )

    file_details = st.columns(3)

    file_details[0].metric(
        "File",
        workbook_info.file_name,
    )

    file_details[1].metric(
        "File Size",
        _format_file_size(file_size),
    )

    file_details[2].metric(
        "Worksheets",
        len(workbook_info.sheet_names),
    )

    selected_sheet = st.selectbox(
        "Select worksheet",
        options=workbook_info.sheet_names,
    )

    header_row_number = st.number_input(
        "Header row number",
        min_value=1,
        value=1,
        step=1,
        help="Enter the Excel row containing column names.",
    )

    if st.button(
        "Load Raw Data",
        type="primary",
        use_container_width=True,
    ):
        try:
            dataframe = read_worksheet(
                file=uploaded_file,
                sheet_name=selected_sheet,
                header_row=int(header_row_number) - 1,
            )

            project_service = SurveyProjectService()

            project = project_service.load_raw_data(
                project=project,
                dataframe=dataframe,
                file_name=workbook_info.file_name,
                file_size_bytes=file_size,
                sheet_names=workbook_info.sheet_names,
                selected_sheet=selected_sheet,
                header_row=int(header_row_number),
            )

            st.session_state["project"] = project

            st.success(
                "Raw data loaded and question families "
                "detected successfully."
            )

        except (WorkbookReaderError, ValueError) as error:
            st.error(str(error))
            return

    if not project.has_raw_data:
        return

    dataframe = project.raw_dataframe

    if dataframe is None:
        return

    st.divider()
    st.subheader("Loaded Dataset")

    if project.workbook:
        st.caption(
            f"Workbook: {project.workbook.file_name} | "
            f"Worksheet: {project.workbook.selected_sheet} | "
            f"Header row: {project.workbook.header_row}"
        )

    _display_dataframe_summary(dataframe)
    _display_column_warnings(dataframe)

    maximum_preview_rows = min(
        100,
        max(5, len(dataframe)),
    )

    default_preview_rows = min(
        10,
        maximum_preview_rows,
    )

    preview_row_count = st.slider(
        "Rows to preview",
        min_value=5,
        max_value=maximum_preview_rows,
        value=default_preview_rows,
    )

    st.dataframe(
        dataframe.head(preview_row_count),
        use_container_width=True,
        height=420,
    )

    _display_column_list(dataframe)

    st.info(
        "The detected question families are available "
        "on the 'Raw Data Dictionary' page."
    )