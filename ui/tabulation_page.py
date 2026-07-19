from __future__ import annotations

import pandas as pd
import streamlit as st

from core.models import (
    SurveyProject,
    TabulationTable,
)
from core.tabulation_parser import TabulationParserError
from services.tabulation_service import TabulationService


def _get_project() -> SurveyProject | None:
    project = st.session_state.get("project")

    if isinstance(project, SurveyProject):
        return project

    return None


def _build_table_summary(
    tables: list[TabulationTable],
) -> pd.DataFrame:
    rows = []

    for table in tables:
        rows.append(
            {
                "Table No.": table.table_number,
                "Question ID": table.question_id,
                "Table Title": table.table_title,
                "Total Respondents": table.total_respondents,
                "Options": table.option_count,
                "Start Row": table.start_row,
                "End Row": table.end_row,
                "Warnings": "; ".join(table.warnings),
            }
        )

    return pd.DataFrame(rows)


def _build_option_dataframe(
    table: TabulationTable,
) -> pd.DataFrame:
    rows = []

    for option in table.options:
        rows.append(
            {
                "Option": option.label,
                "Reported Value": option.reported_value,
                "Reported Percentage": (
                    option.reported_percentage
                ),
                "Source Row": option.source_row,
            }
        )

    return pd.DataFrame(rows)


def _display_tabulation_summary(
    project: SurveyProject,
) -> None:
    tabulation = project.tabulation_workbook

    if tabulation is None:
        return

    warning_count = sum(
        len(table.warnings)
        for table in tabulation.tables
    )

    option_count = sum(
        table.option_count
        for table in tabulation.tables
    )

    metrics = st.columns(4)

    metrics[0].metric(
        "Tables Detected",
        f"{tabulation.table_count:,}",
    )

    metrics[1].metric(
        "Question IDs",
        f"{len(tabulation.question_ids):,}",
    )

    metrics[2].metric(
        "Reported Options",
        f"{option_count:,}",
    )

    metrics[3].metric(
        "Parser Warnings",
        f"{warning_count:,}",
    )


def render_tabulation_page() -> None:
    st.title("Import Tabulation Workbook")

    st.write(
        "Upload the survey tabulation workbook. The application "
        "will detect the individual tables, respondent totals, "
        "option labels and reported percentages."
    )

    project = _get_project()

    if project is None:
        st.error("Survey project is not initialized.")
        return

    uploaded_file = st.file_uploader(
        "Upload tabulation workbook",
        type=["xlsx", "xlsm"],
        key="tabulation_uploader",
    )

    if uploaded_file is not None:
        service = TabulationService()

        try:
            sheet_names = service.get_sheet_names(
                uploaded_file
            )
        except TabulationParserError as error:
            st.error(str(error))
            return

        selected_sheet = st.selectbox(
            "Select tabulation worksheet",
            options=sheet_names,
        )

        if st.button(
            "Import and Parse Tables",
            type="primary",
            use_container_width=True,
        ):
            try:
                project = service.import_tabulation(
                    project=project,
                    file=uploaded_file,
                    file_name=uploaded_file.name,
                    sheet_name=selected_sheet,
                )

                st.session_state["project"] = project

                st.success(
                    "Tabulation workbook imported and "
                    "parsed successfully."
                )

            except TabulationParserError as error:
                st.error(str(error))
                return

    tabulation = project.tabulation_workbook

    if tabulation is None:
        st.info(
            "Upload a tabulation workbook to begin."
        )
        return

    st.divider()

    st.caption(
        f"Workbook: {tabulation.file_name} | "
        f"Worksheet: {tabulation.selected_sheet}"
    )

    _display_tabulation_summary(project)

    st.subheader("Detected Tables")

    summary_dataframe = _build_table_summary(
        tabulation.tables
    )

    st.dataframe(
        summary_dataframe,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    st.divider()
    st.subheader("Inspect Parsed Table")

    table_options = {
        (
            f"Table {table.table_number} — "
            f"{table.question_id} — "
            f"{table.table_title}"
        ): table
        for table in tabulation.tables
    }

    selected_table_label = st.selectbox(
        "Select table",
        options=list(table_options.keys()),
    )

    selected_table = table_options[
        selected_table_label
    ]

    details = st.columns(4)

    details[0].metric(
        "Table Number",
        selected_table.table_number,
    )

    details[1].metric(
        "Question ID",
        selected_table.question_id,
    )

    details[2].metric(
        "Total Respondents",
        (
            selected_table.total_respondents
            if selected_table.total_respondents
            is not None
            else "Not found"
        ),
    )

    details[3].metric(
        "Options",
        selected_table.option_count,
    )

    st.write(
        f"**Table title:** "
        f"{selected_table.table_title}"
    )

    st.write(
        f"**Question text:** "
        f"{selected_table.question_text}"
    )

    if selected_table.base_label:
        st.write(
            f"**Base:** {selected_table.base_label}"
        )

    if selected_table.column_code:
        st.write(
            f"**Column code:** "
            f"{selected_table.column_code}"
        )

    if selected_table.warnings:
        for warning in selected_table.warnings:
            st.warning(warning)

    option_dataframe = _build_option_dataframe(
        selected_table
    )

    st.dataframe(
        option_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The Reported Percentage column converts values such "
        "as 0.75 into 75%. Raw-data comparison will be added "
        "in the next step."
    )