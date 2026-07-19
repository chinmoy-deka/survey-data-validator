from __future__ import annotations

import streamlit as st

from core.models import SurveyProject
from ui.dictionary_editor import render_dictionary_editor
from ui.raw_data_page import render_raw_data_page
from ui.tabulation_page import render_tabulation_page
from ui.question_details_page import render_question_details_page
from ui.validation_page import render_validation_page


st.set_page_config(
    page_title="Survey Data Validator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    if "project" not in st.session_state:
        st.session_state["project"] = SurveyProject()


def render_placeholder_page(
    title: str,
    message: str,
) -> None:
    st.title(title)
    st.info(message)


def render_sidebar_summary() -> None:
    project = st.session_state.get("project")

    if (
        not isinstance(project, SurveyProject)
        or not project.has_raw_data
    ):
        st.sidebar.info("No dataset loaded.")
        return

    st.sidebar.divider()
    st.sidebar.subheader("Current Project")

    st.sidebar.write(
        f"**Project:** {project.project_name}"
    )

    if project.workbook:
        st.sidebar.write(
            f"**File:** {project.workbook.file_name}"
        )
        st.sidebar.write(
            f"**Sheet:** "
            f"{project.workbook.selected_sheet}"
        )

    st.sidebar.write(
        f"**Rows:** {project.row_count:,}"
    )
    st.sidebar.write(
        f"**Columns:** {project.column_count:,}"
    )
    st.sidebar.write(
        f"**Question Families:** "
        f"{project.question_family_count:,}"
    )

    if project.tabulation_workbook:
            st.sidebar.write(
                f"**Tabulation Tables:** "
                f"{project.tabulation_workbook.table_count:,}"
    )

    if project.dictionary_saved:
        st.sidebar.success("Dictionary reviewed")
    else:
        st.sidebar.warning("Dictionary not reviewed")


def main() -> None:
    initialize_session_state()

    st.sidebar.title("Survey Data Validator")

    selected_page = st.sidebar.radio(
        "Navigation",
            options=[
            "Upload Raw Data",
            "Raw Data Dictionary",
            "Import Tabulation",
            "Question Details",
            "Validation",
            "Report",
            "Settings",
            ],
    )

    render_sidebar_summary()

    if selected_page == "Upload Raw Data":
        render_raw_data_page()

    elif selected_page == "Raw Data Dictionary":
        render_dictionary_editor()

    elif selected_page == "Import Tabulation":
        render_tabulation_page()

    elif selected_page == "Question Details":
        render_question_details_page()

    elif selected_page == "Validation":
        render_validation_page()

    elif selected_page == "Report":
        render_placeholder_page(
            "Report",
            "Validation report generation will be "
            "implemented later.",
        )

    elif selected_page == "Settings":
        render_placeholder_page(
            "Settings",
            "Project and application settings will "
            "be implemented later.",
        )


if __name__ == "__main__":
    main()