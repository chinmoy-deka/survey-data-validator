from __future__ import annotations

import pandas as pd
import streamlit as st

from core.models import (
    QuestionType,
    SurveyProject,
)


def _get_project() -> SurveyProject | None:
    project = st.session_state.get("project")

    if isinstance(project, SurveyProject):
        return project

    return None


def _build_dictionary_dataframe(
    project: SurveyProject,
) -> pd.DataFrame:
    rows = []

    for family in project.question_families:
        rows.append(
            {
                "Question Family": family.name,
                "Detected Type": family.detected_type.value,
                "Confirmed Type": family.final_type.value,
                "Confidence": round(
                    family.confidence * 100,
                    1,
                ),
                "No. of Columns": family.number_of_columns,
                "Variables": ", ".join(
                    family.column_names
                ),
                "Detection Reason": family.reason,
            }
        )

    return pd.DataFrame(rows)


def _apply_dictionary_edits(
    project: SurveyProject,
    edited_dataframe: pd.DataFrame,
) -> None:
    family_lookup = {
        family.name: family
        for family in project.question_families
    }

    for _, row in edited_dataframe.iterrows():
        family_name = str(row["Question Family"])
        selected_type = str(row["Confirmed Type"])

        family = family_lookup.get(family_name)

        if family is None:
            continue

        selected_question_type = QuestionType(
            selected_type
        )

        if selected_question_type == family.detected_type:
            family.confirmed_type = None
        else:
            family.confirmed_type = selected_question_type

    project.dictionary_saved = True
    st.session_state["project"] = project


def _display_dictionary_summary(
    project: SurveyProject,
    dictionary_dataframe: pd.DataFrame,
) -> None:
    low_confidence_count = sum(
        family.confidence < 0.70
        for family in project.question_families
    )

    manual_review_count = sum(
        family.final_type == QuestionType.MANUAL_REVIEW
        for family in project.question_families
    )

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Question Families",
        project.question_family_count,
    )

    metric_columns[1].metric(
        "Physical Variables",
        project.column_count,
    )

    metric_columns[2].metric(
        "Low Confidence",
        low_confidence_count,
    )

    metric_columns[3].metric(
        "Manual Review",
        manual_review_count,
    )


def _display_type_distribution(
    dictionary_dataframe: pd.DataFrame,
) -> None:
    st.subheader("Question Type Distribution")

    type_distribution = (
        dictionary_dataframe["Confirmed Type"]
        .value_counts()
        .rename_axis("Question Type")
        .reset_index(name="Count")
    )

    st.dataframe(
        type_distribution,
        use_container_width=True,
        hide_index=True,
    )


def render_dictionary_editor() -> None:
    st.title("Raw Data Dictionary")

    project = _get_project()

    if project is None or not project.has_raw_data:
        st.warning(
            "No raw data is loaded. Open 'Upload Raw Data' "
            "and load a workbook first."
        )
        return

    if not project.question_families:
        st.warning(
            "No question families were detected."
        )
        return

    if project.workbook:
        st.caption(
            f"Workbook: {project.workbook.file_name} | "
            f"Worksheet: {project.workbook.selected_sheet}"
        )

    dictionary_dataframe = (
        _build_dictionary_dataframe(project)
    )

    _display_dictionary_summary(
        project,
        dictionary_dataframe,
    )

    st.divider()

    st.write(
        "Review the automatically detected question types. "
        "Use the Confirmed Type column to correct a "
        "classification."
    )

    question_type_options = [
        question_type.value
        for question_type in QuestionType
    ]

    edited_dataframe = st.data_editor(
        dictionary_dataframe,
        use_container_width=True,
        hide_index=True,
        height=600,
        disabled=[
            "Question Family",
            "Detected Type",
            "Confidence",
            "No. of Columns",
            "Variables",
            "Detection Reason",
        ],
        column_config={
            "Question Family":
                st.column_config.TextColumn(
                    "Question Family",
                    width="small",
                ),
            "Detected Type":
                st.column_config.TextColumn(
                    "Detected Type",
                    width="medium",
                ),
            "Confirmed Type":
                st.column_config.SelectboxColumn(
                    "Confirmed Type",
                    options=question_type_options,
                    required=True,
                    width="medium",
                ),
            "Confidence":
                st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                    width="small",
                ),
            "No. of Columns":
                st.column_config.NumberColumn(
                    "No. of Columns",
                    width="small",
                ),
            "Variables":
                st.column_config.TextColumn(
                    "Variables",
                    width="large",
                ),
            "Detection Reason":
                st.column_config.TextColumn(
                    "Detection Reason",
                    width="large",
                ),
        },
        key="dictionary_data_editor",
    )

    if st.button(
        "Save Dictionary Changes",
        type="primary",
        use_container_width=True,
    ):
        try:
            _apply_dictionary_edits(
                project,
                edited_dataframe,
            )

            st.success(
                "Dictionary changes saved successfully."
            )

        except ValueError as error:
            st.error(
                f"Unable to save the dictionary: {error}"
            )

    if project.dictionary_saved:
        st.success("Dictionary has been reviewed and saved.")

    st.divider()

    _display_type_distribution(edited_dataframe)