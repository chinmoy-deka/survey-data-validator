from __future__ import annotations

import pandas as pd
import streamlit as st

from core.models import (
    OptionValidationResult,
    SurveyProject,
    TableValidationResult,
    ValidationStatus,
)
from core.validation_engine import ValidationEngineError
from services.validation_service import ValidationService


def _get_project() -> SurveyProject | None:
    project = st.session_state.get("project")
    return project if isinstance(project, SurveyProject) else None


def _format_percentage(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}%"


def _status_icon(status: ValidationStatus) -> str:
    icons = {
        ValidationStatus.PASS: "✅",
        ValidationStatus.FAIL: "❌",
        ValidationStatus.WARNING: "⚠️",
        ValidationStatus.UNMATCHED: "🔗",
        ValidationStatus.NOT_VALIDATED: "➖",
    }
    return icons.get(status, "➖")


def _status_text(status: ValidationStatus) -> str:
    return f"{_status_icon(status)} {status.value}"


def _derive_option_status(
    result: TableValidationResult,
) -> ValidationStatus:
    statuses = [item.status for item in result.option_results]
    if not statuses:
        return ValidationStatus.NOT_VALIDATED
    if ValidationStatus.FAIL in statuses:
        return ValidationStatus.FAIL
    if ValidationStatus.UNMATCHED in statuses:
        return ValidationStatus.WARNING
    if ValidationStatus.WARNING in statuses:
        return ValidationStatus.WARNING
    if all(status == ValidationStatus.PASS for status in statuses):
        return ValidationStatus.PASS
    return ValidationStatus.NOT_VALIDATED


def _build_summary_dataframe(
    results: list[TableValidationResult],
) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "Overall": _status_text(result.overall_status),
                "Base": _status_text(result.respondent_status),
                "Options Status": _status_text(
                    _derive_option_status(result)
                ),
                "Table": result.table_number,
                "Question ID": result.question_id,
                "Matched Family": result.matched_family_name or "",
                "Question Type": result.question_type,
                "Reported Base": result.reported_respondents,
                "Calculated Base": result.calculated_respondents,
                "Base Difference": result.respondent_difference,
                "Options": result.option_count,
                "Passed": result.passed_options,
                "Failed": result.failed_options,
                "Warnings": result.warning_options,
                "Title": result.table_title,
            }
        )
    return pd.DataFrame(rows)


def _build_option_dataframe(
    results: list[OptionValidationResult],
) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "Status": _status_text(result.status),
                "Tabulation Option": result.option_label,
                "Raw Variable": result.raw_variable or "",
                "Raw Code": (
                    result.raw_value
                    if result.raw_value is not None
                    else ""
                ),
                "Reported": _format_percentage(
                    result.reported_percentage
                ),
                "Calculated Count": result.calculated_count,
                "Calculated": _format_percentage(
                    result.calculated_percentage
                ),
                "Difference": (
                    ""
                    if result.difference is None
                    else f"{result.difference:+.2f} pp"
                ),
                "Message": result.message,
            }
        )
    return pd.DataFrame(rows)


def _display_metrics(
    results: list[TableValidationResult],
) -> None:
    counts = {
        status: sum(
            result.overall_status == status for result in results
        )
        for status in ValidationStatus
    }
    columns = st.columns(5)
    columns[0].metric("Tables Validated", len(results))
    columns[1].metric("Passed", counts[ValidationStatus.PASS])
    columns[2].metric("Failed", counts[ValidationStatus.FAIL])
    columns[3].metric(
        "Warnings", counts[ValidationStatus.WARNING]
    )
    columns[4].metric(
        "Unmatched", counts[ValidationStatus.UNMATCHED]
    )


def _display_message(message: str) -> None:
    lowered = message.lower()
    if "invalid" in lowered or "outside" in lowered:
        st.error(message)
    elif "not automatically validated" in lowered:
        st.info(message)
    else:
        st.warning(message)


def _display_result_details(
    result: TableValidationResult,
) -> None:
    option_status = _derive_option_status(result)

    st.subheader(
        f"Table {result.table_number} — {result.question_id}"
    )
    st.write(f"**Title:** {result.table_title}")
    st.write(
        "**Matched raw-data family:** "
        f"{result.matched_family_name or 'Not matched'}"
    )
    st.write(f"**Question type:** {result.question_type}")

    status_columns = st.columns(3)
    status_columns[0].metric(
        "Overall Status", _status_text(result.overall_status)
    )
    status_columns[1].metric(
        "Base Validation", _status_text(result.respondent_status)
    )
    status_columns[2].metric(
        "Option Validation", _status_text(option_status)
    )

    st.caption(
        "Overall status combines base validation, option validation, "
        "coding-format checks, and matching status. A passing base does "
        "not override an option-level failure or warning."
    )

    base_columns = st.columns(3)
    base_columns[0].metric(
        "Reported Respondents",
        result.reported_respondents
        if result.reported_respondents is not None
        else "Not found",
    )
    base_columns[1].metric(
        "Calculated Respondents",
        result.calculated_respondents
        if result.calculated_respondents is not None
        else "Not calculated",
    )
    base_columns[2].metric(
        "Base Difference",
        result.respondent_difference
        if result.respondent_difference is not None
        else "—",
    )

    if result.messages:
        st.markdown("#### Validation Notes")
        for message in result.messages:
            _display_message(message)

    if not result.option_results:
        st.info(
            "No option-level validation results are available for "
            "this table."
        )
        return

    st.markdown("#### Option Validation")
    st.dataframe(
        _build_option_dataframe(result.option_results),
        use_container_width=True,
        hide_index=True,
        height=500,
    )


def render_validation_page() -> None:
    st.title("Tabulation Validation")
    st.write(
        "Compare the imported tabulation tables against the uploaded "
        "raw survey data."
    )

    project = _get_project()
    if project is None:
        st.error("Survey project is not initialized.")
        return

    raw_dataframe = getattr(
        project,
        "raw_dataframe",
        getattr(project, "raw_data", None),
    )
    if raw_dataframe is None:
        st.warning(
            "Upload and process the raw-data workbook first."
        )
        return
    if project.tabulation_workbook is None:
        st.warning("Import the tabulation workbook first.")
        return
    if not project.question_families:
        st.warning(
            "No question families are available. Process the raw-data "
            "dictionary first."
        )
        return

    settings_columns = st.columns(2)
    percentage_tolerance = settings_columns[0].number_input(
        "Percentage tolerance (percentage points)",
        min_value=0.0,
        max_value=10.0,
        value=0.5,
        step=0.1,
        help=(
            "Reported and calculated percentages pass when their "
            "absolute difference is within this tolerance."
        ),
    )
    respondent_tolerance = settings_columns[1].number_input(
        "Respondent-count tolerance",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help=(
            "Normally this should remain zero. Increase it only when "
            "small base differences are acceptable."
        ),
    )

    action_columns = st.columns([2, 1])
    with action_columns[0]:
        validate_clicked = st.button(
            "Validate All Tables",
            type="primary",
            use_container_width=True,
        )
    with action_columns[1]:
        clear_clicked = st.button(
            "Clear Results", use_container_width=True
        )

    if clear_clicked:
        project.validation_results = []
        st.session_state["project"] = project
        st.rerun()

    if validate_clicked:
        service = ValidationService()
        try:
            with st.spinner("Validating tabulation tables..."):
                results = service.validate_all(
                    project=project,
                    percentage_tolerance=percentage_tolerance,
                    respondent_tolerance=int(
                        respondent_tolerance
                    ),
                )
            st.session_state["project"] = project
            st.success(
                f"Validation completed for {len(results):,} tables."
            )
        except ValidationEngineError as error:
            st.error(str(error))
            return
        except Exception as error:
            st.exception(error)
            return

    results = project.validation_results
    if not results:
        st.info(
            "Select the validation tolerances and click 'Validate All "
            "Tables'."
        )
        return

    st.divider()
    _display_metrics(results)
    st.subheader("Validation Summary")

    status_filter = st.multiselect(
        "Filter by status",
        options=[status.value for status in ValidationStatus],
        default=[
            ValidationStatus.PASS.value,
            ValidationStatus.FAIL.value,
            ValidationStatus.WARNING.value,
            ValidationStatus.UNMATCHED.value,
            ValidationStatus.NOT_VALIDATED.value,
        ],
    )
    question_search = st.text_input(
        "Search question ID or title"
    ).strip().lower()

    filtered_results = []
    for result in results:
        if result.overall_status.value not in status_filter:
            continue
        searchable_text = (
            f"{result.question_id} {result.table_title} "
            f"{result.matched_family_name or ''}"
        ).lower()
        if question_search and question_search not in searchable_text:
            continue
        filtered_results.append(result)

    st.dataframe(
        _build_summary_dataframe(filtered_results),
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    if not filtered_results:
        st.warning(
            "No validation results match the selected filters."
        )
        return

    st.divider()
    result_labels = {
        (
            f"{_status_icon(result.overall_status)} "
            f"Table {result.table_number} — {result.question_id} — "
            f"{result.table_title}"
        ): result
        for result in filtered_results
    }
    selected_label = st.selectbox(
        "Inspect validation result",
        options=list(result_labels.keys()),
    )
    _display_result_details(result_labels[selected_label])
