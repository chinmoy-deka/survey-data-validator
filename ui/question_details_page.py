from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st

from core.models import (
    QuestionFamily,
    SurveyProject,
    TableValidationResult,
    ValidationStatus,
)


def _get_project() -> SurveyProject | None:
    project = st.session_state.get("project")
    return project if isinstance(project, SurveyProject) else None


def _normalize(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _family_name(family: QuestionFamily) -> str:
    return str(
        getattr(family, "name", None)
        or getattr(family, "family_name", None)
        or getattr(family, "question_id", None)
        or ""
    )


def _type_text(value: Any) -> str:
    if value is None:
        return "Not set"
    return str(getattr(value, "value", value))


def _status_icon(status: ValidationStatus | None) -> str:
    icons = {
        ValidationStatus.PASS: "✅",
        ValidationStatus.FAIL: "❌",
        ValidationStatus.WARNING: "⚠️",
        ValidationStatus.UNMATCHED: "🔗",
        ValidationStatus.NOT_VALIDATED: "➖",
    }
    return icons.get(status, "➖")


def _matching_families(
    project: SurveyProject,
    question_id: str,
) -> list[QuestionFamily]:
    target = _normalize(question_id)
    matches: list[QuestionFamily] = []

    for family in project.question_families:
        names = {
            _normalize(_family_name(family)),
            _normalize(getattr(family, "source_family_name", "")),
        }

        metadata = getattr(family, "metadata", {}) or {}
        for key in ("question_id", "source_question_id", "base_question_id"):
            names.add(_normalize(metadata.get(key, "")))

        # Exact family match, source-family match, or refined grid row/rank family.
        if target in names or any(
            name.startswith(target + "R") or name.startswith(target + "RANK")
            for name in names
            if name
        ):
            matches.append(family)

    def sort_key(family: QuestionFamily) -> tuple[int, str]:
        name = _family_name(family)
        number_match = re.search(r"(?:_r|rank\s*)(\d+)", name, re.IGNORECASE)
        return (
            int(number_match.group(1)) if number_match else 0,
            name.lower(),
        )

    return sorted(matches, key=sort_key)


def _matching_results(
    project: SurveyProject,
    question_id: str,
) -> list[TableValidationResult]:
    target = _normalize(question_id)
    return [
        result
        for result in project.validation_results
        if _normalize(result.question_id) == target
    ]


def _build_family_dataframe(
    families: list[QuestionFamily],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in families:
        metadata = getattr(family, "metadata", {}) or {}
        rows.append(
            {
                "Family": _family_name(family),
                "Columns": family.number_of_columns,
                "Detected Type": _type_text(family.detected_type),
                "Confirmed Type": _type_text(family.confirmed_type),
                "Final Type": _type_text(family.final_type),
                "Structural Type": family.structural_type,
                "Confidence": f"{float(family.confidence or 0):.1f}%",
                "Strategy / Mode": (
                    metadata.get("validation_strategy")
                    or metadata.get("denominator_strategy")
                    or metadata.get("rank_mode")
                    or ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_table_dataframe(tables: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Table": table.table_number,
                "Title": table.table_title,
                "Question Text": table.question_text,
                "Reported Base": table.total_respondents,
                "Options": table.option_count,
                "Column Code": table.column_code or "",
                "Warnings": "; ".join(table.warnings),
            }
            for table in tables
        ]
    )


def _build_validation_dataframe(
    results: list[TableValidationResult],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Status": f"{_status_icon(result.overall_status)} {result.overall_status.value}",
                "Table": result.table_number,
                "Title": result.table_title,
                "Matched Family": result.matched_family_name or "",
                "Question Type": result.question_type,
                "Reported Base": result.reported_respondents,
                "Calculated Base": result.calculated_respondents,
                "Passed Options": result.passed_options,
                "Failed Options": result.failed_options,
                "Warnings": result.warning_options,
            }
            for result in results
        ]
    )


def _metadata_dataframe(metadata: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, value in sorted(metadata.items()):
        if isinstance(value, dict):
            value = ", ".join(f"{k}: {v}" for k, v in value.items())
        elif isinstance(value, (list, tuple, set)):
            value = ", ".join(map(str, value))
        rows.append({"Metadata": key, "Value": value})
    return pd.DataFrame(rows)


def _render_family_details(family: QuestionFamily) -> None:
    metadata = getattr(family, "metadata", {}) or {}
    columns = st.columns(4)
    columns[0].metric("Detected", _type_text(family.detected_type))
    columns[1].metric("Confirmed", _type_text(family.confirmed_type))
    columns[2].metric("Final Type", _type_text(family.final_type))
    columns[3].metric("Confidence", f"{float(family.confidence or 0):.1f}%")

    st.write(f"**Family:** {_family_name(family)}")
    st.write(f"**Structural type:** {family.structural_type or 'Not set'}")
    st.write(f"**Grouping method:** {family.grouping_method or 'Not set'}")
    st.write(f"**Reason:** {family.reason or metadata.get('classification_reason') or 'Not recorded'}")

    st.markdown("#### Raw variables")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Index": column.index,
                    "Variable": column.name,
                    "Data Type": column.dtype,
                    "Suffix": column.suffix or "",
                }
                for column in family.columns
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if metadata:
        with st.expander("Classification and strategy metadata"):
            st.dataframe(
                _metadata_dataframe(metadata),
                use_container_width=True,
                hide_index=True,
            )


def _render_result_details(result: TableValidationResult) -> None:
    st.write(
        f"**Overall status:** {_status_icon(result.overall_status)} "
        f"{result.overall_status.value}"
    )
    st.write(f"**Matched family:** {result.matched_family_name or 'Not matched'}")
    st.write(f"**Question type:** {result.question_type}")

    metrics = st.columns(4)
    metrics[0].metric("Reported Base", result.reported_respondents or "Not found")
    metrics[1].metric("Calculated Base", result.calculated_respondents or "Not calculated")
    metrics[2].metric("Passed Options", result.passed_options)
    metrics[3].metric("Failed Options", result.failed_options)

    for message in result.messages:
        st.warning(message)

    if result.option_results:
        rows = []
        for option in result.option_results:
            rows.append(
                {
                    "Status": f"{_status_icon(option.status)} {option.status.value}",
                    "Tabulation Option": option.option_label,
                    "Raw Variable": option.raw_variable or "",
                    "Raw Value": option.raw_value if option.raw_value is not None else "",
                    "Base / Count": option.calculated_count,
                    "Calculated %": option.calculated_percentage,
                    "Reported %": option.reported_percentage,
                    "Difference (pp)": option.difference,
                    "Message": option.message,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_question_details_page() -> None:
    st.title("Question Inspector")
    st.write(
        "Inspect the imported question metadata, raw-data family classification, "
        "tabulation tables and the latest validation trace in one place."
    )

    project = _get_project()
    if project is None:
        st.error("Survey project is not initialized.")
        return

    if project.tabulation_workbook is None:
        st.warning("Import the tabulation workbook first.")
        return

    tables_by_question: dict[str, list[Any]] = defaultdict(list)
    for table in project.tabulation_workbook.tables:
        qid = str(table.question_id or "").strip()
        if qid:
            tables_by_question[qid].append(table)

    if not tables_by_question:
        st.info("No question IDs were detected in the imported tabulation workbook.")
        return

    question_ids = sorted(
        tables_by_question,
        key=lambda value: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)
        ],
    )

    selected_question = st.selectbox(
        "Select question",
        options=question_ids,
        key="question_inspector_question",
    )

    tables = tables_by_question[selected_question]
    families = _matching_families(project, selected_question)
    results = _matching_results(project, selected_question)

    headline = tables[0].question_text or tables[0].table_title
    st.subheader(f"{selected_question} — {headline}")

    metrics = st.columns(4)
    metrics[0].metric("Tabulation Tables", len(tables))
    metrics[1].metric("Matched Families", len(families))
    metrics[2].metric("Raw Variables", sum(f.number_of_columns for f in families))
    metrics[3].metric("Validation Results", len(results))

    overview_tab, family_tab, tabulation_tab, validation_tab, raw_tab = st.tabs(
        [
            "Overview",
            "Classification & Variables",
            "Tabulation",
            "Validation Trace",
            "Raw Preview",
        ]
    )

    with overview_tab:
        st.markdown("### Question overview")
        st.dataframe(_build_table_dataframe(tables), use_container_width=True, hide_index=True)

        if families:
            st.markdown("### Raw-data families")
            st.dataframe(_build_family_dataframe(families), use_container_width=True, hide_index=True)
        else:
            st.warning("No raw-data family could be matched to this question ID.")

        if results:
            st.markdown("### Latest validation status")
            st.dataframe(_build_validation_dataframe(results), use_container_width=True, hide_index=True)
        else:
            st.info("Run validation to populate the validation trace.")

    with family_tab:
        if not families:
            st.info("No matching raw-data family is available.")
        else:
            family_labels = {
                f"{_family_name(family)} — {_type_text(family.final_type)}": family
                for family in families
            }
            selected_family_label = st.selectbox(
                "Select family",
                options=list(family_labels),
                key="question_inspector_family",
            )
            _render_family_details(family_labels[selected_family_label])

    with tabulation_tab:
        table_labels = {
            f"Table {table.table_number} — {table.table_title}": table
            for table in tables
        }
        selected_table_label = st.selectbox(
            "Select tabulation table",
            options=list(table_labels),
            key="question_inspector_table",
        )
        table = table_labels[selected_table_label]

        st.write(f"**Question text:** {table.question_text or 'Not available'}")
        st.write(f"**Base label:** {table.base_label or 'Not available'}")
        st.write(f"**Column code:** {table.column_code or 'Not available'}")
        if table.warnings:
            for warning in table.warnings:
                st.warning(warning)

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Option": option.label,
                        "Reported Value": option.reported_value,
                        "Reported Percentage": option.reported_percentage,
                        "Source Row": option.source_row,
                    }
                    for option in table.options
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with validation_tab:
        if not results:
            st.info("Run validation first to see calculated values, bases and messages.")
        else:
            result_labels = {
                (
                    f"{_status_icon(result.overall_status)} Table {result.table_number} — "
                    f"{result.table_title}"
                ): result
                for result in results
            }
            selected_result_label = st.selectbox(
                "Select validation result",
                options=list(result_labels),
                key="question_inspector_result",
            )
            _render_result_details(result_labels[selected_result_label])

    with raw_tab:
        dataframe = getattr(project, "raw_dataframe", None)
        matched_columns = [
            name
            for family in families
            for name in family.column_names
            if dataframe is not None and name in dataframe.columns
        ]
        matched_columns = list(dict.fromkeys(matched_columns))

        if dataframe is None or not matched_columns:
            st.info("No matched raw variables are available for preview.")
        else:
            preview_rows = st.slider(
                "Preview rows",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                key="question_inspector_preview_rows",
            )
            st.dataframe(
                dataframe[matched_columns].head(preview_rows),
                use_container_width=True,
                hide_index=True,
            )

            summary_rows = []
            for column in matched_columns:
                series = dataframe[column]
                summary_rows.append(
                    {
                        "Variable": column,
                        "Nonblank": int(series.notna().sum()),
                        "Blank": int(series.isna().sum()),
                        "Unique": int(series.nunique(dropna=True)),
                        "Minimum": pd.to_numeric(series, errors="coerce").min(),
                        "Maximum": pd.to_numeric(series, errors="coerce").max(),
                    }
                )
            st.markdown("#### Variable profile")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
