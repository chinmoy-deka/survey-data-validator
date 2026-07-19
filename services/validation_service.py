from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict

from core.models import ColumnInfo, QuestionFamily, QuestionType, SurveyProject, TableValidationResult
from core.validation_engine import TabulationValidationEngine, ValidationEngineError
from core.validation_strategy_engine import ValidationStrategy, ValidationStrategyEngine


class ValidationService:
    """Coordinate matching, strategy selection, and table validation."""

    def __init__(self) -> None:
        self.strategy_engine = ValidationStrategyEngine()

    @staticmethod
    def _get_raw_dataframe(project: SurveyProject):
        return getattr(project, "raw_dataframe", getattr(project, "raw_data", None))

    def validate_all(
        self,
        project: SurveyProject,
        percentage_tolerance: float = 0.5,
        respondent_tolerance: int = 0,
    ) -> list[TableValidationResult]:
        raw = self._get_raw_dataframe(project)
        self._validate_project_state(project, raw)
        raw, base_note = self._select_analysis_base(raw, project.tabulation_workbook.tables)
        engine = TabulationValidationEngine(percentage_tolerance, respondent_tolerance)
        family_index = self._build_family_index(project.question_families)
        tables = project.tabulation_workbook.tables
        groups = self._group_tables(tables)
        strategy_map = self._build_strategy_map(groups, list(raw.columns))
        grid_assignment = self._build_grid_assignments(groups, strategy_map, raw)
        occurrence = defaultdict(int)
        results = []

        for table in tables:
            qid = self._normalize_question_id(table.question_id)
            index = occurrence[qid]
            occurrence[qid] += 1
            decision = strategy_map[table.table_number]
            family = self._prepare_family_for_strategy(
                question_id=table.question_id,
                family_index=family_index,
                raw_columns=list(raw.columns),
                occurrence_index=index,
                decision=decision,
                assigned_row=grid_assignment.get(table.table_number),
            )
            result = engine.validate_table(raw, table, family)
            result.messages.insert(0, f"Analysis base: {base_note}")
            result.messages.insert(0, f"Validation strategy: {decision.strategy.value}. {decision.reason}")
            results.append(result)

        project.validation_results = results
        return results

    def validate_one(
        self,
        project: SurveyProject,
        table_number: int,
        percentage_tolerance: float = 0.5,
        respondent_tolerance: int = 0,
    ) -> TableValidationResult:
        raw = self._get_raw_dataframe(project)
        self._validate_project_state(project, raw)
        tables = project.tabulation_workbook.tables
        table = next((x for x in tables if x.table_number == table_number), None)
        if table is None:
            raise ValidationEngineError(f"Table {table_number} was not found.")
        raw, base_note = self._select_analysis_base(raw, tables)
        groups = self._group_tables(tables)
        qid = self._normalize_question_id(table.question_id)
        group = groups[qid]
        index = group.index(table)
        local_strategy = self.strategy_engine.classify_group(group, list(raw.columns))
        decision = local_strategy[table.table_number]
        assigned_row = self._build_grid_assignments({qid: group}, local_strategy, raw).get(table.table_number)
        family = self._prepare_family_for_strategy(
            table.question_id,
            self._build_family_index(project.question_families),
            list(raw.columns),
            index,
            decision,
            assigned_row=assigned_row,
        )
        result = TabulationValidationEngine(
            percentage_tolerance, respondent_tolerance
        ).validate_table(raw, table, family)
        result.messages.insert(0, f"Analysis base: {base_note}")
        result.messages.insert(0, f"Validation strategy: {decision.strategy.value}. {decision.reason}")
        return result

    @staticmethod
    def _validate_project_state(project: SurveyProject, raw) -> None:
        if raw is None:
            raise ValidationEngineError("Upload the raw-data workbook first.")
        if project.tabulation_workbook is None:
            raise ValidationEngineError("Import the tabulation workbook first.")
        if not project.question_families:
            raise ValidationEngineError("No raw-data question families are available.")

    def _group_tables(self, tables):
        groups = defaultdict(list)
        for table in tables:
            groups[self._normalize_question_id(table.question_id)].append(table)
        return groups

    def _build_strategy_map(self, groups, raw_columns):
        result = {}
        for tables in groups.values():
            result.update(self.strategy_engine.classify_group(tables, raw_columns))
        return result

    def _build_family_index(self, families: list[QuestionFamily]):
        index = defaultdict(list)
        for family in families:
            index[self._normalize_question_id(family.name)].append(family)
        return index

    def _prepare_family_for_strategy(
        self,
        question_id: str,
        family_index,
        raw_columns: list[str],
        occurrence_index: int,
        decision,
        assigned_row: int | None = None,
    ) -> QuestionFamily | None:
        qid = self._normalize_question_id(question_id)
        exact = family_index.get(qid, [])

        if decision.strategy in {ValidationStrategy.RANK_EXACT, ValidationStrategy.RANK_CUMULATIVE}:
            if len(exact) != 1:
                return None
            family = copy.deepcopy(exact[0])
            family.name = f"{exact[0].name}_rank{decision.target_rank}"
            family.source_family_name = exact[0].name
            family.detected_type = QuestionType.RANKING
            family.confirmed_type = QuestionType.RANKING
            family.structural_type = "ranking_item"
            family.metadata["target_rank"] = int(decision.target_rank or 1)
            family.metadata["rank_mode"] = (
                "cumulative" if decision.strategy == ValidationStrategy.RANK_CUMULATIVE else "exact"
            )
            # Allow the highest captured raw rank, not merely the current displayed threshold.
            populated = []
            return family

        if decision.strategy == ValidationStrategy.GRID_SINGLE_SELECT:
            row_number = assigned_row or (occurrence_index + 1)
            wanted = re.compile(rf"^{re.escape(qid)}_?r{row_number}$", re.I)
            columns = [c for c in raw_columns if wanted.match(c)]
            return self._temporary_family(
                question_id,
                columns,
                QuestionType.SINGLE_SELECT,
                "single_select_grid_row",
                {"grid_row": row_number},
            )

        if decision.strategy == ValidationStrategy.GRID_MULTI_SELECT:
            row_number = assigned_row or (occurrence_index + 1)
            wanted = re.compile(rf"^{re.escape(qid)}_?r{row_number}_?c(\d+)$", re.I)
            columns = sorted(
                [c for c in raw_columns if wanted.match(c)],
                key=lambda c: int(wanted.match(c).group(1)),
            )
            return self._temporary_family(
                question_id,
                columns,
                QuestionType.MULTI_SELECT,
                "multi_select_grid_row",
                {"grid_row": row_number},
            )

        if decision.strategy in {ValidationStrategy.TOP_2_BOX, ValidationStrategy.BOTTOM_2_BOX}:
            columns = self._find_row_columns(question_id, raw_columns)
            return self._temporary_family(
                question_id,
                columns,
                QuestionType.RATING_SCALE,
                "derived_box",
                {
                    "box_mode": "top2" if decision.strategy == ValidationStrategy.TOP_2_BOX else "bottom2"
                },
            )

        if len(exact) == 1:
            family = exact[0]
            # A displayed option table backed by one physical column is
            # a direct coded frequency table, even when raw-value heuristics
            # labelled the column Numeric Entry or Rating Scale.
            if len(family.column_names) == 1:
                temporary = copy.deepcopy(family)
                temporary.detected_type = QuestionType.SINGLE_SELECT
                temporary.confirmed_type = QuestionType.SINGLE_SELECT
                temporary.structural_type = "single_select"
                return temporary
            return family
        return None

    def _build_grid_assignments(self, groups, strategy_map, raw):
        assignments = {}
        for qid, tables in groups.items():
            if not tables:
                continue
            strategy = strategy_map[tables[0].table_number].strategy
            if strategy not in {ValidationStrategy.GRID_SINGLE_SELECT, ValidationStrategy.GRID_MULTI_SELECT}:
                continue
            rows = self._available_grid_rows(qid, list(raw.columns), strategy)
            if not rows:
                continue
            costs = []
            for table in tables:
                costs.append([self._grid_match_cost(table, raw, qid, row, strategy) for row in rows])
            try:
                from scipy.optimize import linear_sum_assignment
                row_idx, col_idx = linear_sum_assignment(costs)
                for i, j in zip(row_idx, col_idx):
                    assignments[tables[int(i)].table_number] = rows[int(j)]
            except Exception:
                unused = set(rows)
                for table, row_costs in zip(tables, costs):
                    choices = [(cost, row) for cost, row in zip(row_costs, rows) if row in unused]
                    if choices:
                        _, selected = min(choices)
                        assignments[table.table_number] = selected
                        unused.remove(selected)
        return assignments

    @staticmethod
    def _available_grid_rows(qid, columns, strategy):
        if strategy == ValidationStrategy.GRID_SINGLE_SELECT:
            pattern = re.compile(rf"^{re.escape(qid)}_?r(\d+)$", re.I)
        else:
            pattern = re.compile(rf"^{re.escape(qid)}_?r(\d+)_?c\d+$", re.I)
        return sorted({int(pattern.match(c).group(1)) for c in columns if pattern.match(c)})

    @staticmethod
    def _grid_match_cost(table, raw, qid, row, strategy):
        reported = [o.reported_percentage for o in table.options]
        if strategy == ValidationStrategy.GRID_SINGLE_SELECT:
            pattern = re.compile(rf"^{re.escape(qid)}_?r{row}$", re.I)
            column = next((c for c in raw.columns if pattern.match(c)), None)
            if column is None:
                return 1e9
            series = raw[column]
            numeric = __import__('pandas').to_numeric(series, errors='coerce')
            base = int(numeric.notna().sum())
            calculated = [100 * int((numeric == i + 1).sum()) / base if base else 0 for i in range(len(reported))]
        else:
            pattern = re.compile(rf"^{re.escape(qid)}_?r{row}_?c(\d+)$", re.I)
            columns = sorted([c for c in raw.columns if pattern.match(c)], key=lambda c: int(pattern.match(c).group(1)))
            base = len(raw)
            calculated = [100 * int((__import__('pandas').to_numeric(raw[c], errors='coerce') == 1).sum()) / base for c in columns]
        diffs = [abs(float(r) - float(c)) for r, c in zip(reported, calculated) if r is not None]
        profile_cost = sum(diffs) / len(diffs) if diffs else 1000
        base_cost = 0
        if table.total_respondents is not None:
            base_cost = abs(base - int(table.total_respondents)) * 2
        return profile_cost + base_cost

    def _find_row_columns(self, question_id: str, raw_columns: list[str]) -> list[str]:
        qid = self._normalize_question_id(question_id)
        pattern = re.compile(rf"^{re.escape(qid)}_?r(\d+)$", re.I)
        matches = [(int(pattern.match(c).group(1)), c) for c in raw_columns if pattern.match(c)]
        return [c for _, c in sorted(matches)]

    @staticmethod
    def _temporary_family(name, columns, qtype, structural_type, metadata):
        if not columns:
            return None
        infos = [
            ColumnInfo(index=i, name=column, dtype="unknown", family=name)
            for i, column in enumerate(columns)
        ]
        return QuestionFamily(
            name=name,
            columns=infos,
            detected_type=qtype,
            confirmed_type=qtype,
            confidence=1.0,
            reason="Created by validation strategy engine.",
            grouping_method="validation_strategy",
            structural_type=structural_type,
            source_family_name=name,
            metadata=metadata,
        )

    @staticmethod
    def _select_analysis_base(raw, tables):
        """Restrict validation to the completion status matching the modal reported base."""
        reported = [int(t.total_respondents) for t in tables if t.total_respondents]
        if not reported or "sys_RespStatus" not in raw.columns:
            return raw, f"all {len(raw)} raw records (no completion-status match available)."
        modal_base = Counter(reported).most_common(1)[0][0]
        counts = raw["sys_RespStatus"].value_counts(dropna=False)
        exact = [value for value, count in counts.items() if int(count) == modal_base]
        if len(exact) == 1:
            status = exact[0]
            filtered = raw[raw["sys_RespStatus"] == status].copy()
            return filtered, f"{len(filtered)} completed records selected where sys_RespStatus = {status}; this matches the modal reported base {modal_base}."
        return raw, f"all {len(raw)} raw records; no unique sys_RespStatus count matched modal reported base {modal_base}."

    @staticmethod
    def _normalize_question_id(value: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
