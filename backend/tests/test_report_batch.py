"""报表批量预测与发刊编排测试。

完整走查需求场景：
1. 批量提交两张同指标/同截止日的日报 + 一张训练期缺数的周报；
2. 两张日报共用一次预测，周报因观测迟到挂起；
3. 记录预测调用次数与发刊就绪记录；
4. 补数后再次提交，周报续算、原有日报不被重复登记；
5. 统一发刊时间复算，输入不变时零新增调用。
"""
from __future__ import annotations

import pytest

from app.services import report_batch as rb
from app.services.forecast_engine import run_forecast_for_metric

BATCH_ID = "BATCH-20260901"
CUTOFF = "2026-08-30"
METRIC = "METR-GMV"


def _daily_report(code: str, scope: str) -> dict:
    """两张日报：同一指标、同一截止日、同一训练窗口，业务范围各自登记。"""
    return {
        "报表编号": code,
        "报表类型": "日报",
        "业务范围": scope,
        "预测指标": METRIC,
        "截止日": CUTOFF,
        "训练窗口": "30天",
        "预测周期": "7天",
    }


def _weekly_report() -> dict:
    """周报：同指标同截止日，但训练窗口 40 天，fixture 只有 30 天观测，必然缺数。"""
    return {
        "报表编号": "REPO-WEEKLY-GMV",
        "报表类型": "周报",
        "业务范围": "2026-07-22至2026-08-30",
        "预测指标": METRIC,
        "截止日": CUTOFF,
        "训练窗口": "40天",
        "预测周期": "14天",
    }


@pytest.fixture()
def initial_batch():
    """首次整组提交：两张日报 + 一张缺数周报。"""
    return rb.submit_batch(BATCH_ID, [
        _daily_report("REPO-DAILY-A", "华东区"),
        _daily_report("REPO-DAILY-B", "华南区"),
        _weekly_report(),
    ])


class TestInitialBatchSubmission:
    """首次批量提交：共享预测、缺数挂起、就绪登记。"""

    def test_daily_reports_ready_weekly_blocked(self, initial_batch):
        assert set(initial_batch["就绪报表"]) == {"REPO-DAILY-A", "REPO-DAILY-B"}
        assert len(initial_batch["缺数挂起"]) == 1
        blocked = initial_batch["缺数挂起"][0]
        assert blocked["报表编号"] == "REPO-WEEKLY-GMV"
        # 40 天窗口只有 30 天观测，应缺 10 天
        assert len(blocked["缺失观测"]) == 10
        assert "2026-07-22" in blocked["缺失观测"]

    def test_forecast_called_only_once_for_shared_inputs(self, initial_batch):
        # 两张日报指标、截止日、训练样本完全相同 -> 只允许一次预测调用
        assert initial_batch["本次预测调用次数"] == 1
        assert rb.forecast_call_count() == 1
        chapters = rb.list_chapters()
        assert len(chapters) == 1
        # 同一份章节被两张日报消费
        assert set(chapters[0]["消费报表"]) == {"REPO-DAILY-A", "REPO-DAILY-B"}

    def test_ready_records_one_per_report(self, initial_batch):
        # 周报缺数不产生发刊就绪记录；日报各一条
        assert initial_batch["发刊就绪记录数"] == 2
        records = rb.list_ready_records()
        codes = {r["报表编号"] for r in records}
        assert codes == {"REPO-DAILY-A", "REPO-DAILY-B"}
        for record in records:
            assert record["状态"] == rb.READY_ACTIVE
            # 每份报表只登记自己的业务范围
            assert record["业务范围"] in {"华东区", "华南区"}

    def test_blocked_report_has_no_chapter_and_no_call(self, initial_batch):
        calls = rb.list_calls()
        assert all(call["报表编号"] != "REPO-WEEKLY-GMV" for call in calls)
        weekly = next(
            r for r in rb.list_registrations(BATCH_ID)
            if r["报表编号"] == "REPO-WEEKLY-GMV"
        )
        assert weekly["status"] == rb.STATUS_BLOCKED
        assert weekly["章节编号"] is None

    def test_registration_keeps_own_scope_only(self, initial_batch):
        registrations = {r["报表编号"]: r for r in rb.list_registrations(BATCH_ID)}
        assert registrations["REPO-DAILY-A"]["业务范围"] == "华东区"
        assert registrations["REPO-DAILY-B"]["业务范围"] == "华南区"
        assert registrations["REPO-WEEKLY-GMV"]["业务范围"] == "2026-07-22至2026-08-30"


class TestRepeatedSubmissionIdempotency:
    """重复提交：已经成功的日报不能再次进入分发范围。"""

    def test_resubmit_does_not_duplicate_ready_records(self, initial_batch):
        before_records = len(rb.list_ready_records())
        before_calls = rb.forecast_call_count()

        result = rb.submit_batch(BATCH_ID, [
            _daily_report("REPO-DAILY-A", "华东区"),
            _daily_report("REPO-DAILY-B", "华南区"),
        ])

        # 没有新增预测调用，也没有新增就绪记录
        assert result["本次预测调用次数"] == 0
        assert rb.forecast_call_count() == before_calls
        assert len(rb.list_ready_records()) == before_records
        registrations = {r["报表编号"]: r for r in rb.list_registrations(BATCH_ID)}
        # 每张日报仍只有一条登记，且被标记为重复跳过
        assert registrations["REPO-DAILY-A"]["重复跳过"] >= 1
        assert registrations["REPO-DAILY-B"]["重复跳过"] >= 1


class TestBackfillAndRetry:
    """补数后续算：周报拿到章节，日报不被重复登记。"""

    def test_backfill_then_retry(self, initial_batch):
        missing = next(
            r["缺失观测"]
            for r in rb.list_registrations(BATCH_ID)
            if r["报表编号"] == "REPO-WEEKLY-GMV"
        )
        # 模拟迟到的 10 个观测点（数值前向填充由 backfill 兜底，这里显式给值）
        points = [
            {"预测指标": METRIC, "日期": day, "数值": 200000.0, "来源": "late_feed"}
            for day in missing
        ]
        backfill_result = rb.backfill_observations(points)
        assert backfill_result["added"] == 10

        calls_before = rb.forecast_call_count()
        records_before = len(rb.list_ready_records())

        # 补数后把原批次三张报表整体再次提交
        retry = rb.submit_batch(BATCH_ID, [
            _daily_report("REPO-DAILY-A", "华东区"),
            _daily_report("REPO-DAILY-B", "华南区"),
            _weekly_report(),
        ])

        # 周报这次就绪；日报仍是原来两张
        assert set(retry["就绪报表"]) == {
            "REPO-DAILY-A", "REPO-DAILY-B", "REPO-WEEKLY-GMV",
        }
        assert retry["缺数挂起"] == []
        # 只有周报触发了一次新预测（共享键不同：样本签名不同）
        assert retry["本次预测调用次数"] == 1
        assert rb.forecast_call_count() == calls_before + 1
        # 日报没有被重复登记：只新增周报一条就绪记录
        assert len(rb.list_ready_records()) == records_before + 1
        assert rb.count_ready_records() == 3

        chapters = rb.list_chapters()
        assert len(chapters) == 2
        weekly_chapter = next(
            c for c in chapters if "REPO-WEEKLY-GMV" in c["消费报表"]
        )
        assert weekly_chapter["训练样本数"] == 40
        weekly_reg = next(
            r for r in rb.list_registrations(BATCH_ID)
            if r["报表编号"] == "REPO-WEEKLY-GMV"
        )
        # 生成状态与真正拿到的章节一致
        assert weekly_reg["status"] == rb.STATUS_READY
        assert weekly_reg["章节编号"] == weekly_chapter["章节编号"]

    def test_backfill_is_idempotent(self, initial_batch):
        first = rb.backfill_observations([
            {"预测指标": METRIC, "日期": "2026-07-22", "数值": 1.0},
        ])
        second = rb.backfill_observations([
            {"预测指标": METRIC, "日期": "2026-07-22", "数值": 2.0},
        ])
        assert first["added"] == 1
        # 已到位的观测不会被覆盖或重复写入
        assert second["added"] == 0


class TestPublicationVerification:
    """统一发刊时间：日报输入可复算，迟到周报先挂起、补数后可分发。"""

    def test_verify_before_backfill_suspends_weekly(self, initial_batch):
        result = rb.verify_publication(BATCH_ID)
        assert set(result["可分发"]) == {"REPO-DAILY-A", "REPO-DAILY-B"}
        assert [item["报表编号"] for item in result["待补数"]] == ["REPO-WEEKLY-GMV"]
        # 输入未变化，复算全部命中章节缓存，零新增调用
        assert result["复算新增预测调用次数"] == 0

    def test_verify_after_backfill_all_distributable_without_extra_calls(self, initial_batch):
        missing = next(
            r["缺失观测"]
            for r in rb.list_registrations(BATCH_ID)
            if r["报表编号"] == "REPO-WEEKLY-GMV"
        )
        rb.backfill_observations([
            {"预测指标": METRIC, "日期": day, "数值": 210000.0} for day in missing
        ])
        rb.retry_batch(BATCH_ID)
        calls_before = rb.forecast_call_count()

        result = rb.verify_publication(BATCH_ID)
        assert set(result["可分发"]) == {
            "REPO-DAILY-A", "REPO-DAILY-B", "REPO-WEEKLY-GMV",
        }
        assert result["待补数"] == []
        # 发刊复算不应再产生预测调用
        assert result["复算新增预测调用次数"] == 0
        assert rb.forecast_call_count() == calls_before


class TestSharingRules:
    """共享口径：只有指标、截止日、训练样本都相同才共用一次预测。"""

    def test_different_cutoff_separates_forecast(self):
        rb.submit_batch("BATCH-SPLIT", [
            {**_daily_report("REPO-C1", "范围1"), "截止日": "2026-08-29", "训练窗口": "7天"},
            {**_daily_report("REPO-C2", "范围2"), "截止日": "2026-08-30", "训练窗口": "7天"},
        ])
        # 截止日不同 -> 训练样本不同 -> 两次调用、两份章节
        assert rb.forecast_call_count("BATCH-SPLIT") == 2
        assert len(rb.list_chapters()) == 2

    def test_different_metric_separates_forecast(self):
        rb.submit_batch("BATCH-METRIC", [
            {**_daily_report("REPO-GMV", "范围1"), "预测指标": "METR-GMV"},
            {**_daily_report("REPO-UV", "范围2"), "预测指标": "METR-UV"},
        ])
        assert rb.forecast_call_count("BATCH-METRIC") == 2

    def test_same_inputs_across_batches_reuse_cached_chapter(self):
        spec = _daily_report("REPO-CROSS", "跨批次范围")
        first = rb.submit_batch("BATCH-X", [spec])
        second = rb.submit_batch("BATCH-Y", [spec])
        assert first["本次预测调用次数"] == 1
        # 新批次同口径提交直接命中章节缓存
        assert second["本次预测调用次数"] == 0


class TestForecastEngineBackwardCompatibility:
    """引擎改造后旧入口行为保持不变。"""

    def test_legacy_entry_still_works(self):
        result = run_forecast_for_metric("METR-GMV", "线性趋势", 7)
        assert result["model"] == "linear_trend"
        assert len(result["predictions"]) == 7
        assert result["actual_count"] == 30
        assert result["forecast_dates"][0] == "2026-08-31"

    def test_sample_entry_rejects_short_sample(self):
        from app.services.forecast_engine import run_forecast_on_sample

        payload = run_forecast_on_sample(
            "METR-GMV", [1.0], ["2026-08-30"], "线性趋势", 7,
        )
        assert payload == {"error": "数据不足"}
