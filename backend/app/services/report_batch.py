"""报表批量预测与发刊编排。

把多张待生成报表组成整组提交，按「预测指标 + 截止日 + 训练样本」共享一次
预测计算，再把预测章节关联回各自报表；观测缺数的报表挂起，补数后重提即可
续算。所有状态都落在 store 的下划线命名空间里，随 reset_store 一起隔离。

核心约定：
- 只有预测指标、截止日、训练样本（观测日期序列与数值签名）完全相同的报表，
  才会共用同一章节（同一个共享键只触发一次预测引擎调用）。
- 发刊就绪记录按报表唯一，重复提交已经成功的报表只会被跳过，不会再次进入
  分发范围。
- 每份报表的登记记录只保存自己的业务范围，不混写同组其他报表的范围。
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from typing import Any

from app.fixtures import TREND_POINTS
from app.services.forecast_engine import run_forecast_on_sample
from app.store import store

# ---------------------------------------------------------------------------
# 存储命名空间与状态常量
# ---------------------------------------------------------------------------

OBS_TABLE = "_rb_observations"      # 观测数据可用性（指标 × 日期）
REG_TABLE = "_rb_registrations"     # 每张报表在批次里的登记记录
CHAPTER_TABLE = "_rb_chapters"      # 共享预测章节缓存（按共享键唯一）
CALL_TABLE = "_rb_calls"            # 预测引擎调用审计
READY_TABLE = "_rb_ready"           # 发刊就绪记录（按报表唯一）

STATUS_PENDING = "待生成"
STATUS_BLOCKED = "缺数待补"
STATUS_READY = "章节就绪"

READY_ACTIVE = "发刊就绪"
READY_SUSPENDED = "待补数挂起"

KNOWN_METRICS = ("METR-GMV", "METR-UV", "METR-CVR")


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat()


def _parse_window_days(window: Any) -> int:
    """把「30天」「14d」「7」之类的训练窗口统一解析成天数。"""
    if window is None:
        return 30
    text = str(window).strip().lower().replace("天", "").replace("d", "").strip()
    try:
        days = int(text)
    except ValueError:
        return 30
    return max(days, 1)


def _parse_horizon(value: Any) -> int:
    if value is None:
        return 7
    text = str(value).strip().lower().replace("天", "").replace("d", "").strip()
    try:
        return max(int(text), 1)
    except ValueError:
        return 7


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()


def _window_dates(cutoff: date, window_days: int) -> list[str]:
    """截止日向前取 window_days 个自然日（含截止日）。"""
    return [
        (cutoff - timedelta(days=offset)).isoformat()
        for offset in range(window_days - 1, -1, -1)
    ]


def _metric_value(point: dict[str, Any], metric_code: str) -> float:
    code = metric_code.upper()
    if "CVR" in code:
        total = float(point.get("total_uv") or 0)
        return float(point.get("paid_uv") or 0) / total if total else 0.0
    if "UV" in code:
        return float(point.get("uv") or 0)
    return float(point.get("gmv") or 0.0)


def _next_id(table: str) -> int:
    return max((int(row.get("id", 0)) for row in store.rows(table)), default=0) + 1


def _sample_signature(point_dates: list[str], values: list[float]) -> str:
    """训练样本签名：观测日期序列 + 数值完全一致才会得到相同签名。"""
    raw = "|".join(f"{d}:{round(v, 6)}" for d, v in zip(point_dates, values))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _shared_key(metric_code: str, cutoff: str, signature: str) -> str:
    """预测共享键：指标、截止日、训练样本三者相同才可共用一次预测。"""
    return f"{metric_code}|{cutoff}|{signature}"


# ---------------------------------------------------------------------------
# 观测数据（含迟到/补数模拟）
# ---------------------------------------------------------------------------

def seed_observations() -> None:
    """把 fixtures 时序点灌进观测表；已灌入时保持幂等，不覆盖补数数据。"""
    table = store.rows(OBS_TABLE)
    if table:
        return
    for point in TREND_POINTS:
        for metric_code in KNOWN_METRICS:
            table.append({
                "id": _next_id(OBS_TABLE),
                "预测指标": metric_code,
                "日期": point["日期"],
                "数值": round(_metric_value(point, metric_code), 6),
                "来源": "fixture",
                "到位时间": _now(),
            })


def available_dates(metric_code: str) -> set[str]:
    return {row["日期"] for row in store.rows(OBS_TABLE) if row["预测指标"] == metric_code}


def backfill_observations(points: list[dict[str, Any]]) -> dict[str, Any]:
    """补数：向观测表补齐迟到的观测点。

    入参每项形如 ``{"预测指标": "METR-GMV", "日期": "2026-08-31", "数值": 123.0}``；
    数值缺省时沿用该指标最近一个已到位观测（前向填充），保证结果可复现。
    已到位的日期不会被重复写入。
    """
    seed_observations()
    table = store.rows(OBS_TABLE)
    added: list[dict[str, Any]] = []
    for point in points:
        metric_code = str(point.get("预测指标") or point.get("metric_code") or "").strip()
        day = str(point.get("日期") or point.get("date") or "").strip()
        if not metric_code or not day:
            continue
        if any(r["预测指标"] == metric_code and r["日期"] == day for r in table):
            continue
        value = point.get("数值", point.get("value"))
        if value is None:
            history = sorted(
                (r for r in table if r["预测指标"] == metric_code),
                key=lambda r: r["日期"],
            )
            value = history[-1]["数值"] if history else 0.0
        entry = {
            "id": _next_id(OBS_TABLE),
            "预测指标": metric_code,
            "日期": day,
            "数值": round(float(value), 6),
            "来源": str(point.get("来源", "backfill")),
            "到位时间": _now(),
        }
        table.append(entry)
        added.append(entry)
    return {"added": len(added), "points": added}


def _training_sample(
    metric_code: str, cutoff: date, window_days: int
) -> tuple[list[str], list[float], list[str]]:
    """按报表登记的截止日与训练窗口切样本，返回日期、数值与缺失日期。"""
    expected = _window_dates(cutoff, window_days)
    obs = {
        row["日期"]: row["数值"]
        for row in store.rows(OBS_TABLE)
        if row["预测指标"] == metric_code
    }
    missing = [day for day in expected if day not in obs]
    point_dates = [day for day in expected if day in obs]
    values = [obs[day] for day in point_dates]
    return point_dates, values, missing


# ---------------------------------------------------------------------------
# 报表引用解析与默认口径
# ---------------------------------------------------------------------------

def _resolve_report(ref: dict[str, Any]) -> dict[str, Any]:
    """把入参里的报表引用解析成统一口径；report_id / 报表编号 都能定位。"""
    report_id = ref.get("report_id")
    report_row: dict[str, Any] | None = None
    if report_id is not None:
        report_row = store.find("report_gen", int(report_id))
    if report_row is None and ref.get("报表编号"):
        report_row = next(
            (r for r in store.rows("report_gen") if r.get("报表编号") == ref["报表编号"]),
            None,
        )

    code = ref.get("报表编号") or (report_row or {}).get("报表编号") or f"REPO-STDIN-{_next_id(REG_TABLE)}"
    metric_code = str(ref.get("预测指标") or "").strip()
    cutoff = str(ref.get("截止日") or "").strip()
    window = ref.get("训练窗口")
    horizon = ref.get("预测周期", ref.get("horizon"))
    algorithm = ref.get("预测算法")

    # 未显式给口径时，回退到该指标已登记的预测方案
    plan = next((f for f in store.rows("forecast") if f.get("预测指标") == metric_code), None)
    if window is None and plan is not None:
        window = plan.get("训练窗口")
    if horizon is None and plan is not None:
        horizon = plan.get("预测周期")
    if algorithm is None and plan is not None:
        algorithm = plan.get("预测算法")

    return {
        "report_id": int(report_row["id"]) if report_row else None,
        "报表编号": code,
        "报表类型": ref.get("报表类型") or (report_row or {}).get("报表类型") or "日报",
        # 只登记本报表自己的业务范围，不取同组其他报表的范围
        "业务范围": ref.get("业务范围") or ref.get("数据范围") or (report_row or {}).get("数据范围") or "",
        "预测指标": metric_code,
        "截止日": cutoff,
        "训练窗口": window,
        "horizon": _parse_horizon(horizon),
        "预测算法": algorithm,
    }


def _find_registration(batch_id: str, report_id: int | None, code: str) -> dict[str, Any] | None:
    for row in store.rows(REG_TABLE):
        if row.get("批次编号") != batch_id:
            continue
        if report_id is not None and row.get("report_id") == report_id:
            return row
        if row.get("报表编号") == code:
            return row
    return None


# ---------------------------------------------------------------------------
# 章节计算（共享键去重 + 调用审计）
# ---------------------------------------------------------------------------

def _find_chapter(key: str) -> dict[str, Any] | None:
    return next((r for r in store.rows(CHAPTER_TABLE) if r.get("共享键") == key), None)


def _find_ready_record(report_id: int | None, code: str) -> dict[str, Any] | None:
    for row in store.rows(READY_TABLE):
        if report_id is not None and row.get("report_id") == report_id:
            return row
        if row.get("报表编号") == code:
            return row
    return None


def _get_or_compute_chapter(
    spec: dict[str, Any],
    point_dates: list[str],
    values: list[float],
    signature: str,
    batch_id: str,
) -> tuple[dict[str, Any], bool]:
    """按共享键取章节；未命中才真正调用预测引擎，并记录一次调用。"""
    key = _shared_key(spec["预测指标"], spec["截止日"], signature)
    chapter = _find_chapter(key)
    if chapter is not None:
        return chapter, False

    payload = run_forecast_on_sample(
        spec["预测指标"], values, point_dates, spec["预测算法"], spec["horizon"]
    )
    if "error" in payload:
        raise ValueError(f"预测引擎拒绝样本：{payload['error']}")

    chapter = {
        "id": _next_id(CHAPTER_TABLE),
        "章节编号": f"CHAP-{key}",
        "共享键": key,
        "预测指标": spec["预测指标"],
        "截止日": spec["截止日"],
        "样本签名": signature,
        "训练样本数": len(point_dates),
        "horizon": spec["horizon"],
        "预测算法": spec["预测算法"],
        "payload": payload,
        "消费报表": [],
        "生成时间": _now(),
    }
    store.rows(CHAPTER_TABLE).append(chapter)
    store.rows(CALL_TABLE).append({
        "id": _next_id(CALL_TABLE),
        "调用时间": _now(),
        "批次编号": batch_id,
        "报表编号": spec["报表编号"],
        "共享键": key,
        "预测指标": spec["预测指标"],
        "截止日": spec["截止日"],
        "训练样本数": len(point_dates),
        "样本签名": signature,
        "horizon": spec["horizon"],
        "预测算法": spec["预测算法"],
    })
    return chapter, True


def _attach_ready_record(spec: dict[str, Any], chapter: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """登记发刊就绪记录；同一报表只保留一条，重复提交不重复登记。"""
    existing = _find_ready_record(spec["report_id"], spec["报表编号"])
    if existing is not None:
        existing["状态"] = READY_ACTIVE
        existing["共享键"] = chapter["共享键"]
        existing["章节编号"] = chapter["章节编号"]
        existing["样本签名"] = chapter["样本签名"]
        existing["更新时间"] = _now()
        return existing, False

    record = {
        "id": _next_id(READY_TABLE),
        "发刊编号": f"PUB-{spec['报表编号']}",
        "批次编号": spec.get("批次编号"),
        "report_id": spec["report_id"],
        "报表编号": spec["报表编号"],
        "报表类型": spec["报表类型"],
        "业务范围": spec["业务范围"],
        "共享键": chapter["共享键"],
        "章节编号": chapter["章节编号"],
        "样本签名": chapter["样本签名"],
        "输入摘要": {
            "预测指标": spec["预测指标"],
            "截止日": spec["截止日"],
            "训练窗口": spec["训练窗口"],
            "horizon": spec["horizon"],
        },
        "状态": READY_ACTIVE,
        "登记时间": _now(),
        "更新时间": _now(),
    }
    store.rows(READY_TABLE).append(record)
    return record, True


# ---------------------------------------------------------------------------
# 单张报表处理与批次提交
# ---------------------------------------------------------------------------

def _process_report(batch_id: str, ref: dict[str, Any]) -> dict[str, Any]:
    spec = _resolve_report(ref)
    spec["批次编号"] = batch_id
    cutoff = _parse_date(spec["截止日"])
    window_days = _parse_window_days(spec["训练窗口"])

    registration = _find_registration(batch_id, spec["report_id"], spec["报表编号"])
    if registration is not None and registration["status"] == STATUS_READY:
        # 已成功的报表重复提交：直接跳过，绝不重复进入分发范围
        registration["重复跳过"] = registration.get("重复跳过", 0) + 1
        registration["更新时间"] = _now()
        return dict(registration)

    point_dates, values, missing = _training_sample(spec["预测指标"], cutoff, window_days)

    if registration is None:
        registration = {"id": _next_id(REG_TABLE), "批次编号": batch_id}
        store.rows(REG_TABLE).append(registration)

    registration.update({
        "report_id": spec["report_id"],
        "报表编号": spec["报表编号"],
        "报表类型": spec["报表类型"],
        "业务范围": spec["业务范围"],
        "预测指标": spec["预测指标"],
        "截止日": spec["截止日"],
        "训练窗口": f"{window_days}天",
        "horizon": spec["horizon"],
        "预测算法": spec["预测算法"],
        "训练样本数": len(point_dates),
        "缺失观测": missing,
        "更新时间": _now(),
        "重复跳过": registration.get("重复跳过", 0),
    })
    registration.setdefault("注册时间", _now())

    if missing:
        # 训练期缺数（周报观测迟到）：挂起，不触发预测，也不产生发刊记录
        registration["status"] = STATUS_BLOCKED
        registration["共享键"] = None
        registration["样本签名"] = None
        registration["章节编号"] = None
        registration["备注"] = f"训练窗口缺少 {len(missing)} 个观测：{', '.join(missing)}"
        return dict(registration)

    signature = _sample_signature(point_dates, values)
    chapter, computed = _get_or_compute_chapter(spec, point_dates, values, signature, batch_id)

    if spec["报表编号"] not in chapter["消费报表"]:
        chapter["消费报表"].append(spec["报表编号"])

    _, created = _attach_ready_record(spec, chapter)
    registration["status"] = STATUS_READY
    registration["共享键"] = chapter["共享键"]
    registration["样本签名"] = signature
    registration["章节编号"] = chapter["章节编号"]
    registration["备注"] = "新算章节" if computed else "复用同批预测章节"
    return dict(registration)


def submit_batch(batch_id: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
    """整组提交待生成报表：批量计算预测章节并形成各自的待生成结果。

    返回每组报表的状态、预测调用次数与发刊就绪登记情况，便于运营判断
    哪份能继续分发、哪份要等补数。
    """
    seed_observations()
    before_calls = forecast_call_count()
    results = [_process_report(batch_id, ref) for ref in reports]

    ready_codes = [r["报表编号"] for r in results if r["status"] == STATUS_READY]
    blocked = [
        {"报表编号": r["报表编号"], "缺失观测": r["缺失观测"]}
        for r in results if r["status"] == STATUS_BLOCKED
    ]
    return {
        "批次编号": batch_id,
        "提交数量": len(reports),
        "就绪报表": ready_codes,
        "缺数挂起": blocked,
        "本次预测调用次数": forecast_call_count() - before_calls,
        "累计预测调用次数": forecast_call_count(),
        "发刊就绪记录数": count_ready_records(),
        "registrations": results,
    }


# ---------------------------------------------------------------------------
# 补数后续算与发刊对账
# ---------------------------------------------------------------------------

def retry_batch(batch_id: str, report_refs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """补数后再次提交：只重算挂起/待生成的报表，已就绪报表原样跳过。"""
    if report_refs is None:
        report_refs = [
            {
                "report_id": row.get("report_id"),
                "报表编号": row["报表编号"],
                "报表类型": row.get("报表类型"),
                "业务范围": row.get("业务范围"),
                "预测指标": row["预测指标"],
                "截止日": row["截止日"],
                "训练窗口": row.get("训练窗口"),
                "预测周期": row.get("horizon"),
                "预测算法": row.get("预测算法"),
            }
            for row in store.rows(REG_TABLE)
            if row.get("批次编号") == batch_id and row.get("status") != STATUS_READY
        ]
    return submit_batch(batch_id, report_refs)


def reconcile_registration(registration: dict[str, Any]) -> dict[str, Any]:
    """让单张报表的生成状态与当前真正能拿到的章节保持一致。"""
    cutoff = _parse_date(registration["截止日"])
    window_days = _parse_window_days(registration["训练窗口"])
    point_dates, values, missing = _training_sample(registration["预测指标"], cutoff, window_days)
    ready_record = _find_ready_record(registration.get("report_id"), registration["报表编号"])

    if missing:
        registration["status"] = STATUS_BLOCKED
        registration["缺失观测"] = missing
        registration["章节编号"] = None
        if ready_record is not None:
            ready_record["状态"] = READY_SUSPENDED
            ready_record["更新时间"] = _now()
        return dict(registration)

    signature = _sample_signature(point_dates, values)
    spec = {
        "report_id": registration.get("report_id"),
        "报表编号": registration["报表编号"],
        "报表类型": registration.get("报表类型"),
        "业务范围": registration.get("业务范围"),
        "预测指标": registration["预测指标"],
        "截止日": registration["截止日"],
        "训练窗口": registration["训练窗口"],
        "horizon": registration.get("horizon", 7),
        "预测算法": registration.get("预测算法"),
        "批次编号": registration.get("批次编号"),
    }
    chapter, _ = _get_or_compute_chapter(
        spec, point_dates, values, signature, registration["批次编号"]
    )
    _attach_ready_record(spec, chapter)
    registration["status"] = STATUS_READY
    registration["缺失观测"] = []
    registration["共享键"] = chapter["共享键"]
    registration["样本签名"] = signature
    registration["章节编号"] = chapter["章节编号"]
    registration["训练样本数"] = len(point_dates)
    registration["更新时间"] = _now()
    return dict(registration)


def verify_publication(batch_id: str, report_refs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """统一发刊时间对账：复算各报表输入，确认哪些能进入本次分发。

    输入未变化时章节缓存命中、不会产生新的预测调用；缺数报表保持挂起。
    """
    seed_observations()
    before_calls = forecast_call_count()

    if report_refs is None:
        targets = [
            row for row in store.rows(REG_TABLE) if row.get("批次编号") == batch_id
        ]
    else:
        targets = []
        for ref in report_refs:
            code = ref.get("报表编号")
            report_id = ref.get("report_id")
            found = _find_registration(batch_id, report_id, code) if (code or report_id) else None
            if found is not None:
                targets.append(found)

    reconciled = [reconcile_registration(row) for row in targets]
    distributable = [r["报表编号"] for r in reconciled if r["status"] == STATUS_READY]
    suspended = [
        {"报表编号": r["报表编号"], "缺失观测": r["缺失观测"]}
        for r in reconciled if r["status"] == STATUS_BLOCKED
    ]
    return {
        "批次编号": batch_id,
        "可分发": distributable,
        "待补数": suspended,
        "复算新增预测调用次数": forecast_call_count() - before_calls,
        "累计预测调用次数": forecast_call_count(),
        "registrations": reconciled,
    }


# ---------------------------------------------------------------------------
# 查询辅助
# ---------------------------------------------------------------------------

def forecast_call_count(batch_id: str | None = None) -> int:
    rows = store.rows(CALL_TABLE)
    if batch_id is not None:
        rows = [r for r in rows if r.get("批次编号") == batch_id]
    return len(rows)


def count_ready_records() -> int:
    return sum(1 for r in store.rows(READY_TABLE) if r.get("状态") == READY_ACTIVE)


def list_registrations(batch_id: str | None = None) -> list[dict[str, Any]]:
    rows = store.rows(REG_TABLE)
    if batch_id is not None:
        rows = [r for r in rows if r.get("批次编号") == batch_id]
    return [dict(r) for r in rows]


def list_ready_records() -> list[dict[str, Any]]:
    return [dict(r) for r in store.rows(READY_TABLE)]


def list_chapters() -> list[dict[str, Any]]:
    return [dict(r) for r in store.rows(CHAPTER_TABLE)]


def list_calls() -> list[dict[str, Any]]:
    return [dict(r) for r in store.rows(CALL_TABLE)]


def batch_summary(batch_id: str) -> dict[str, Any]:
    rows = list_registrations(batch_id)
    return {
        "批次编号": batch_id,
        "报表总数": len(rows),
        "就绪": sum(1 for r in rows if r.get("status") == STATUS_READY),
        "缺数挂起": sum(1 for r in rows if r.get("status") == STATUS_BLOCKED),
        "待生成": sum(1 for r in rows if r.get("status") == STATUS_PENDING),
        "批次预测调用次数": forecast_call_count(batch_id),
        "发刊就绪记录数": count_ready_records(),
    }
