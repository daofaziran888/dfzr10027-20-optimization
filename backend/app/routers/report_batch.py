"""报表批量预测接口：整组提交、补数续算、发刊对账与审计查询。"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.schemas import ActionResult
from app.services import report_batch

router = APIRouter(prefix="/api/report-batch", tags=["报表批量预测"])


class BatchSubmitPayload(BaseModel):
    batch_id: str = Field(..., description="批次编号，同一批次重复提交按报表幂等")
    reports: list[dict] = Field(default_factory=list, description="待生成报表清单，每张只登记自己的业务范围")


class BackfillPayload(BaseModel):
    points: list[dict] = Field(
        default_factory=list,
        description="迟到观测点：预测指标、日期、数值（缺省时按最近观测前向填充）",
    )


class PublicationPayload(BaseModel):
    batch_id: str
    reports: list[dict] | None = Field(
        default=None, description="留空则对账该批次全部登记报表",
    )


@router.post("/submit", response_model=ActionResult)
def submit_batch(payload: BatchSubmitPayload) -> ActionResult:
    """整组提交报表：共享键去重计算预测章节，缺数报表挂起等待补数。"""
    if not payload.reports:
        return ActionResult(ok=False, message="批次内没有待生成报表")
    result = report_batch.submit_batch(payload.batch_id, payload.reports)
    return ActionResult(ok=True, message="批次已完成批量计算", entry=result)


@router.post("/backfill", response_model=ActionResult)
def backfill(payload: BackfillPayload) -> ActionResult:
    """补齐迟到观测，补数后可对原批次再次提交。"""
    result = report_batch.backfill_observations(payload.points)
    return ActionResult(ok=True, message=f"已补 {result['added']} 个观测点", entry=result)


@router.post("/retry", response_model=ActionResult)
def retry_batch(payload: BatchSubmitPayload) -> ActionResult:
    """补数后再次提交：已就绪报表不会被重复登记，只续算挂起报表。"""
    result = report_batch.retry_batch(payload.batch_id, payload.reports or None)
    return ActionResult(ok=True, message="补数后续算完成", entry=result)


@router.post("/verify-publication", response_model=ActionResult)
def verify_publication(payload: PublicationPayload) -> ActionResult:
    """统一发刊时间对账：复算各报表输入，给出可分发与待补数清单。"""
    result = report_batch.verify_publication(payload.batch_id, payload.reports)
    return ActionResult(ok=True, message="发刊对账完成", entry=result)


@router.get("/registrations")
def registrations(batch_id: str | None = None) -> dict[str, object]:
    """查询批次内各报表的登记与生成状态。"""
    return {"items": report_batch.list_registrations(batch_id)}


@router.get("/chapters")
def chapters() -> dict[str, object]:
    """查询共享预测章节及其消费报表。"""
    return {"items": report_batch.list_chapters()}


@router.get("/calls")
def calls(batch_id: str | None = None) -> dict[str, object]:
    """查询预测引擎调用审计（调用次数与每次的共享键、样本签名）。"""
    return {
        "items": report_batch.list_calls() if batch_id is None
        else [r for r in report_batch.list_calls() if r.get("批次编号") == batch_id],
        "total": report_batch.forecast_call_count(batch_id),
    }


@router.get("/ready-records")
def ready_records() -> dict[str, object]:
    """查询发刊就绪记录（按报表唯一）。"""
    return {"items": report_batch.list_ready_records()}


@router.get("/summary/{batch_id}")
def summary(batch_id: str) -> dict[str, object]:
    """批次摘要：就绪/挂起数量、预测调用次数、就绪记录数。"""
    return report_batch.batch_summary(batch_id)
