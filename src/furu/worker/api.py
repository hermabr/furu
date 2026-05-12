from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException

from furu.worker.protocol import BlockedRequest, FinishRequest, GetJobResponse

if TYPE_CHECKING:
    from furu.execution.manager import Manager


def create_manager_app(manager: Manager) -> FastAPI:
    app = FastAPI()

    @app.get("/get_job", response_model=GetJobResponse)
    def get_job() -> GetJobResponse:
        return manager.get_job()

    @app.post("/finish/{lease_id}")
    def finish(lease_id: str, request: FinishRequest) -> dict[str, bool]:
        try:
            manager.finish(lease_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/blocked/{lease_id}")
    def blocked(lease_id: str, request: BlockedRequest) -> dict[str, bool]:
        try:
            manager.block(lease_id, request.dependencies)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True}

    return app
