"""API-level integration tests (mocked pipeline, real HTTP + persistence)."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _jpeg_bytes(color=(200, 30, 30), size=(160, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _create_project(client, name="API Test") -> str:
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def test_health_and_capabilities(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/api/system/capabilities")
    caps = r.json()
    assert "modes" in caps and "dependencies" in caps
    assert isinstance(caps["modes"]["cpu_reconstruction"], bool)


def test_project_crud_and_persistence(client):
    pid = _create_project(client, "Persist Me")
    r = client.get(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert r.json()["name"] == "Persist Me"
    assert r.json()["status"] == "draft"

    r = client.get("/api/projects")
    assert any(p["id"] == pid for p in r.json())

    assert client.delete(f"/api/projects/{pid}").status_code == 204
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_upload_files_and_validation_report(client):
    pid = _create_project(client)
    files = [("files", (f"view_{i}.jpg", _jpeg_bytes(), "image/jpeg")) for i in range(4)]
    r = client.post(f"/api/projects/{pid}/images", files=files)
    assert r.status_code == 200
    report = r.json()
    assert len(report["added"]) == 4
    assert report["summary"]["image_count"] == 4

    detail = client.get(f"/api/projects/{pid}").json()
    assert detail["imageCount"] == 4
    assert len(detail["images"]) == 4
    # every image must expose a working preview URL (path guard verified)
    for img in detail["images"]:
        assert client.get(img["url"]).status_code == 200


def test_upload_rejects_unsupported_types(client):
    pid = _create_project(client)
    r = client.post(f"/api/projects/{pid}/images",
                    files=[("files", ("malware.exe", b"MZ...", "application/octet-stream"))])
    assert r.status_code == 200
    assert len(r.json()["added"]) == 0
    assert "unsupported file type" in r.json()["errors"][0]


def test_upload_zip_with_traversal_is_rejected(client):
    pid = _create_project(client)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../evil.jpg", _jpeg_bytes())
    r = client.post(f"/api/projects/{pid}/images",
                    files=[("files", ("data.zip", buf.getvalue(), "application/zip"))])
    assert r.status_code == 200
    body = r.json()
    assert len(body["added"]) == 0
    assert any("Unsafe path" in e for e in body["errors"])


def test_upload_valid_zip_of_images(client):
    pid = _create_project(client)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(3):
            zf.writestr(f"nested/sub/img{i}.jpg", _jpeg_bytes())
    r = client.post(f"/api/projects/{pid}/images",
                    files=[("files", ("scene.zip", buf.getvalue(), "application/zip"))])
    body = r.json()
    assert len(body["added"]) == 3


def test_artifact_route_rejects_unknown_and_traversal(client):
    pid = _create_project(client)
    r = client.get(f"/api/projects/{pid}/artifacts/sparse_cloud/nope.ply")
    assert r.status_code == 404
    r = client.get(f"/api/projects/{pid}/artifacts/sparse_cloud/..%2F..%2Fx.ply")
    assert r.status_code in (404, 400)


def test_reconstruct_blocked_without_images(client):
    pid = _create_project(client)
    r = client.post(f"/api/projects/{pid}/reconstruct", json={})
    assert r.status_code == 400


def test_job_lifecycle_with_mocked_pipeline(client, monkeypatch):
    """Mock-free HTTP + job manager; only the COLMAP pipeline is stubbed."""
    pid = _create_project(client)
    files = [("files", (f"v{i}.jpg", _jpeg_bytes(), "image/jpeg")) for i in range(5)]
    client.post(f"/api/projects/{pid}/images", files=files)

    from backend.app import db
    from backend.app.reconstruction.pipeline import PipelineError

    def fake_pipeline(ctx, project, params):
        ctx.stage("preprocessing", "fake")
        ctx.progress(0.5)
        ctx.log("fake pipeline running")
        from backend.app.config import settings
        art_dir = settings.project_dir(ctx.project_id) / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "sparse_cloud.ply").write_bytes(b"ply fake")
        db.add_artifact(ctx.project_id, "sparse_cloud", "sparse_cloud.ply", "ply", 8)
        if params.get("fail"):
            raise PipelineError("deliberate test failure")

    import backend.app.api.projects as projects_api

    monkeypatch.setattr(projects_api, "run_reconstruction", fake_pipeline)

    r = client.post(f"/api/projects/{pid}/reconstruct", json={"mode": "cpu"})
    assert r.status_code == 202

    # wait for the single worker to finish (bounded)
    import time

    for _ in range(100):
        st = client.get(f"/api/projects/{pid}/status").json()
        if st["project"]["status"] in ("completed", "failed", "canceled"):
            break
        time.sleep(0.05)
    assert st["project"]["status"] == "completed"
    assert st["job"]["status"] == "completed"
    assert st["job"]["progress"] == 1.0

    # artifacts + metrics + logs endpoints reflect real state
    arts = client.get(f"/api/projects/{pid}/artifacts").json()
    assert any(a["kind"] == "sparse_cloud" for a in arts)
    assert client.get(arts[0]["url"]).status_code == 200

    logs = client.get(f"/api/projects/{pid}/logs").json()
    assert any("fake pipeline" in entry["message"] for entry in logs["logs"])

    # double reconstruction guard
    monkeypatch.setattr(projects_api, "run_reconstruction",
                        lambda ctx, project, params: None)
    db.update_project(pid, status="processing")
    r = client.post(f"/api/projects/{pid}/reconstruct", json={})
    assert r.status_code == 409
    db.update_project(pid, status="completed")


def test_failed_job_surfaces_error(client, monkeypatch):
    pid = _create_project(client)
    files = [("files", (f"v{i}.jpg", _jpeg_bytes(), "image/jpeg")) for i in range(5)]
    client.post(f"/api/projects/{pid}/images", files=files)

    import backend.app.api.projects as projects_api
    from backend.app.reconstruction.pipeline import PipelineError

    def failing_pipeline(ctx, project, params):
        raise PipelineError("COLMAP is not installed (simulated)")

    monkeypatch.setattr(projects_api, "run_reconstruction", failing_pipeline)
    client.post(f"/api/projects/{pid}/reconstruct", json={"mode": "cpu"})

    import time

    st = {}
    for _ in range(100):
        st = client.get(f"/api/projects/{pid}/status").json()
        if st["project"]["status"] == "failed":
            break
        time.sleep(0.05)
    assert st["project"]["status"] == "failed"
    assert "COLMAP is not installed" in st["project"]["error"]


def test_missing_project_returns_404(client):
    assert client.get("/api/projects/doesnotexist").status_code == 404
    assert client.post("/api/projects/doesnotexist/reconstruct", json={}).status_code == 404


def test_logs_can_be_scoped_to_one_job(client):
    from backend.app import db

    pid = client.post("/api/projects", json={"name": "log scope"}).json()["id"]
    db.add_log(pid, "info", "old run", job_id="job-old")
    db.add_log(pid, "info", "new run", job_id="job-new")

    everything = client.get(f"/api/projects/{pid}/logs").json()["logs"]
    latest = client.get(f"/api/projects/{pid}/logs", params={"job_id": "job-new"}).json()["logs"]

    assert [entry["message"] for entry in everything] == ["old run", "new run"]
    assert [entry["message"] for entry in latest] == ["new run"]
