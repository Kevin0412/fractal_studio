"""T15: Platform browser API uses real C++ Compute and ingests its private artifact."""

from __future__ import annotations

import os

import httpx
import pytest

from tests.e2e.release_helpers import become_creator, create_ready_asset, register, release_api_url


pytestmark = pytest.mark.skipif(
    not os.getenv("E2E_REAL_COMPUTE_PLATFORM"),
    reason="set E2E_REAL_COMPUTE_PLATFORM for the real Compute Platform gate",
)


@pytest.mark.asyncio
async def test_platform_render_ingests_real_compute_artifact_without_leakage() -> None:
    async with httpx.AsyncClient(base_url=release_api_url(), timeout=90, trust_env=False) as client:
        await register(client, label="real-compute")
        await become_creator(client, label="realcompute")
        asset_id = await create_ready_asset(client, label="real-compute")

        asset = await client.get(f"/v1/me/assets/{asset_id}")
        assert asset.status_code == 200, asset.text
        payload = asset.json()["data"]
        assert payload["id"] == asset_id and payload["status"] == "ready"
        assert all(value not in asset.text for value in ("objectKey", "computeRunId", "artifactId", "sha256"))
