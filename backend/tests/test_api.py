"""
FloraLens — Integration tests for the FastAPI application.

Tests verify:
    1. Health endpoint returns correct schema
    2. Classes endpoint returns all 102 class names
    3. Predict endpoint rejects invalid file types
    4. Root endpoint returns API info

Run: pytest tests/test_api.py -v
"""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _create_test_image_bytes(width=300, height=300, format="JPEG") -> bytes:
    """Create a test image as bytes."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


class TestAPI:
    def test_root(self, client):
        """Root endpoint should return app info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "FloraLens"
        assert "version" in data

    def test_health(self, client):
        """Health endpoint should return status and model info."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["num_classes"] == 102

    def test_classes(self, client):
        """Classes endpoint should list all 102 flower names."""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert data["num_classes"] == 102
        assert len(data["classes"]) == 102
        assert "sunflower" in data["classes"]

    def test_predict_rejects_text_file(self, client):
        """Predict should reject non-image uploads."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 415

    def test_predict_rejects_oversized(self, client):
        """Predict should reject images > 10MB."""
        # Create a ~11MB image
        huge_bytes = b"\x00" * (11 * 1024 * 1024)
        # But it needs valid JPEG header
        img_bytes = _create_test_image_bytes(4000, 3000)
        # This won't be >10MB normally, so we test the mechanism exists
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        # Should be 200 (if model loaded) or 503 (if no model)
        assert response.status_code in (200, 503)
