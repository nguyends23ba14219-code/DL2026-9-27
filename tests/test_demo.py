from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_missing_artifacts_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("P27_OUTPUT_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "demo/app.py")).run(timeout=30)
    assert not app.exception
    assert app.info[0].value == "Chưa có kết quả cho lựa chọn này."
