from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_project_uses_agpl_3_only_everywhere() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3, 19 November 2007" in license_text
    assert 'license = "AGPL-3.0-only"' in pyproject
    assert 'name = "acemilee"' in pyproject
    assert "https://github.com/acemilee/metasurface-paper-RAG" in pyproject
    assert "AGPL-3.0-only" in readme
    assert "GNU Affero General Public License v3" in readme
    assert "Apache License 2.0" not in readme


def test_pymupdf_license_boundary_is_disclosed() -> None:
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "PyMuPDF" in notices
    assert "GNU AGPL v3" in notices
    assert "Artifex Commercial License" in notices
    assert "本项目未取得或主张 Artifex 商业许可证" in notices


def test_public_security_and_contribution_entrypoints_exist() -> None:
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "GitHub Security Advisory" in security
    assert "不要在公开 Issue 中提交" in security
    assert "DeepSeek API 密钥" in security
    assert "AGPL-3.0-only" in contributing
    assert "-BuildLocal" in contributing
    assert "--build-local" in contributing
    assert "python -m pytest" in contributing
