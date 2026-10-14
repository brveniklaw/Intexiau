from pathlib import Path

from bailey_b.config import PROJECT_ROOT, CorpusSettings


def test_relative_data_dir_is_anchored_to_project_root():
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert CorpusSettings().data_dir == PROJECT_ROOT / "data"


def test_absolute_data_dir_is_kept():
    srv = Path("/srv/bailey-b/data")
    assert CorpusSettings(data_dir=srv).data_dir == srv
