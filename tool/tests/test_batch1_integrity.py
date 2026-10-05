"""Run-integrity fixes, batch 1 (audit D5, N5 — 2026-10-04).

D5: concurrent_harness.knowledge_io.RowBuffer reads only appended bytes and
    never parses a torn trailing row.
N5: CV manifests written on Windows (backslash paths) resolve on any OS.
"""

import sys
import types
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOOL))
sys.path.insert(0, str(_TOOL / "concurrent_harness"))
# knowledge_io imports fcntl (Linux-only, audit D10); RowBuffer doesn't use it.
sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))

import knowledge_io as kio  # noqa: E402
from adapters.cv_imagedir import _load_manifest  # noqa: E402

_FIELDS = ["step", "y_true"]


def test_rowbuffer_missing_file_is_empty(tmp_path):
    buf = kio.RowBuffer(tmp_path / "predictions.csv")
    buf.poll()
    assert buf.drain_chunk(1) is None
    assert buf.committed_count == 0


def test_rowbuffer_incremental_and_torn_row(tmp_path):
    path = tmp_path / "predictions.csv"
    buf = kio.RowBuffer(path)
    for i in range(3):
        kio.append_row(path, {"step": i, "y_true": i * 1.5}, _FIELDS)
    with open(path, "a", newline="") as f:
        f.write("3,4.")  # writer mid-append: no newline yet

    buf.poll()
    assert [r["step"] for r in buf.drain_chunk(3)] == ["0", "1", "2"]
    assert buf.drain_chunk(1) is None  # torn row not parsed
    assert buf.committed_count == 3

    with open(path, "a", newline="") as f:
        f.write("5\r\n")
    buf.poll()
    assert buf.drain_chunk(1) == [{"step": "3", "y_true": "4.5"}]
    assert buf.committed_count == 4


def test_rowbuffer_does_not_reread_consumed_rows(tmp_path):
    path = tmp_path / "predictions.csv"
    buf = kio.RowBuffer(path)
    kio.append_row(path, {"step": 0, "y_true": 1.0}, _FIELDS)
    buf.poll()
    buf.poll()  # nothing new: must not stage row 0 twice
    assert buf.drain_chunk(1) is not None
    assert buf.drain_chunk(1) is None


def test_manifest_windows_paths_are_portable(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "sample_id,input_path,label_path,label\n"
        "a,data\\imagenet_c\\clean\\a.jpg,masks\\a.png,3\n"
    )
    images, labels, inline = _load_manifest(manifest, tmp_path)
    assert "\\" not in Path(images[0]).as_posix()
    assert Path(images[0]) == tmp_path / "data" / "imagenet_c" / "clean" / "a.jpg"
    assert Path(labels[0]) == tmp_path / "masks" / "a.png"
    assert inline == [3]
