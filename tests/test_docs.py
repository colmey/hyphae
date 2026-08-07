"""Documentation generation and relative-link policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyphae.config.settings import LLMSettings, Settings
from scripts import check_markdown_links, render_settings_reference


def test_settings_reference_uses_metadata_without_constructing_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_construction(*args: object, **kwargs: object) -> None:
        raise AssertionError("settings models must not be constructed")

    monkeypatch.setattr(Settings, "__init__", fail_construction)
    monkeypatch.setattr(LLMSettings, "__init__", fail_construction)
    monkeypatch.setenv("GEMINI_API_KEY", "environment-secret")

    table = render_settings_reference.render_settings_table()
    setting_rows = [line for line in table.splitlines() if line.startswith("| `")]

    assert len(setting_rows) == 39
    assert "`LLM_MODEL`" in table
    assert "`LLM_MODEL_NAME`" in table
    assert "`AGENT_PROMPT_PATH`" in table
    assert "`SYSTEM_PROMPT_TIME_ENABLED`" in table
    assert "`hyphae/config/models.yaml`" in table
    assert "`traces/harness.jsonl`" in table
    assert "environment-secret" not in table


@pytest.mark.parametrize(
    ("field_name", "default", "expected"),
    [
        ("enabled", True, "`true`"),
        ("count", 3, "`3`"),
        ("ratio", 0.5, "`0.5`"),
        ("mode", "fast", "`fast`"),
        ("optional", None, "*none*"),
        ("empty", "", "*unset*"),
        ("future_api_key", "must-not-appear", "*unset*"),
    ],
)
def test_settings_default_rendering_is_deterministic_and_secret_safe(
    field_name: str,
    default: object,
    expected: str,
) -> None:
    assert render_settings_reference._render_default(field_name, default) == expected


def test_settings_path_defaults_are_portable() -> None:
    repository_path = (
        render_settings_reference.REPOSITORY_ROOT / "hyphae/config/models.yaml"
    )

    assert (
        render_settings_reference._render_default("models_config_path", repository_path)
        == "`hyphae/config/models.yaml`"
    )
    assert (
        render_settings_reference._render_default(
            "trace_jsonl_path", Path("traces/harness.jsonl")
        )
        == "`traces/harness.jsonl`"
    )


def test_markdown_cells_escape_delimiters_and_newlines() -> None:
    assert (
        render_settings_reference._escape_markdown_cell("first | second\nthird")
        == "first \\| second<br>third"
    )


@pytest.mark.parametrize(
    "document",
    [
        "no markers",
        "<!-- BEGIN GENERATED SETTINGS -->\nmissing end",
        "<!-- END GENERATED SETTINGS -->\n<!-- BEGIN GENERATED SETTINGS -->",
        (
            "<!-- BEGIN GENERATED SETTINGS -->\n<!-- END GENERATED SETTINGS -->\n"
            "<!-- BEGIN GENERATED SETTINGS -->\n<!-- END GENERATED SETTINGS -->"
        ),
    ],
)
def test_generated_settings_markers_must_be_one_ordered_pair(document: str) -> None:
    with pytest.raises(ValueError):
        render_settings_reference.replace_generated_region(document, "table")


def test_generated_region_replacement_is_scoped_and_idempotent() -> None:
    document = (
        "before\n<!-- BEGIN GENERATED SETTINGS -->\nold\n"
        "<!-- END GENERATED SETTINGS -->\nafter\n"
    )

    replaced = render_settings_reference.replace_generated_region(document, "new")

    assert replaced == (
        "before\n<!-- BEGIN GENERATED SETTINGS -->\n\nnew\n\n"
        "<!-- END GENERATED SETTINGS -->\nafter\n"
    )
    assert (
        render_settings_reference.replace_generated_region(replaced, "new") == replaced
    )


def test_check_mode_is_non_mutating_and_reports_a_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "configuration.md"
    original = (
        "before\n<!-- BEGIN GENERATED SETTINGS -->\nstale\n"
        "<!-- END GENERATED SETTINGS -->\nafter\n"
    )
    document.write_text(original, encoding="utf-8")

    assert not render_settings_reference.process_document(document, write=False)

    assert document.read_text(encoding="utf-8") == original
    assert "@@" in capsys.readouterr().err

    monkeypatch.setattr(render_settings_reference, "DOCUMENT_PATH", document)
    assert render_settings_reference.main(["--check"]) == 1
    assert document.read_text(encoding="utf-8") == original
    capsys.readouterr()

    assert render_settings_reference.main(["--write"]) == 0
    first_write = document.read_text(encoding="utf-8")
    assert render_settings_reference.process_document(document, write=True)
    assert document.read_text(encoding="utf-8") == first_write


def test_settings_cli_requires_exactly_one_mode() -> None:
    with pytest.raises(SystemExit):
        render_settings_reference.main([])
    with pytest.raises(SystemExit):
        render_settings_reference.main(["--write", "--check"])


def test_relative_link_check_handles_urls_anchors_images_and_encoded_paths(
    tmp_path: Path,
) -> None:
    repository = tmp_path
    docs = repository / "docs"
    nested = docs / "nested"
    nested.mkdir(parents=True)
    (docs / "target file.md").write_text("# Target\n", encoding="utf-8")
    (docs / "image.png").write_bytes(b"image")
    source = nested / "source.md"
    source.write_text(
        "\n".join(
            (
                "[target](../target%20file.md#target)",
                "![image](../image.png)",
                "[local](#section)",
                "[web](https://example.com/docs)",
                "[mail](mailto:operator@example.com)",
            )
        ),
        encoding="utf-8",
    )

    assert (
        check_markdown_links.find_missing_links(
            [source], repository_root=repository
        )
        == []
    )


def test_relative_link_check_reports_every_missing_target(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    source.write_text(
        "[first](missing-one.md)\n[second](folder/missing-two.md#section)\n",
        encoding="utf-8",
    )

    missing = check_markdown_links.find_missing_links(
        [source], repository_root=tmp_path
    )

    assert missing == [
        (Path("README.md"), 1, "missing-one.md"),
        (Path("README.md"), 2, "folder/missing-two.md#section"),
    ]


def test_relative_link_check_reads_reference_definitions(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    source.write_text(
        "[guide][guide-reference]\n\n[guide-reference]: missing-guide.md\n",
        encoding="utf-8",
    )

    missing = check_markdown_links.find_missing_links(
        [source], repository_root=tmp_path
    )

    assert missing == [(Path("README.md"), 3, "missing-guide.md")]


def test_relative_link_check_balances_parentheses_in_inline_targets(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "guide_(old).md"
    existing.write_text("# Existing\n", encoding="utf-8")
    source = tmp_path / "README.md"
    source.write_text(
        "[existing](guide_(old).md)\n[missing](missing_(old).md)\n",
        encoding="utf-8",
    )

    missing = check_markdown_links.find_missing_links(
        [source], repository_root=tmp_path
    )

    assert missing == [(Path("README.md"), 2, "missing_(old).md")]


def test_tracked_markdown_inventory_excludes_ignored_refactoring() -> None:
    documents = check_markdown_links.tracked_markdown_files(
        check_markdown_links.REPOSITORY_ROOT
    )

    relative_paths = {
        path.relative_to(check_markdown_links.REPOSITORY_ROOT) for path in documents
    }
    assert Path("README.md") in relative_paths
    assert all(
        path.parts[:2] != ("docs", "refactoring") for path in relative_paths
    )
