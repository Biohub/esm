"""Contracts on the published `esm` metadata.

Parsed with ``tomllib``, never grepped: the bug these guard against is a table
header going missing, and a grep for the keys under it still passes.
"""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# In the monorepo the template lives under `opensource/root/`; the sync moves it
# to the published repo root, beside `tests/`.
_CANDIDATES = (
    Path(__file__).parents[1] / "root" / "pyproject.toml",
    Path(__file__).parents[1] / "pyproject.toml",
)

#: Everything a CPU-only install must not drag in.
ACCELERATORS = frozenset(
    {
        "triton",
        "xformers",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
        "cuequivariance-ops-torch-cu13",
        "flash-attn",
        "transformer-engine",
    }
)

LINUX_X86 = {"sys_platform": "linux", "platform_machine": "x86_64"}
OTHER_PLATFORMS = {
    "macos-arm64": {"sys_platform": "darwin", "platform_machine": "arm64"},
    "linux-aarch64": {"sys_platform": "linux", "platform_machine": "aarch64"},
    "windows": {"sys_platform": "win32", "platform_machine": "AMD64"},
}

# PEP 621. Anything else directly under `[project]` is a table header that was
# dropped, not a field.
PROJECT_FIELDS = frozenset(
    {
        "name",
        "version",
        "description",
        "readme",
        "requires-python",
        "license",
        "license-files",
        "authors",
        "maintainers",
        "keywords",
        "classifiers",
        "urls",
        "scripts",
        "gui-scripts",
        "entry-points",
        "dependencies",
        "optional-dependencies",
        "dynamic",
    }
)


@pytest.fixture(scope="module")
def project() -> dict:
    for path in _CANDIDATES:
        if path.exists():
            return tomllib.loads(path.read_text())["project"]
    raise AssertionError(f"no pyproject.toml at any of {[str(p) for p in _CANDIDATES]}")


@pytest.fixture(scope="module")
def base_requirements(project) -> list[Requirement]:
    return [Requirement(spec) for spec in project["dependencies"]]


def test_there_are_no_extras(project):
    assert "optional-dependencies" not in project


def test_no_unrecognised_project_fields(project):
    assert not set(project) - PROJECT_FIELDS


def test_no_direct_references(project):
    """PyPI rejects any release whose metadata carries a URL requirement."""
    for spec in project["dependencies"]:
        assert Requirement(spec).url is None, f"direct reference: {spec}"


def test_triton_is_the_only_accelerator_and_linux_x86_only(base_requirements):
    found = [r for r in base_requirements if canonicalize_name(r.name) in ACCELERATORS]
    assert [canonicalize_name(r.name) for r in found] == ["triton"]
    assert found[0].marker is not None
    assert found[0].marker.evaluate(LINUX_X86)


@pytest.mark.parametrize("environment", OTHER_PLATFORMS.values(), ids=OTHER_PLATFORMS)
def test_base_install_is_accelerator_free_everywhere(base_requirements, environment):
    """The real minimal-install contract; there is no `cpu` extra to express it.

    An extra cannot remove a base dependency, so the only thing that can keep a
    platform clean is the markers on the requirements themselves.
    """
    selected = {
        canonicalize_name(r.name)
        for r in base_requirements
        if r.marker is None or r.marker.evaluate(environment)
    }
    assert not selected & ACCELERATORS
