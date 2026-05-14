"""Sanity tests that exercise project plumbing, not behaviour."""

import app


def test_package_imports():
    assert app.__version__ == "0.1.0"
