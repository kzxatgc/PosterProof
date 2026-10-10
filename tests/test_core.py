"""Core tests for PosterProof application."""

from app import AuditError, audit_images


def test_import():
    """Verify that main module symbols can be imported properly."""
    assert AuditError is not None
    assert callable(audit_images)
