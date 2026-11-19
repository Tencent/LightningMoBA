"""Compatibility checks shared by Python operator registrations."""
from packaging.version import Version


def supports_custom_ops(version):
    return Version(str(version)).release >= (2, 4)
