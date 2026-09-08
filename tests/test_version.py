import re

import anicrop


def test_version_attribute_is_string():
    """Verify that anicrop exports __version__ as a non-empty string."""
    version = getattr(anicrop, "__version__", None)

    assert isinstance(version, str)
    assert len(version) > 0


def test_version_semver_format():
    """Verify that anicrop.__version__ conforms to semantic versioning."""
    version = anicrop.__version__
    pattern = r"^\d+\.\d+\.\d+(?:[a-zA-Z0-9\.\-]+)?$"

    assert re.match(pattern, version) is not None
