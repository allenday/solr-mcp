"""Time-related fixtures for unit tests."""

import time
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_time(request):
    """Parameterized time.time mock.

    Args:
        request: Pytest request object that can contain parameters:
            - minutes_offset: Minutes to add to current time
    """
    # Get parameters or use defaults
    minutes_offset = getattr(request, "param", {}).get("minutes_offset", 0)
    seconds_offset = minutes_offset * 60

    with patch("time.time", return_value=time.time() + seconds_offset) as mock:
        yield mock
