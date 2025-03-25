"""HTTP-related fixtures for unit tests."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests

from .common import MOCK_RESPONSES


@pytest.fixture
def mock_http_response(request):
    """Parameterized mock HTTP response.

    Args:
        request: Pytest request object that can contain parameters:
            - status_code: HTTP status code
            - content_type: Response content type
            - response_data: Data to return in the response
    """
    # Get parameters or use defaults
    status_code = getattr(request, "param", {}).get("status_code", 200)
    content_type = getattr(request, "param", {}).get("content_type", "application/json")
    response_data = getattr(request, "param", {}).get(
        "response_data", MOCK_RESPONSES["select"]
    )

    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.headers = {"Content-Type": content_type}

    if status_code >= 400:
        response.text = "Error response"
        response.ok = False
        if isinstance(response_data, str):
            response._content = response_data.encode("utf-8")
        else:
            response._content = json.dumps({"error": "Error response"}).encode("utf-8")
    else:
        response.ok = True
        if isinstance(response_data, str):
            response._content = response_data.encode("utf-8")
            response.text = response_data
        else:
            response._content = json.dumps(response_data).encode("utf-8")
            response.text = json.dumps(response_data)
            response.json = Mock(return_value=response_data)

    return response


@pytest.fixture
def mock_http_client(request):
    """Parameterized mock HTTP client for Solr requests.

    Args:
        request: Pytest request object that can contain parameters for different endpoint responses.
    """
    # Get parameters or use defaults
    params = getattr(request, "param", {})
    select_response = params.get("select_response", MOCK_RESPONSES["select"])
    schema_response = params.get("schema_response", MOCK_RESPONSES["schema"])
    fields_response = params.get(
        "fields_response", {"fields": MOCK_RESPONSES["schema"]["schema"]["fields"]}
    )
    error = params.get("error", False)

    mock = Mock(spec=requests)

    # Mock response object
    mock_response = Mock(spec=requests.Response)
    mock_response.status_code = 500 if error else 200

    if error:
        mock_response.ok = False
        mock_response.text = "Error response"
        mock_response.json.side_effect = ValueError("Invalid JSON")
    else:
        mock_response.ok = True

        # Configure responses for different endpoints
        def mock_request(method, url, **kwargs):
            if error:
                mock_response.status_code = 500
                mock_response.ok = False
                mock_response.text = "Error response"
                mock_response.json.side_effect = ValueError("Invalid JSON")
            else:
                mock_response.status_code = 200
                mock_response.ok = True

                if "/sql" in url:
                    mock_response.json.return_value = select_response
                elif "/schema" in url:
                    mock_response.json.return_value = schema_response
                elif "/fields" in url:
                    mock_response.json.return_value = fields_response
                else:
                    # Default for other endpoints
                    mock_response.json.return_value = {"status": "ok"}

            return mock_response

        # Setup the mock methods
        mock.get = Mock(
            side_effect=lambda url, **kwargs: mock_request("get", url, **kwargs)
        )
        mock.post = Mock(
            side_effect=lambda url, **kwargs: mock_request("post", url, **kwargs)
        )

    return mock


@pytest.fixture
def mock_requests_patch(mock_http_response):
    """Patch the requests module with a mock."""
    with (
        patch("requests.get", return_value=mock_http_response) as mock_get,
        patch("requests.post", return_value=mock_http_response) as mock_post,
    ):
        yield {"get": mock_get, "post": mock_post, "response": mock_http_response}


@pytest.fixture
def mock_schema_requests(mock_http_client):
    """Mock requests module for schema operations."""
    with patch("solr_mcp.solr.schema.fields.requests", mock_http_client):
        yield mock_http_client


@pytest.fixture
def mock_solr_requests(mock_http_client):
    """Mock requests module for Solr operations."""
    with (
        patch("requests.post", mock_http_client.post),
        patch("requests.get", mock_http_client.get),
    ):
        yield mock_http_client


@pytest.fixture
def mock_aiohttp_session(request):
    """Parameterized mock aiohttp session with proper async context management.

    Args:
        request: Pytest request object that can contain parameters:
            - error: Whether to simulate an error
            - response_data: Data to return in the response
    """
    # Get parameters or use defaults
    error = getattr(request, "param", {}).get("error", False)
    response_data = getattr(request, "param", {}).get(
        "response_data", '{"result-set": {"docs": [{"id": "1"}], "numFound": 1}}'
    )
    vector_response = getattr(request, "param", {}).get(
        "vector_response",
        {
            "response": {
                "docs": [{"_docid_": "1", "score": 0.9, "_vector_distance_": 0.1}],
                "numFound": 1,
            }
        },
    )

    mock_response = AsyncMock()

    if error:
        mock_response.status = 500
        mock_response.text = AsyncMock(side_effect=Exception("Mock HTTP error"))
        mock_response.__aenter__ = AsyncMock(
            side_effect=Exception("Mock session error")
        )
    else:
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.text = AsyncMock(return_value=response_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)

    mock_response.__aexit__ = AsyncMock()

    mock_session = AsyncMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()

    # Mock the vector search response
    mock_solr_response = AsyncMock()
    mock_solr_response.search = AsyncMock(return_value=vector_response)

    return mock_session
