# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Tests for ImageCache and ResourceLink integration in extract_output."""

import time
import base64
from unittest.mock import patch

import pytest
from mcp.types import ImageContent, ResourceLink

from jupyter_mcp_server.image_cache import ImageCache, get_image_cache


# ---------------------------------------------------------------------------
# ImageCache unit tests
# ---------------------------------------------------------------------------

class TestImageCache:
    def test_store_and_get(self):
        cache = ImageCache()
        data = base64.b64encode(b"fake-png-data").decode()
        image_id = cache.store(data, mime_type="image/png")

        entry = cache.get(image_id)
        assert entry is not None
        assert entry.data == data
        assert entry.mime_type == "image/png"
        assert entry.image_id == image_id

    def test_get_missing_returns_none(self):
        cache = ImageCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        cache = ImageCache(ttl_seconds=1)
        data = base64.b64encode(b"ttl-test").decode()
        image_id = cache.store(data)

        assert cache.get(image_id) is not None
        time.sleep(1.1)
        assert cache.get(image_id) is None

    def test_max_entries_eviction(self):
        cache = ImageCache(max_entries=3)
        ids = []
        for i in range(5):
            image_id = cache.store(f"data-{i}")
            ids.append(image_id)

        # First two should have been evicted
        assert cache.get(ids[0]) is None
        assert cache.get(ids[1]) is None
        # Last three should still be present
        assert cache.get(ids[2]) is not None
        assert cache.get(ids[3]) is not None
        assert cache.get(ids[4]) is not None

    def test_list_images_excludes_expired(self):
        cache = ImageCache(ttl_seconds=1)
        cache.store("old-data")
        time.sleep(1.1)
        cache.store("new-data")

        images = cache.list_images()
        assert len(images) == 1
        assert images[0].data == "new-data"

    def test_list_images_returns_all_valid(self):
        cache = ImageCache()
        cache.store("a")
        cache.store("b")
        assert len(cache.list_images()) == 2

    def test_singleton(self):
        cache1 = get_image_cache()
        cache2 = get_image_cache()
        assert cache1 is cache2


# ---------------------------------------------------------------------------
# extract_output with ResourceLink (resource mode)
# ---------------------------------------------------------------------------

class TestExtractOutputResourceMode:
    """Test extract_output returns ResourceLink when IMG_OUTPUT_MODE=resource."""

    def _make_display_data(self, png_data: str = "iVBORw0KGgo="):
        return {
            "output_type": "display_data",
            "data": {"image/png": png_data},
        }

    @patch("jupyter_mcp_server.utils.IMG_OUTPUT_MODE", "resource")
    @patch("jupyter_mcp_server.utils.ALLOW_IMG_OUTPUT", True)
    def test_returns_resource_link(self):
        from jupyter_mcp_server.utils import extract_output

        output = self._make_display_data()
        result = extract_output(output)

        assert isinstance(result, ResourceLink)
        assert result.type == "resource_link"
        assert str(result.uri).startswith("jupyter://images/")
        assert result.mimeType == "image/png"

    @patch("jupyter_mcp_server.utils.IMG_OUTPUT_MODE", "resource")
    @patch("jupyter_mcp_server.utils.ALLOW_IMG_OUTPUT", True)
    def test_resource_link_data_in_cache(self):
        from jupyter_mcp_server.utils import extract_output
        from jupyter_mcp_server.image_cache import get_image_cache

        png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
        output = self._make_display_data(png_data)
        result = extract_output(output)

        # Extract image_id from URI
        image_id = str(result.uri).split("/")[-1]
        entry = get_image_cache().get(image_id)
        assert entry is not None
        assert entry.data == png_data

    @patch("jupyter_mcp_server.utils.IMG_OUTPUT_MODE", "inline")
    @patch("jupyter_mcp_server.utils.ALLOW_IMG_OUTPUT", True)
    def test_inline_mode_returns_image_content(self):
        from jupyter_mcp_server.utils import extract_output

        output = self._make_display_data()
        result = extract_output(output)

        assert isinstance(result, ImageContent)
        assert result.type == "image"

    @patch("jupyter_mcp_server.utils.ALLOW_IMG_OUTPUT", False)
    def test_disabled_returns_placeholder(self):
        from jupyter_mcp_server.utils import extract_output

        output = self._make_display_data()
        result = extract_output(output)

        assert isinstance(result, str)
        assert "disabled" in result.lower()

    def test_stream_output_unchanged(self):
        from jupyter_mcp_server.utils import extract_output

        output = {"output_type": "stream", "text": "hello world"}
        result = extract_output(output)
        assert result == "hello world"

    def test_text_plain_output_unchanged(self):
        from jupyter_mcp_server.utils import extract_output

        output = {
            "output_type": "execute_result",
            "data": {"text/plain": "42"},
        }
        result = extract_output(output)
        assert result == "42"


# ---------------------------------------------------------------------------
# safe_extract_outputs integration
# ---------------------------------------------------------------------------

class TestSafeExtractOutputsWithResourceLink:

    @patch("jupyter_mcp_server.utils.IMG_OUTPUT_MODE", "resource")
    @patch("jupyter_mcp_server.utils.ALLOW_IMG_OUTPUT", True)
    def test_mixed_outputs(self):
        from jupyter_mcp_server.utils import safe_extract_outputs

        outputs = [
            {"output_type": "stream", "text": "loading..."},
            {
                "output_type": "display_data",
                "data": {"image/png": "iVBORw0KGgo="},
            },
            {
                "output_type": "execute_result",
                "data": {"text/plain": "Done"},
            },
        ]

        result = safe_extract_outputs(outputs)
        assert len(result) == 3
        assert isinstance(result[0], str)
        assert isinstance(result[1], ResourceLink)
        assert isinstance(result[2], str)
