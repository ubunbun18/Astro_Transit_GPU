import pytest
import os
import shutil
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from astrotransit_gpu.data.bulk_downloader import BulkDownloader

@pytest.fixture
def temp_dir():
    path = "tests/temp_download_async"
    os.makedirs(path, exist_ok=True)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

@pytest.mark.asyncio
async def test_init_creates_dir(temp_dir):
    new_dir = os.path.join(temp_dir, "new_bulk_async")
    BulkDownloader(base_dir=new_dir)
    assert os.path.exists(new_dir)

@pytest.mark.asyncio
async def test_download_from_list_skipped_existing(temp_dir):
    # Prepare an existing file
    file_path = os.path.join(temp_dir, "test_async.fits")
    with open(file_path, "w") as f:
        f.write("existing content")
    
    downloader = BulkDownloader(base_dir=temp_dir, workers=1)
    tasks = [("http://example.com/test.fits", file_path)]
    
    results = await downloader.download_from_list(tasks)
    assert results["skipped"] == 1
    assert results["ok"] == 0

@pytest.mark.asyncio
async def test_download_from_list_success(temp_dir):
    file_path = os.path.join(temp_dir, "new_async.fits")
    downloader = BulkDownloader(base_dir=temp_dir, workers=1)
    tasks = [("http://example.com/new.fits", file_path)]
    
    # Mock aiohttp ClientSession.get
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.content.read.side_effect = [b"data", b""]
    mock_response.__aenter__.return_value = mock_response
    
    with patch("aiohttp.ClientSession.get", return_value=mock_response):
        results = await downloader.download_from_list(tasks)
        assert results["ok"] == 1
        assert os.path.exists(file_path)
        with open(file_path, "rb") as f:
            assert f.read() == b"data"
        # Invariant: No .part file left
        assert not os.path.exists(file_path + ".part")

@pytest.mark.asyncio
async def test_download_error_cleanup(temp_dir):
    file_path = os.path.join(temp_dir, "fail_async.fits")
    downloader = BulkDownloader(base_dir=temp_dir, workers=1)
    tasks = [("http://example.com/fail.fits", file_path)]
    
    # Mock an exception
    with patch("aiohttp.ClientSession.get", side_effect=Exception("Network Error")):
        results = await downloader.download_from_list(tasks)
        assert results["error"] == 1
        assert not os.path.exists(file_path)
        # Invariant: No .part file left on failure
        assert not os.path.exists(file_path + ".part")

@pytest.mark.asyncio
async def test_empty_tasks(temp_dir):
    downloader = BulkDownloader(base_dir=temp_dir, workers=1)
    results = await downloader.download_from_list([])
    assert results["ok"] == 0
    assert results["error"] == 0
    assert results["skipped"] == 0

@pytest.mark.asyncio
async def test_invalid_product_type_handling():
    # Use AsyncMock for manifest fetching
    downloader = BulkDownloader(base_dir="tmp", workers=1)
    with patch("astrotransit_gpu.data.bulk_downloader.Observations.query_criteria", return_value=[]):
        df = await downloader.get_sector_manifest(1, product_type="INVALID")
        assert df.empty

@pytest.mark.asyncio
async def test_download_error_continues(temp_dir):
    # Test that one error doesn't stop other downloads
    file_path1 = os.path.join(temp_dir, "fail_cont.fits")
    file_path2 = os.path.join(temp_dir, "ok_cont.fits")
    downloader = BulkDownloader(base_dir=temp_dir, workers=2)
    tasks = [
        ("http://example.com/fail.fits", file_path1),
        ("http://example.com/ok.fits", file_path2)
    ]
    
    def mocked_get(url, **kwargs):
        if "fail.fits" in url:
            raise Exception("Fail")
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.content.read.side_effect = [b"data", b""]
        mock_resp.__aenter__.return_value = mock_resp
        return mock_resp

    with patch("aiohttp.ClientSession.get", side_effect=mocked_get):
        results = await downloader.download_from_list(tasks)
        assert results["error"] == 1
        assert results["ok"] == 1
        assert os.path.exists(file_path2)
        assert not os.path.exists(file_path1)
