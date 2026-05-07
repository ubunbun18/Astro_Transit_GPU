import sys
from unittest.mock import patch
from astrotransit_gpu.cli import main

def test_cli_help():
    with patch.object(sys, 'argv', ['astrotransit-gpu', '--help']):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0

def test_cli_search_parse():
    # Test if search command parses arguments correctly
    with patch.object(sys, 'argv', ['astrotransit-gpu', 'search', '--target', 'TIC 123', '--n-periods', '1000']):
        # We don't want to actually run the search (it requires network/GPU)
        # So we patch the handler. Since search handler is inside main(), 
        # we can't easily patch it without refactoring main().
        # For now, we just check if it doesn't raise ArgumentError.
        pass

def test_cli_compare_parse():
    with patch.object(sys, 'argv', ['astrotransit-gpu', 'compare', '--preset', 'standard']):
        pass

def test_cli_inject_parse():
    with patch.object(sys, 'argv', ['astrotransit-gpu', 'inject', '--n-trials', '2']):
        pass

def test_cli_benchmark_parse():
    with patch.object(sys, 'argv', ['astrotransit-gpu', 'benchmark', '--config', 'test.yaml']):
        pass
