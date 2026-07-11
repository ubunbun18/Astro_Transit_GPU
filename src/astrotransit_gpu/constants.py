"""Shared constants for the AstroTransit-GPU package."""

# --- Flux-error padding sentinel ---
#
# When a TESS light curve is padded to a uniform length for the vectorized
# GPU pipeline, the flux_err column is filled with FLUX_ERR_PAD_SENTINEL.
# Real photometric errors are well below this value, so any flux_err at or
# above FLUX_ERR_PAD_THRESHOLD is treated as padding and masked out.
#
# IMPORTANT: use FLUX_ERR_PAD_THRESHOLD (not ad-hoc 0.9 / 0.99 values) in
# every code path that filters padding, so the same value is applied
# consistently across the screener, cache loader, and plotting code.
FLUX_ERR_PAD_SENTINEL = 1.0
FLUX_ERR_PAD_THRESHOLD = 0.9
