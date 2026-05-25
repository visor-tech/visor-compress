from pathlib import Path
from typing import Any, cast

import numpy as np
import numcodecs
import pytest  # pyright: ignore[reportMissingImports]
import zarr

import numcodecs_ffmpeg
from zarr_op import build_requested_compression


NUMCODECS_REGISTRY = cast(Any, numcodecs).registry
DATA = (
	(np.arange(4 * 64 * 64, dtype=np.uint16).reshape(4, 64, 64) * 17)
	% np.uint16(65535)
)


@pytest.fixture
def sample_data() -> np.ndarray:
	return DATA.copy()


def test_registry_smoke() -> None:
	blosc = NUMCODECS_REGISTRY.get_codec(
		{"id": "blosc", "cname": "zstd", "clevel": 5}
	)
	ffmpeg_v2 = NUMCODECS_REGISTRY.get_codec({"id": "ffmpeg", "crf": 28})
	ffmpeg_v3 = build_requested_compression("ffmpeg", {}, 3)

	assert blosc.codec_id == "blosc"
	assert ffmpeg_v2.codec_id == "ffmpeg"
	assert len(ffmpeg_v3) == 1
	assert getattr(ffmpeg_v3[0], "codec_name", None) == "ffmpeg"


def test_v2_round_trip(tmp_path: Path, sample_data: np.ndarray) -> None:
	path = tmp_path / "ffmpeg_codec_v2_test.zarr"
	compressor = NUMCODECS_REGISTRY.get_codec({"id": "ffmpeg", "crf": 28})
	arr = cast(Any, zarr.open(
		str(path),
		mode="w",
		shape=sample_data.shape,
		chunks=sample_data.shape,
		dtype=sample_data.dtype,
		compressor=compressor,
		zarr_format=2,
	))
	arr[:] = sample_data
	loaded = np.asarray(arr[:])

	assert loaded.shape == sample_data.shape
	assert loaded.dtype == sample_data.dtype


def test_v3_round_trip(tmp_path: Path, sample_data: np.ndarray) -> None:
	path = tmp_path / "ffmpeg_codec_v3_test.zarr"
	compressors = build_requested_compression("ffmpeg", {}, 3)
	arr = cast(Any, zarr.create_array(
		str(path),
		shape=sample_data.shape,
		dtype=sample_data.dtype,
		chunks=sample_data.shape,
		compressors=compressors,
		zarr_format=3,
		overwrite=True,
	))
	arr[:] = sample_data
	loaded = np.asarray(arr[:])
	reopened = cast(Any, zarr.open(str(path), mode="r"))
	reopened_loaded = np.asarray(reopened[:])

	assert loaded.shape == sample_data.shape
	assert loaded.dtype == sample_data.dtype
	assert reopened_loaded.shape == sample_data.shape
	assert reopened_loaded.dtype == sample_data.dtype
	assert getattr(reopened, "compressors", ())
