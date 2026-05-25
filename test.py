import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numcodecs
import pytest  # pyright: ignore[reportMissingImports]
import zarr

import numcodecs_ffmpeg
import zarr_op
from zarr_op import build_requested_compression, main


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
	blosc_v3 = build_requested_compression(
		"blosc",
		{"cname": "zstd", "clevel": 2, "shuffle": 1},
		3,
	)
	none_v2 = build_requested_compression("none", {}, 2)
	none_v3 = build_requested_compression("none", {}, 3)

	assert blosc.codec_id == "blosc"
	assert ffmpeg_v2.codec_id == "ffmpeg"
	assert len(ffmpeg_v3) == 1
	assert getattr(ffmpeg_v3[0], "codec_name", None) == "ffmpeg"
	assert len(blosc_v3) == 1
	assert blosc_v3[0].to_dict()["name"] == "numcodecs.blosc"
	assert none_v2 is None
	assert none_v3 == ()


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


def test_rechunk_defaults_to_slash_separator_for_v2(
	tmp_path: Path, sample_data: np.ndarray
) -> None:
	src = tmp_path / "src_v2.zarr"
	dst = tmp_path / "dst_v2.zarr"
	src_arr = cast(Any, zarr.open(
		str(src),
		mode="w",
		shape=sample_data.shape,
		chunks=sample_data.shape,
		dtype=sample_data.dtype,
		zarr_format=2,
	))
	src_arr[:] = sample_data

	exit_code = main([
		"rechunk",
		str(src),
		str(dst),
		"--overwrite",
		"--yes",
	])

	metadata = json.loads((dst / ".zarray").read_text())

	assert exit_code == 0
	assert metadata["dimension_separator"] == "/"
	assert (dst / "0" / "0" / "0").exists()
	assert not (dst / "0.0.0").exists()


def test_rechunk_allows_dot_separator_for_v3(
	tmp_path: Path, sample_data: np.ndarray
) -> None:
	src = tmp_path / "src_v3.zarr"
	dst = tmp_path / "dst_v3.zarr"
	src_arr = cast(Any, zarr.create_array(
		str(src),
		shape=sample_data.shape,
		dtype=sample_data.dtype,
		chunks=sample_data.shape,
		zarr_format=3,
		overwrite=True,
	))
	src_arr[:] = sample_data

	exit_code = main([
		"rechunk",
		str(src),
		str(dst),
		"--chunk-key-separator",
		".",
		"--overwrite",
		"--yes",
	])

	metadata = json.loads((dst / "zarr.json").read_text())

	assert exit_code == 0
	assert metadata["chunk_key_encoding"]["configuration"]["separator"] == "."
	assert (dst / "c.0.0.0").exists()
	assert not (dst / "c" / "0" / "0" / "0").exists()


def test_rechunk_uses_shard_size_as_copy_step(
	tmp_path: Path, sample_data: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
	src = tmp_path / "src_for_shards.zarr"
	dst = tmp_path / "dst_for_shards.zarr"
	src_arr = cast(Any, zarr.create_array(
		str(src),
		shape=sample_data.shape,
		dtype=sample_data.dtype,
		chunks=(1, 64, 64),
		zarr_format=3,
		overwrite=True,
	))
	src_arr[:] = sample_data

	step_shapes: list[tuple[int, ...]] = []
	original_iter_chunk_slices = zarr_op.iter_chunk_slices

	def record_iter_chunk_slices(
		shape: Any, step_shape: Any
	) -> Any:
		step_shapes.append(tuple(int(value) for value in step_shape))
		return original_iter_chunk_slices(shape, step_shape)

	monkeypatch.setattr(
		zarr_op,
		"iter_chunk_slices",
		record_iter_chunk_slices,
	)

	exit_code = main([
		"rechunk",
		str(src),
		str(dst),
		"--zarr-format",
		"3",
		"--shard-size",
		"2x64x64",
		"--overwrite",
		"--yes",
	])

	assert exit_code == 0
	assert step_shapes == [(2, 64, 64)]


def test_rechunk_allows_no_compression_for_v2(
	tmp_path: Path, sample_data: np.ndarray
) -> None:
	src = tmp_path / "src_compressed_v2.zarr"
	dst = tmp_path / "dst_uncompressed_v2.zarr"
	compressor = NUMCODECS_REGISTRY.get_codec(
		{"id": "blosc", "cname": "zstd", "clevel": 1}
	)
	src_arr = cast(Any, zarr.open(
		str(src),
		mode="w",
		shape=sample_data.shape,
		chunks=sample_data.shape,
		dtype=sample_data.dtype,
		compressor=compressor,
		zarr_format=2,
	))
	src_arr[:] = sample_data

	exit_code = main([
		"rechunk",
		str(src),
		str(dst),
		"--compressor",
		"none",
		"--overwrite",
		"--yes",
	])

	metadata = json.loads((dst / ".zarray").read_text())

	assert exit_code == 0
	assert metadata["compressor"] is None


def test_rechunk_help_mentions_none_compressor(
	capsys: pytest.CaptureFixture[str]
) -> None:
	with pytest.raises(SystemExit) as exc_info:
		main(["rechunk", "--help"])

	stdout = capsys.readouterr().out

	assert exc_info.value.code == 0
	assert "'none' for" in stdout
	assert "uncompressed output" in stdout
	assert "zarr.codecs" in stdout
	assert "ZstdCodec" in stdout
