#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, cast

import zarr

try:
    import numcodecs
except ImportError:
    numcodecs = None

try:
    from zarr import codecs as zarr_codecs
except ImportError:
    zarr_codecs = None

FFMPEG_IMPORT_ERROR = None
ffmpeg_codec = None
try:
    from numcodecs_ffmpeg import ffmpeg_codec as _ffmpeg_codec
except Exception as exc:  # pragma: no cover - environment-dependent import
    FFMPEG_IMPORT_ERROR = exc
else:
    ffmpeg_codec = _ffmpeg_codec

def parse_shape(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in text.lower().split("x") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "shape must be formatted like 64x64x64"
        )
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}") from exc
    if any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError(
            "shape values must be positive integers"
        )
    return shape


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(
            "compressor options must decode to a JSON object"
        )
    return value


def format_shape(shape: Sequence[int] | None) -> str:
    if shape is None:
        return "None"
    return "x".join(str(int(value)) for value in shape)


def to_plain_data(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): to_plain_data(sub_value)
            for key, sub_value in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [to_plain_data(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        return to_plain_data(value.to_dict())
    if hasattr(value, "get_config"):
        return to_plain_data(value.get_config())
    if hasattr(value, "value"):
        return to_plain_data(value.value)
    return repr(value)


def format_json(value: Any) -> str:
    return json.dumps(to_plain_data(value), sort_keys=True)


def is_array(node: Any) -> bool:
    return hasattr(node, "shape") and hasattr(node, "dtype")


def open_node(path: Path) -> Any:
    return zarr.open(str(path), mode="r")


def get_zarr_format(node: Any) -> int:
    metadata = getattr(node, "metadata", None)
    zarr_format = getattr(metadata, "zarr_format", None)
    if zarr_format in (2, 3):
        return int(zarr_format)
    if hasattr(node, "shards"):
        return 3
    return 2


def get_nbytes_stored(node: Any) -> int | None:
    value = getattr(node, "nbytes_stored", None)
    if callable(value):
        value = cast(Callable[[], int], value)()
    if value is None:
        return None
    return int(value)


def get_group_member_count(group: Any) -> int | None:
    if hasattr(group, "members"):
        try:
            return sum(1 for _ in group.members())
        except TypeError:
            pass
    keys_attr = getattr(group, "keys", None)
    if callable(keys_attr):
        return len(list(cast(Callable[[], Iterable[str]], keys_attr)()))
    return None


def get_array_chunks(array: Any) -> tuple[int, ...] | None:
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    return tuple(int(value) for value in chunks)


def get_array_shards(array: Any) -> tuple[int, ...] | None:
    shards = getattr(array, "shards", None)
    if shards is not None:
        return tuple(int(value) for value in shards)
    return None


def get_array_compression(array: Any) -> Any:
    compressors = getattr(array, "compressors", None)
    if compressors is not None:
        compressors = tuple(compressors or ())
        if get_zarr_format(array) == 3:
            return compressors
        if len(compressors) == 0:
            return None
        if len(compressors) == 1:
            return compressors[0]
        return compressors
    return getattr(array, "compressor", None)


def format_compression(array: Any) -> str:
    compression = get_array_compression(array)
    if compression is None:
        return "none"
    if isinstance(compression, tuple) and not compression:
        return "none"
    return format_json(compression)


def cleanup_destination(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def ask_user_confirmation(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def require_numcodecs() -> Any:
    if numcodecs is None:
        raise RuntimeError("numcodecs is required for this operation")
    return numcodecs


def build_v2_compressor(name: str, codec_config: dict[str, Any]) -> Any:
    if name == "ffmpeg":
        if ffmpeg_codec is None:
            detail = (
                f": {FFMPEG_IMPORT_ERROR}"
                if FFMPEG_IMPORT_ERROR is not None
                else ""
            )
            raise RuntimeError(f"ffmpeg compressor is unavailable{detail}")
        return ffmpeg_codec(**codec_config)
    registry = require_numcodecs().registry
    return registry.get_codec({"id": name, **codec_config})


def build_v3_compressors(
    name: str, codec_config: dict[str, Any]
) -> tuple[Any, ...]:
    if zarr_codecs is None:
        raise RuntimeError("zarr.codecs is not available in this environment")
    if name == "ffmpeg":
        raise ValueError("ffmpeg compressor requires Zarr format 2 output")
    codec_class = getattr(zarr_codecs, name)
    return (codec_class(**codec_config),)


def build_requested_compression(
    name: str, codec_config: dict[str, Any], target_format: int
) -> Any:
    if target_format == 3:
        return build_v3_compressors(name, codec_config)
    return build_v2_compressor(name, codec_config)


def get_source_filters(array: Any, zarr_format: int) -> Any:
    filters = getattr(array, "filters", None)
    if zarr_format == 3:
        return tuple(filters or ())
    return filters


def get_source_serializer(array: Any) -> Any:
    return getattr(array, "serializer", None)


def resolve_target_format(
    source_format: int,
    requested_format: int | None,
    shard_shape: tuple[int, ...] | None,
) -> int:
    if requested_format is not None:
        return requested_format
    if shard_shape is not None and source_format != 3:
        return 3
    return source_format


def create_destination_array(
    src: Any,
    dst_path: Path,
    chunk_shape: tuple[int, ...],
    shard_shape: tuple[int, ...] | None,
    compression: Any,
    target_format: int,
    overwrite: bool,
) -> Any:
    if overwrite:
        cleanup_destination(dst_path)
    elif dst_path.exists():
        raise FileExistsError(f"destination already exists: {dst_path}")

    if target_format == 3:
        create_kwargs: dict[str, Any] = {
            "shape": src.shape,
            "dtype": src.dtype,
            "chunks": chunk_shape,
            "fill_value": src.fill_value,
            "compressors": compression,
            "zarr_format": 3,
            "overwrite": overwrite,
        }
        if shard_shape is not None:
            create_kwargs["shards"] = shard_shape
        filters = get_source_filters(src, get_zarr_format(src))
        if filters not in (None, (), []):
            create_kwargs["filters"] = filters
        serializer = get_source_serializer(src)
        if serializer is not None:
            create_kwargs["serializer"] = serializer
        return zarr.create_array(str(dst_path), **create_kwargs)

    if shard_shape is not None:
        raise ValueError("shard size requires Zarr format 3 output")

    create_kwargs = {
        "shape": src.shape,
        "chunks": chunk_shape,
        "dtype": src.dtype,
        "fill_value": src.fill_value,
        "compressor": compression,
        "store": str(dst_path),
        "zarr_format": 2,
        "overwrite": overwrite,
    }
    filters = get_source_filters(src, get_zarr_format(src))
    if filters is not None:
        create_kwargs["filters"] = filters
    order = getattr(src, "order", None)
    if order is not None:
        create_kwargs["order"] = order
    return zarr.create(**create_kwargs)


def copy_attributes(src: Any, dst: Any) -> None:
    attrs = dict(src.attrs)
    if attrs:
        dst.attrs.update(attrs)


def iter_chunk_slices(
    shape: Sequence[int], chunk_shape: Sequence[int]
) -> Iterator[tuple[slice, ...]]:
    counts = [math.ceil(dim / chunk) for dim, chunk in zip(shape, chunk_shape)]
    for chunk_indices in product(*(range(count) for count in counts)):
        slices = []
        for chunk_index, dim, chunk in zip(chunk_indices, shape, chunk_shape):
            start = chunk_index * chunk
            stop = min(start + chunk, dim)
            slices.append(slice(start, stop))
        yield tuple(slices)


def show(path_text: str) -> int:
    path = Path(path_text).expanduser().resolve()
    print(f"absolute path: {path}")
    try:
        node = open_node(path)
    except Exception:
        print("NON_ZARR_OR_CORRUPT")
        return 1

    if not is_array(node):
        print("ZARR_GROUP")
        member_count = get_group_member_count(node)
        if member_count is not None:
            print(f"members: {member_count}")
        stored_bytes = get_nbytes_stored(node)
        if stored_bytes is not None:
            print(f"number of bytes in storage: {stored_bytes}")
        return 0

    chunk_shape = get_array_chunks(node)
    shard_shape = get_array_shards(node)
    nbytes = int(getattr(node, "nbytes"))
    stored_bytes = get_nbytes_stored(node)
    if stored_bytes is None:
        stored_bytes = 0
    ratio = float("inf") if stored_bytes == 0 else nbytes / stored_bytes

    print("ZARR_ARRAY")
    print(f"shape: {format_shape(node.shape)}")
    print(f"chunk size: {format_shape(chunk_shape)}")
    if shard_shape is not None:
        print(f"shard size: {format_shape(shard_shape)}")
    print(f"default fill value: {format_json(node.fill_value)}")
    print(f"number of bytes in storage: {stored_bytes}")
    print(f"number of bytes non-compressed: {nbytes}")
    print(f"compression ratio: {ratio:.6g}")
    print(f"compression algorithm and parameters: {format_compression(node)}")
    return 0


def rechunk(args: argparse.Namespace) -> int:
    src_path = Path(args.src).expanduser().resolve()
    dst_path = Path(args.dst).expanduser().resolve()
    if src_path == dst_path:
        raise ValueError("source and destination must differ")

    if args.overwrite and dst_path.exists() and not args.yes:
        confirmed = ask_user_confirmation(
            f"destination exists and will be removed: {dst_path}. Continue"
        )
        if not confirmed:
            print("error: aborted by user", file=sys.stderr)
            return 2

    src = open_node(src_path)
    if not is_array(src):
        raise ValueError("rechunk only supports Zarr arrays, not groups")

    source_format = get_zarr_format(src)
    chunk_shape = args.chunk_size or get_array_chunks(src)
    shard_shape = (
        args.shard_size
        if args.shard_size is not None
        else get_array_shards(src)
    )
    if chunk_shape is None:
        raise ValueError("source array does not expose a chunk shape")

    target_format = resolve_target_format(
        source_format, args.zarr_format, shard_shape
    )
    if args.compressor_opt is not None and args.compressor is None:
        raise ValueError("--compressor-opt requires --compressor")
    if target_format == 2 and shard_shape is not None:
        raise ValueError("shard size requires --zarr-format 3")

    if args.compressor is None:
        compression = get_array_compression(src)
        if target_format != source_format:
            raise ValueError("changing Zarr format requires --compressor")
    else:
        compression = build_requested_compression(
            args.compressor,
            args.compressor_opt or {},
            target_format,
        )

    dst = create_destination_array(
        src=src,
        dst_path=dst_path,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        compression=compression,
        target_format=target_format,
        overwrite=args.overwrite,
    )
    copy_attributes(src, dst)

    copied_chunks = 0
    for slices in iter_chunk_slices(src.shape, chunk_shape):
        dst[slices] = src[slices]
        copied_chunks += 1

    print(f"source: {src_path}")
    print(f"destination: {dst_path}")
    print(f"zarr format: {target_format}")
    print(f"shape: {format_shape(src.shape)}")
    print(f"chunk size: {format_shape(chunk_shape)}")
    if shard_shape is not None:
        print(f"shard size: {format_shape(shard_shape)}")
    compression_text = (
        format_json(compression)
        if compression not in (None, ())
        else "none"
    )
    print(f"compression algorithm and parameters: {compression_text}")
    print(f"copied destination chunks: {copied_chunks}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and rechunk local Zarr arrays"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser(
        "show", help="Show basic information about a Zarr path"
    )
    show_parser.add_argument("path", help="Path to a local Zarr array or group")

    rechunk_parser = subparsers.add_parser(
        "rechunk", help="Copy a Zarr array to a new chunking layout"
    )
    rechunk_parser.add_argument("src", help="Source Zarr array path")
    rechunk_parser.add_argument("dst", help="Destination Zarr array path")
    rechunk_parser.add_argument(
        "--chunk-size",
        type=parse_shape,
        help="Chunk size formatted like 64x64x64",
    )
    rechunk_parser.add_argument(
        "--shard-size",
        type=parse_shape,
        help="Shard size formatted like 256x256x256",
    )
    rechunk_parser.add_argument(
        "--compressor",
        help=(
            "Exact compressor name understood by the current Zarr or "
            "numcodecs runtime, or ffmpeg"
        ),
    )
    rechunk_parser.add_argument(
        "--compressor-opt",
        type=parse_json_object,
        help="JSON object with codec-specific compressor options",
    )
    rechunk_parser.add_argument(
        "--zarr-format",
        type=int,
        choices=(2, 3),
        help=(
            "Destination Zarr format. Defaults to the source format, except "
            "shard-size upgrades to v3."
        ),
    )
    rechunk_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination path if it exists",
    )
    rechunk_parser.add_argument(
        "--yes",
        action="store_true",
        help="Assume yes for overwrite confirmation prompts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "show":
            return show(args.path)
        if args.command == "rechunk":
            return rechunk(args)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())