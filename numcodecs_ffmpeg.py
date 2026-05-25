# A customized numcodec based on FFMPEG.
# By default it uses x265 codec.
# Suitable for zarr.
#
# pip install ffmpeg-python   # not ffmpeg, not python-ffmpeg
#
# Usage with zarr (v2):
#    import numcodecs
#    from numcodecs_ffmpeg import ffmpeg_codec
#    import zarr
#    compressor = ffmpeg_codec(debug_mode=0, threads=1)
#    z = zarr.open(dst_zarr, shape = img_size, chunks = chunk_size,
#                  dtype = 'uint16', order = 'C', compressor=compressor)
#
# Test:
#   python3 test_numcodecs_ffmpeg.py

import sys
import re
import numpy as np
import numcodecs
import ffmpeg

ZARR_V3_IMPORT_ERROR = None
BytesBytesCodec = None
try:
    from zarr.abc.codec import BytesBytesCodec
    from zarr.core.common import parse_named_configuration
    from zarr.registry import register_codec as register_zarr_codec
except Exception as exc:  # pragma: no cover - environment-dependent import
    ZARR_V3_IMPORT_ERROR = exc
    parse_named_configuration = None
    register_zarr_codec = None


def _build_debug_hooks(debug_mode):
    if debug_mode >= 1:
        dbg_print = lambda *p: print(*p)
    else:
        dbg_print = lambda *p: None
    if debug_mode >= 0:
        err_print = lambda *p: print(*p, file=sys.stderr)
    else:
        err_print = lambda *p: None
    return dbg_print, err_print


def _normalize_chunk_array(buf):
    if not isinstance(buf, np.ndarray):
        raise TypeError(f"expected numpy.ndarray, got {type(buf)!r}")
    if buf.dtype != np.uint16:
        raise TypeError(f"expected uint16 array, got {buf.dtype!r}")
    if buf.ndim < 2:
        raise ValueError("ffmpeg codec expects at least 2 dimensions")
    if buf.ndim > 3 and np.any(np.asarray(buf.shape[:-3]) != 1):
        raise ValueError(
            "ffmpeg codec only supports 2D/3D chunks or leading singleton dimensions"
        )
    return buf.reshape([-1, *buf.shape[-2:]])


def _extract_video_shape(info_text):
    matches = re.findall(r"(\d+)x(\d+)", info_text)
    if not matches:
        raise ValueError("failed to parse decoded frame size from ffmpeg output")
    width, height = tuple(map(int, matches[0]))
    return width, height


def _collect_ffmpeg_warnings(info_text):
    warnings = re.findall(r".*arning.*", info_text)
    warnings.extend(re.findall(r".*ncompatible.*", info_text))
    return warnings


def _encode_array_with_ffmpeg(buf, conf, dbg_print, err_print):
    chunk_array = _normalize_chunk_array(buf)
    dbg_print('dbg: encode', type(buf), buf.shape)

    height, width = chunk_array.shape[1:]
    encoded, info = (
        ffmpeg
        .input('pipe:',
               s       = f'{width}x{height}',
               pix_fmt = 'gray16le',
               format  = 'rawvideo',
               r       = 25,
              )
        .output('pipe:', f='rawvideo', **conf)
        .run(input=chunk_array.tobytes(),
             capture_stdout=True,
             capture_stderr=True,
             quiet=False)
    )

    info_text = info.decode('utf-8')
    dbg_print(info_text)

    warning_lines = _collect_ffmpeg_warnings(info_text)
    if warning_lines:
        err_print('\n'.join(warning_lines))

    return encoded


def _decode_array_with_ffmpeg(buf, dbg_print, err_print):
    if isinstance(buf, bytearray):
        buf = bytes(buf)
    elif not isinstance(buf, bytes):
        buf = bytes(buf)

    dbg_print('dbg: decode', type(buf), 'len =', len(buf))

    decoded, info = (
        ffmpeg
        .input('pipe:')
        .output('pipe:', format='rawvideo', pix_fmt='gray16le')
        .run(input=buf, capture_stdout=True, capture_stderr=True)
    )

    info_text = info.decode('utf-8')
    dbg_print(info_text)

    warning_lines = _collect_ffmpeg_warnings(info_text)
    if warning_lines:
        err_print('\n'.join(warning_lines))

    width, height = _extract_video_shape(info_text)
    return np.frombuffer(decoded, np.uint16).reshape([-1, height, width])


def _chunk_spec_to_numpy_dtype(chunk_spec):
    dtype = getattr(chunk_spec, 'dtype', None)
    if hasattr(dtype, 'to_native_dtype'):
        dtype = dtype.to_native_dtype()
    return np.dtype(dtype)


def _chunk_spec_shape(chunk_spec):
    return tuple(int(value) for value in chunk_spec.shape)


def _chunk_bytes_to_array(chunk_bytes, chunk_spec):
    dtype = _chunk_spec_to_numpy_dtype(chunk_spec)
    raw = chunk_bytes.as_numpy_array().view(dtype=dtype)
    shape = _chunk_spec_shape(chunk_spec)
    expected_size = int(np.prod(shape, dtype=np.int64))
    if raw.size != expected_size:
        raise ValueError(
            f"chunk byte size does not match shape {shape} and dtype {dtype}"
        )
    return raw.reshape(shape)


def _array_to_chunk_bytes(chunk_array, chunk_spec):
    shape = _chunk_spec_shape(chunk_spec)
    if chunk_array.shape != shape:
        expected_size = int(np.prod(shape, dtype=np.int64))
        if chunk_array.size != expected_size:
            raise ValueError(
                f"decoded chunk shape {chunk_array.shape} is incompatible with expected {shape}"
            )
        chunk_array = chunk_array.reshape(shape)
    return chunk_spec.prototype.buffer.from_bytes(chunk_array.tobytes(order='C'))

class ffmpeg_codec(numcodecs.abc.Codec):
    """
    Codec based on FFMPEG for 3D array of dtype uint16.

    By default, it uses x265 codec with gray12le format (i.e. 12-bit precision).
    i.e. the lowest 4-bits (LSB) will be truncated.
    
    Demo:
      ffmpeg_codec(debug_mode=-1, threads=1, crf=0, **{'x265-params':'lossless=1'})

    Chunk array dimension is assumed to be (n_frames, height, width). Codecs could have internal limit on height and width, such as <= 8000.
    
    Warning to zarr usage:
        This is a lossy codec, which means e.g. setting z[0,0,0] = 100 will not
        result in z[0,0,0] == 100 in general.
        As a consequence, always try to write data in whole chunks, to avoid extra encoding errors.
    
    # TODO: test accuracy and speed in real applications.
    # TODO: add an adapter (encode and decode) for pre-transform the values in array.
    #       such that lower values have higher precision, and higher values have lower precision.
    #       a possible adapter is a scaled version of sqrt(x).
    """
    codec_id = "ffmpeg"

    def __init__(self, debug_mode = 0, **kwargs):
        # default options
        self.conf = {
            # 'pix_fmt': pixel format, we choose gray12le (highest in libx265)
            # options: gray, gray10le, gray12le, gray16le, ...
            # see also `ffmpeg -pix_fmts`
            'pix_fmt': 'gray12le',

            # 'c:v': codec for video
            # options: libx265, libx264, libaom-av1(not tested), libvpx-vp9(not tested), ...
            #      or: hevc, h264, av1(not tested), vp9(not tested)
            # see also `ffmpeg -codecs` and `ffmpeg -encoders`
            #          `ffmpeg -h encoder=libx265`
            # Option tuning ref:
            #   https://trac.ffmpeg.org/wiki/Encode/H.265
            #   https://trac.ffmpeg.org/wiki/Encode/H.264
            'c:v'    : 'hevc',

            # 'crf': Constant Rate Factor, aka. Constant Quality (CQ)
            # For H265  8-bit
            #   crf=24: visually transparent
            #   crf=28: good default
            # For H265  16-bit (12-bit precision)
            #   crf=28: not too bad.
            # For H264, 24-bit color
            #   ranges from 0(lossless) to 51(worst quality)
            #   crf=18: visually transparent
            #   crf=23: good default
            'crf'    : 28,

            # 'preset': compression level, default medium.
            # options: faster, fast, medium, slow, slower, ...
            'preset' : 'medium',
            
            # 'tune': visual quality tuning preference, generally not required.
            # options: film grain stillimage psnr ssim fastdecode
            #'tune'   : 'psnr',

            # lossless option
            #'x265-params': 'lossless=1'
            
            # disables lookahead slices (1), or enable it (0)
            'x265-params': 'no-lookahead-slices=1',

            # manually set lookahead slices
            #'x265-params': 'lookahead-slices=0:rc-lookahead=20',
        }
        if kwargs:
            self.conf.update(kwargs)
        self.debug_mode = debug_mode  # -1:suppress all output, 0:only warning, 1:show all
        self.dbg_print, self.err_print = _build_debug_hooks(self.debug_mode)
    
    def encode(self, buf):
        # Assume `buf` a numpy.ndarray
        # Return "compressed data"
        return _encode_array_with_ffmpeg(
            buf,
            self.conf,
            self.dbg_print,
            self.err_print,
        )
        
    def decode(self, buf, out = None):
        b = _decode_array_with_ffmpeg(buf, self.dbg_print, self.err_print)
        out = b

        return b
    
    def get_config(self):
        conf = dict(self.conf)
        conf.update(id = ffmpeg_codec.codec_id)
        return conf
    
    @classmethod
    def from_config(cls, conf):
        c = cls()
        conf = dict(conf)
        conf.pop('id', None)
        c.conf.update(conf)
        #leave only known options
        #for k in c.conf:
        #    if k in conf:
        #        c.conf[k] = conf[k]
        return c


if BytesBytesCodec is not None:
    class ffmpeg_codec_v3(BytesBytesCodec):
        """Zarr v3 bytes-to-bytes codec wrapping the FFmpeg chunk transform."""

        codec_name = "ffmpeg"
        is_fixed_size = False

        def __init__(self, *, debug_mode=0, **kwargs):
            self.conf = dict(ffmpeg_codec().conf)
            if kwargs:
                self.conf.update(kwargs)
            self.debug_mode = debug_mode
            self.dbg_print, self.err_print = _build_debug_hooks(self.debug_mode)

        @classmethod
        def from_dict(cls, data):
            _, configuration = parse_named_configuration(
                data,
                cls.codec_name,
                require_configuration=False,
            )
            configuration = dict(configuration or {})
            debug_mode = int(configuration.pop('debug_mode', 0))
            return cls(debug_mode=debug_mode, **configuration)

        def to_dict(self):
            configuration = dict(self.conf)
            if self.debug_mode != 0:
                configuration['debug_mode'] = self.debug_mode
            return {
                'name': self.codec_name,
                'configuration': configuration,
            }

        def validate(self, *, shape, dtype, chunk_grid):
            native_dtype = dtype.to_native_dtype() if hasattr(dtype, 'to_native_dtype') else dtype
            if np.dtype(native_dtype) != np.dtype(np.uint16):
                raise ValueError('ffmpeg codec only supports uint16 arrays')
            if len(shape) < 2:
                raise ValueError('ffmpeg codec expects arrays with at least 2 dimensions')
            chunk_shape = getattr(chunk_grid, 'chunk_shape', None)
            if chunk_shape is not None and len(tuple(chunk_shape)) < 2:
                raise ValueError('ffmpeg codec expects chunk shapes with at least 2 dimensions')

        def _decode_sync(self, chunk_bytes, chunk_spec):
            decoded = _decode_array_with_ffmpeg(
                chunk_bytes.as_numpy_array().tobytes(),
                self.dbg_print,
                self.err_print,
            )
            return _array_to_chunk_bytes(decoded, chunk_spec)

        async def _decode_single(self, chunk_bytes, chunk_spec):
            return self._decode_sync(chunk_bytes, chunk_spec)

        def _encode_sync(self, chunk_bytes, chunk_spec):
            chunk_array = _chunk_bytes_to_array(chunk_bytes, chunk_spec)
            encoded = _encode_array_with_ffmpeg(
                chunk_array,
                self.conf,
                self.dbg_print,
                self.err_print,
            )
            return chunk_spec.prototype.buffer.from_bytes(encoded)

        async def _encode_single(self, chunk_bytes, chunk_spec):
            return self._encode_sync(chunk_bytes, chunk_spec)

        def compute_encoded_size(self, _input_byte_length, _chunk_spec):
            raise NotImplementedError('ffmpeg codec has variable encoded size')
else:
    ffmpeg_codec_v3 = None

numcodecs.registry.register_codec(ffmpeg_codec)
if register_zarr_codec is not None and ffmpeg_codec_v3 is not None:
    register_zarr_codec(ffmpeg_codec_v3.codec_name, ffmpeg_codec_v3)

