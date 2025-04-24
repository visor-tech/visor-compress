# A customized numcodec based on FFMPEG.
# By default it uses x265 codec.
# Suitable for zarr.
#
# pip install ffmpeg-python   # not ffmpeg, not python-ffmpeg
#
# Usage with zarr:
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
        }
        if kwargs:
            self.conf.update(kwargs)
        self.debug_mode = debug_mode  # -1:suppress all output, 0:only warning, 1:show all
        if self.debug_mode >= 1:
            self.dbg_print = lambda *p: print(*p)
        else:
            self.dbg_print = lambda *p: None
        if self.debug_mode >= 0:
            self.err_print = lambda *p: print(*p, file = sys.stderr)
        else:
            self.err_print = lambda *p: None
    
    def encode(self, buf):
        # Assume `buf` a numpy.ndarray
        # Return "compressed data"
        self.dbg_print('dbg: encode', type(buf), buf.shape)
        assert isinstance(buf, np.ndarray)
        assert buf.dtype == np.uint16

        assert buf.ndim == 3 or not np.any(buf.shape[:-3] - np.array(1))
        # leave only the three right most dimensions
        buf = buf.reshape([-1, *buf.shape[-2:]])

        # encode numpy array to video stream
        height, width = buf.shape[1:]
        en_buf, info = (
            ffmpeg
            .input('pipe:',
                   s       = f'{width}x{height}',
                   pix_fmt = 'gray16le',
                   format  = 'rawvideo',
                   r       = 25,
                  )
            .output('pipe:', f='rawvideo', **self.conf)
            .run(input = buf.tobytes(),
                 capture_stdout = True,
                 capture_stderr = True,
                 quiet = False)
        )

        self.dbg_print(info.decode('utf-8'))

        # print warnings
        err_pick = re.findall(r".*arning.*", info.decode('utf-8'))
        err_pick.extend(re.findall(r".*ncompatible.*", info.decode('utf-8')))
        if err_pick:
            self.err_print('\n'.join(err_pick))

        return en_buf
        
    def decode(self, buf, out = None):
        self.dbg_print('dbg: decode', type(buf), 'len =', len(buf))
        assert type(buf) == bytes

        # video stream to raw array bytes
        de_buf, info = (
            ffmpeg
            .input('pipe:')
            .output('pipe:', format = 'rawvideo', pix_fmt = 'gray16le')
            .run(input = buf, capture_stdout = True, capture_stderr = True)
        )

        self.dbg_print(info.decode('utf-8'))

        # print warnings
        err_pick = re.findall(r".*arning.*", info.decode('utf-8'))
        if err_pick:
            self.err_print('\n'.err_pick)
        
        # find array dimension from message
        ma = re.findall(r"(\d+)x(\d+)", info.decode('utf-8'))
        width, height = tuple(map(int, ma[0]))

        # Raw array bytes to numpy array
        b = (
            np
            .frombuffer(de_buf, np.uint16)
            .reshape([-1, height, width])
        )
        out = b

        return b
    
    def get_config(self):
        self.conf.update(id = ffmpeg_codec.codec_id)
        return self.conf
    
    @classmethod
    def from_config(cls, conf):
        c = cls()
        c.conf.update(conf)
        #leave only known options
        #for k in c.conf:
        #    if k in conf:
        #        c.conf[k] = conf[k]
        return c

numcodecs.registry.register_codec(ffmpeg_codec)

