# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

from libc.stdint cimport uint8_t, uint32_t
from cython.parallel cimport prange


# =========================================================================
# Funções Auxiliares e Tabelas Pré-Calculadas (LUT)
# =========================================================================

cdef inline uint32_t div255(uint32_t v) noexcept nogil:
    """Divisão inteira exata por 255 usando bitshift (1 ciclo de CPU)."""
    return (v + 1 + (v >> 8)) >> 8

cdef inline uint8_t clamp_u8(uint32_t v) noexcept nogil:
    return 255 if v > 255 else <uint8_t>v

# Tabela LUT de 64 KB para desmultiplicação instantânea sem divisão:
# UNPREMUL_LUT[alpha][color] = (color * 255 + (alpha // 2)) // alpha
cdef uint8_t UNPREMUL_LUT[256][256]

cdef void _init_unpremul_lut() noexcept nogil:
    cdef int a, c
    cdef uint32_t val
    for a in range(256):
        for c in range(256):
            if a == 0:
                UNPREMUL_LUT[a][c] = 0
            elif c >= a:
                UNPREMUL_LUT[a][c] = 255
            else:
                val = (<uint32_t>c * 255 + (a >> 1)) // a
                UNPREMUL_LUT[a][c] = clamp_u8(val)

_init_unpremul_lut()


# =========================================================================
# Conversões RGBA <-> PRGBA
# =========================================================================

def rgba_to_prgba(uint8_t[:, :, :] src, uint8_t[:, :, :] dst):
    """Pré-multiplica canais RGB pelo canal alfa (RGBA -> PRGBA)."""
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef int y, x, idx
    cdef const uint8_t* s_row
    cdef uint8_t* d_row
    cdef uint32_t ae

    with nogil:
        for y in prange(h, schedule='static'):
            s_row = &src[y, 0, 0]
            d_row = &dst[y, 0, 0]
            for x in range(w):
                idx = x << 2
                ae = s_row[idx + 3]
                if ae == 0:
                    (<uint32_t*>&d_row[idx])[0] = 0
                elif ae == 255:
                    (<uint32_t*>&d_row[idx])[0] = (<const uint32_t*>&s_row[idx])[0]
                else:
                    d_row[idx + 0] = <uint8_t>div255(s_row[idx + 0] * ae)
                    d_row[idx + 1] = <uint8_t>div255(s_row[idx + 1] * ae)
                    d_row[idx + 2] = <uint8_t>div255(s_row[idx + 2] * ae)
                    d_row[idx + 3] = <uint8_t>ae


def prgba_to_rgba(uint8_t[:, :, :] src, uint8_t[:, :, :] dst):
    """Desmultiplica canais RGB usando LUT de 64KB sem divisões (PRGBA -> RGBA)."""
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef int y, x, idx
    cdef const uint8_t* s_row
    cdef uint8_t* d_row
    cdef uint8_t ae

    with nogil:
        for y in prange(h, schedule='static'):
            s_row = &src[y, 0, 0]
            d_row = &dst[y, 0, 0]
            for x in range(w):
                idx = x << 2
                ae = s_row[idx + 3]
                if ae == 0:
                    (<uint32_t*>&d_row[idx])[0] = 0
                elif ae == 255:
                    (<uint32_t*>&d_row[idx])[0] = (<const uint32_t*>&s_row[idx])[0]
                else:
                    d_row[idx + 0] = UNPREMUL_LUT[ae][s_row[idx + 0]]
                    d_row[idx + 1] = UNPREMUL_LUT[ae][s_row[idx + 1]]
                    d_row[idx + 2] = UNPREMUL_LUT[ae][s_row[idx + 2]]
                    d_row[idx + 3] = ae


# =========================================================================
# Conversões RGB <-> RGBX / RGBA
# =========================================================================

def rgb_to_rgbx(uint8_t[:, :, :] src, uint8_t[:, :, :] dst):
    """Converte RGB (3 canais) para RGBX (4 canais com padding 255)."""
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef int y, x, s_idx, d_idx
    cdef const uint8_t* s_row
    cdef uint8_t* d_row

    with nogil:
        for y in prange(h, schedule='static'):
            s_row = &src[y, 0, 0]
            d_row = &dst[y, 0, 0]
            for x in range(w):
                s_idx = x * 3
                d_idx = x << 2
                d_row[d_idx + 0] = s_row[s_idx + 0]
                d_row[d_idx + 1] = s_row[s_idx + 1]
                d_row[d_idx + 2] = s_row[s_idx + 2]
                d_row[d_idx + 3] = 255


def rgbx_to_rgb(uint8_t[:, :, :] src, uint8_t[:, :, :] dst):
    """Descarta o canal de padding de RGBX (4 canais -> 3 canais)."""
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef int y, x, s_idx, d_idx
    cdef const uint8_t* s_row
    cdef uint8_t* d_row

    with nogil:
        for y in prange(h, schedule='static'):
            s_row = &src[y, 0, 0]
            d_row = &dst[y, 0, 0]
            for x in range(w):
                s_idx = x << 2
                d_idx = x * 3
                d_row[d_idx + 0] = s_row[s_idx + 0]
                d_row[d_idx + 1] = s_row[s_idx + 1]
                d_row[d_idx + 2] = s_row[s_idx + 2]


def prgba_to_rgb(uint8_t[:, :, :] src, uint8_t[:, :, :] dst):
    """Desmultiplica PRGBA diretamente para RGB (4 canais -> 3 canais)."""
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef int y, x, s_idx, d_idx
    cdef const uint8_t* s_row
    cdef uint8_t* d_row
    cdef uint8_t ae

    with nogil:
        for y in prange(h, schedule='static'):
            s_row = &src[y, 0, 0]
            d_row = &dst[y, 0, 0]
            for x in range(w):
                s_idx = x << 2
                d_idx = x * 3
                ae = s_row[s_idx + 3]
                if ae == 0:
                    d_row[d_idx + 0] = 0
                    d_row[d_idx + 1] = 0
                    d_row[d_idx + 2] = 0
                elif ae == 255:
                    d_row[d_idx + 0] = s_row[s_idx + 0]
                    d_row[d_idx + 1] = s_row[s_idx + 1]
                    d_row[d_idx + 2] = s_row[s_idx + 2]
                else:
                    d_row[d_idx + 0] = UNPREMUL_LUT[ae][s_row[s_idx + 0]]
                    d_row[d_idx + 1] = UNPREMUL_LUT[ae][s_row[s_idx + 1]]
                    d_row[d_idx + 2] = UNPREMUL_LUT[ae][s_row[s_idx + 2]]


def prgba_to_rgbx(uint8_t[:, :, :] src, uint8_t[:, :, :] dst):
    """Desmultiplica PRGBA diretamente para RGBX (4 canais -> 4 canais)."""
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef int y, x, idx
    cdef const uint8_t* s_row
    cdef uint8_t* d_row
    cdef uint8_t ae

    with nogil:
        for y in prange(h, schedule='static'):
            s_row = &src[y, 0, 0]
            d_row = &dst[y, 0, 0]
            for x in range(w):
                idx = x << 2
                ae = s_row[idx + 3]
                if ae == 0:
                    d_row[idx + 0] = 0
                    d_row[idx + 1] = 0
                    d_row[idx + 2] = 0
                    d_row[idx + 3] = 255
                elif ae == 255:
                    d_row[idx + 0] = s_row[idx + 0]
                    d_row[idx + 1] = s_row[idx + 1]
                    d_row[idx + 2] = s_row[idx + 2]
                    d_row[idx + 3] = 255
                else:
                    d_row[idx + 0] = UNPREMUL_LUT[ae][s_row[idx + 0]]
                    d_row[idx + 1] = UNPREMUL_LUT[ae][s_row[idx + 1]]
                    d_row[idx + 2] = UNPREMUL_LUT[ae][s_row[idx + 2]]
                    d_row[idx + 3] = 255
