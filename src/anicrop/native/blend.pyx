# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.math cimport powf, roundf
from libc.string cimport memcpy, memset
from cython.parallel cimport prange


# =========================================================================
# Fused Type para Suporte Multi-Profundidade (uint8, uint16, float32)
# =========================================================================

ctypedef fused pixel_t:
    uint8_t
    uint16_t
    float


# =========================================================================
# Funções Auxiliares e Tabelas Pré-Calculadas (LUT)
# =========================================================================

cdef inline uint8_t clamp_u8(uint32_t v) noexcept nogil:
    return 255 if v > 255 else <uint8_t>v

cdef inline uint8_t clamp_f_u8(float v) noexcept nogil:
    if v <= 0.0:
        return 0
    if v >= 255.0:
        return 255
    return <uint8_t>roundf(v)

cdef inline uint32_t div255(uint32_t v) noexcept nogil:
    """Divisão inteira exata por 255 usando bitshift (1 ciclo de CPU)."""
    return (v + 1 + (v >> 8)) >> 8

cdef inline uint8_t rgb_to_gray_u8(uint8_t r, uint8_t g, uint8_t b) noexcept nogil:
    # ITU-R BT.601: 0.299 * R + 0.587 * G + 0.114 * B
    return <uint8_t>((19595 * <uint32_t>r + 38470 * <uint32_t>g + 7471 * <uint32_t>b + 32768) >> 16)

# Tabela estática pré-calculada sRGB -> Linear (256 valores)
cdef float SRGB_TO_LIN[256]

# Tabela estática pré-calculada Linear -> sRGB (4096 valores para lookup O(1) sem powf)
cdef uint8_t LIN_TO_SRGB[4096]

# Tabela estática de recíproco em ponto fixo Q16 para divisão por out_a:
# RCP_Q16[a] = (65536 + (a >> 1)) // a para a in 1..255
cdef uint32_t RCP_Q16[256]

cdef void _init_lut() noexcept nogil:
    cdef int i
    cdef float lin_val, inv_gamma = 1.0 / 2.2
    for i in range(256):
        SRGB_TO_LIN[i] = powf(<float>i / 255.0, 2.2)
        if i == 0:
            RCP_Q16[0] = 0
        else:
            RCP_Q16[i] = (65536 + (i >> 1)) // i

    for i in range(4096):
        lin_val = <float>i / 4095.0
        LIN_TO_SRGB[i] = clamp_f_u8(powf(lin_val, inv_gamma) * 255.0)

cdef inline uint8_t lin_to_srgb_u8(float lin) noexcept nogil:
    if lin <= 0.0:
        return 0
    if lin >= 1.0:
        return 255
    cdef int idx = <int>(lin * 4095.0 + 0.5)
    if idx > 4095:
        idx = 4095
    return LIN_TO_SRGB[idx]

_init_lut()


# =========================================================================
# 1. BLEND NORMAL (Porter-Duff Over)
# =========================================================================

cdef void _blend_normal_u8(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] edit,
    float opacity,
) noexcept nogil:
    cdef int h = min(base.shape[0], edit.shape[0])
    cdef int w = min(base.shape[1], edit.shape[1])
    cdef int b_ch = base.shape[2]
    cdef int e_ch = edit.shape[2]

    cdef bint b_has_alpha = (b_ch == 2 or b_ch == 4)
    cdef bint e_has_alpha = (e_ch == 2 or e_ch == 4)
    cdef int b_colors = 1 if (b_ch == 1 or b_ch == 2) else 3
    cdef int e_colors = 1 if (e_ch == 1 or e_ch == 2) else 3

    cdef uint32_t op_256 = <uint32_t>(opacity * 256.0)
    if op_256 > 256:
        op_256 = 256

    cdef int y, x, b_idx, e_idx
    cdef uint8_t* b_row
    cdef const uint8_t* e_row
    cdef uint32_t ae_raw, ae, inv_ae, ab, term_b, out_a, rcp_out_a
    cdef uint32_t er, eg, eb

    # Fast-Path 1: RGB -> RGB (Sem Alfa)
    if b_ch == 3 and e_ch == 3:
        if op_256 >= 256:
            for y in prange(h, schedule='static'):
                memcpy(&base[y, 0, 0], &edit[y, 0, 0], w * 3)
        else:
            for y in prange(h, schedule='static'):
                b_row = &base[y, 0, 0]
                e_row = &edit[y, 0, 0]
                for x in range(w * 3):
                    b_row[x] = <uint8_t>div255(e_row[x] * op_256 + b_row[x] * (256 - op_256))

    # Fast-Path 2: RGB -> RGBA (Opaco sobre buffer RGBA)
    elif b_ch == 4 and e_ch == 3:
        if op_256 >= 256:
            for y in prange(h, schedule='static'):
                b_row = &base[y, 0, 0]
                e_row = &edit[y, 0, 0]
                for x in range(w):
                    b_idx = x << 2
                    e_idx = x * 3
                    b_row[b_idx + 0] = e_row[e_idx + 0]
                    b_row[b_idx + 1] = e_row[e_idx + 1]
                    b_row[b_idx + 2] = e_row[e_idx + 2]
                    b_row[b_idx + 3] = 255
        else:
            for y in prange(h, schedule='static'):
                b_row = &base[y, 0, 0]
                e_row = &edit[y, 0, 0]
                for x in range(w):
                    b_idx = x << 2
                    e_idx = x * 3
                    ab = b_row[b_idx + 3]
                    inv_ae = 255 - op_256
                    term_b = div255(ab * inv_ae)
                    out_a = op_256 + term_b
                    if out_a == 0:
                        b_row[b_idx + 3] = 0
                        continue
                    rcp_out_a = RCP_Q16[out_a]
                    b_row[b_idx + 0] = <uint8_t>(((e_row[e_idx + 0] * op_256 + b_row[b_idx + 0] * term_b) * rcp_out_a + 32768) >> 16)
                    b_row[b_idx + 1] = <uint8_t>(((e_row[e_idx + 1] * op_256 + b_row[b_idx + 1] * term_b) * rcp_out_a + 32768) >> 16)
                    b_row[b_idx + 2] = <uint8_t>(((e_row[e_idx + 2] * op_256 + b_row[b_idx + 2] * term_b) * rcp_out_a + 32768) >> 16)
                    b_row[b_idx + 3] = <uint8_t>out_a

    # Fast-Path 3: RGBA -> RGB (Com Alfa sobre Fundo Opaco)
    elif b_ch == 3 and e_ch == 4:
        for y in prange(h, schedule='static'):
            b_row = &base[y, 0, 0]
            e_row = &edit[y, 0, 0]
            for x in range(w):
                b_idx = x * 3
                e_idx = x << 2
                ae_raw = e_row[e_idx + 3]
                if ae_raw == 0:
                    continue
                ae = div255(ae_raw * op_256) if op_256 < 256 else ae_raw
                if ae == 0:
                    continue
                if ae == 255:
                    b_row[b_idx + 0] = e_row[e_idx + 0]
                    b_row[b_idx + 1] = e_row[e_idx + 1]
                    b_row[b_idx + 2] = e_row[e_idx + 2]
                else:
                    inv_ae = 255 - ae
                    b_row[b_idx + 0] = <uint8_t>div255(e_row[e_idx + 0] * ae + b_row[b_idx + 0] * inv_ae)
                    b_row[b_idx + 1] = <uint8_t>div255(e_row[e_idx + 1] * ae + b_row[b_idx + 1] * inv_ae)
                    b_row[b_idx + 2] = <uint8_t>div255(e_row[e_idx + 2] * ae + b_row[b_idx + 2] * inv_ae)

    # Fast-Path 4: RGBA -> RGBA
    elif b_ch == 4 and e_ch == 4:
        for y in prange(h, schedule='static'):
            b_row = &base[y, 0, 0]
            e_row = &edit[y, 0, 0]
            for x in range(w):
                e_idx = x << 2
                b_idx = x << 2

                ae_raw = e_row[e_idx + 3]
                if ae_raw == 0:
                    continue

                ae = div255(ae_raw * op_256) if op_256 < 256 else ae_raw
                if ae == 0:
                    continue

                if ae == 255:
                    (<uint32_t*>&b_row[b_idx])[0] = (<const uint32_t*>&e_row[e_idx])[0]
                    continue

                ab = b_row[b_idx + 3]
                inv_ae = 255 - ae
                term_b = div255(ab * inv_ae)
                out_a = ae + term_b
                if out_a == 0:
                    b_row[b_idx + 3] = 0
                    continue

                rcp_out_a = RCP_Q16[out_a]

                b_row[b_idx + 0] = <uint8_t>(((e_row[e_idx + 0] * ae + b_row[b_idx + 0] * term_b) * rcp_out_a + 32768) >> 16)
                b_row[b_idx + 1] = <uint8_t>(((e_row[e_idx + 1] * ae + b_row[b_idx + 1] * term_b) * rcp_out_a + 32768) >> 16)
                b_row[b_idx + 2] = <uint8_t>(((e_row[e_idx + 2] * ae + b_row[b_idx + 2] * term_b) * rcp_out_a + 32768) >> 16)
                b_row[b_idx + 3] = <uint8_t>out_a

    else:
        # Caminho geral cobrindo Grayscale e Gray-Alpha
        for y in prange(h, schedule='static'):
            b_row = &base[y, 0, 0]
            e_row = &edit[y, 0, 0]
            for x in range(w):
                b_idx = x * b_ch
                e_idx = x * e_ch

                ae_raw = e_row[e_idx + e_ch - 1] if e_has_alpha else 255
                if ae_raw == 0:
                    continue

                ae = div255(ae_raw * op_256) if op_256 < 256 else ae_raw
                if ae == 0:
                    continue

                if e_colors == 3:
                    er = e_row[e_idx + 0]
                    eg = e_row[e_idx + 1]
                    eb = e_row[e_idx + 2]
                    if b_colors == 1:
                        er = rgb_to_gray_u8(<uint8_t>er, <uint8_t>eg, <uint8_t>eb)
                        eg = er
                        eb = er
                else:
                    er = e_row[e_idx + 0]
                    eg = er
                    eb = er

                if b_has_alpha:
                    ab = b_row[b_idx + b_ch - 1]
                    inv_ae = 255 - ae
                    term_b = div255(ab * inv_ae)
                    out_a = ae + term_b
                    if out_a == 0:
                        b_row[b_idx + b_ch - 1] = 0
                        continue

                    rcp_out_a = RCP_Q16[out_a]
                    if b_colors == 3:
                        b_row[b_idx + 0] = <uint8_t>(((er * ae + b_row[b_idx + 0] * term_b) * rcp_out_a + 32768) >> 16)
                        b_row[b_idx + 1] = <uint8_t>(((eg * ae + b_row[b_idx + 1] * term_b) * rcp_out_a + 32768) >> 16)
                        b_row[b_idx + 2] = <uint8_t>(((eb * ae + b_row[b_idx + 2] * term_b) * rcp_out_a + 32768) >> 16)
                    else:
                        b_row[b_idx + 0] = <uint8_t>(((er * ae + b_row[b_idx + 0] * term_b) * rcp_out_a + 32768) >> 16)

                    b_row[b_idx + b_ch - 1] = <uint8_t>out_a
                else:
                    inv_ae = 255 - ae
                    if b_colors == 3:
                        b_row[b_idx + 0] = <uint8_t>div255(er * ae + b_row[b_idx + 0] * inv_ae)
                        b_row[b_idx + 1] = <uint8_t>div255(eg * ae + b_row[b_idx + 1] * inv_ae)
                        b_row[b_idx + 2] = <uint8_t>div255(eb * ae + b_row[b_idx + 2] * inv_ae)
                    else:
                        b_row[b_idx + 0] = <uint8_t>div255(er * ae + b_row[b_idx + 0] * inv_ae)


cdef void _blend_normal_u16(
    uint16_t[:, :, :] base,
    uint16_t[:, :, :] edit,
    float opacity,
) noexcept nogil:
    cdef int h = min(base.shape[0], edit.shape[0])
    cdef int w = min(base.shape[1], edit.shape[1])
    cdef int b_ch = base.shape[2]
    cdef int e_ch = edit.shape[2]

    cdef bint b_has_alpha = (b_ch == 2 or b_ch == 4)
    cdef bint e_has_alpha = (e_ch == 2 or e_ch == 4)
    cdef int b_colors = 1 if (b_ch == 1 or b_ch == 2) else 3
    cdef int e_colors = 1 if (e_ch == 1 or e_ch == 2) else 3

    cdef int y, x, b_idx, e_idx
    cdef uint16_t* b_row
    cdef const uint16_t* e_row
    cdef float ae_raw, ae, inv_ae, ab, out_a, out_a_safe
    cdef float er, eg, eb, br, bg, bb

    for y in prange(h, schedule='static'):
        b_row = &base[y, 0, 0]
        e_row = &edit[y, 0, 0]
        for x in range(w):
            b_idx = x * b_ch
            e_idx = x * e_ch

            ae_raw = <float>(e_row[e_idx + e_ch - 1]) / 65535.0 if e_has_alpha else 1.0
            ae = ae_raw * opacity
            if ae <= 0.0:
                continue

            inv_ae = 1.0 - ae
            if e_colors == 3:
                er = <float>(e_row[e_idx + 0])
                eg = <float>(e_row[e_idx + 1])
                eb = <float>(e_row[e_idx + 2])
                if b_colors == 1:
                    er = 0.299 * er + 0.587 * eg + 0.114 * eb
                    eg = er
                    eb = er
            else:
                er = <float>(e_row[e_idx + 0])
                eg = er
                eb = er

            if b_has_alpha:
                ab = <float>(b_row[b_idx + b_ch - 1]) / 65535.0
                out_a = ae + ab * inv_ae
                out_a_safe = out_a if out_a > 0.0 else 1.0

                if b_colors == 3:
                    br = <float>(b_row[b_idx + 0])
                    bg = <float>(b_row[b_idx + 1])
                    bb = <float>(b_row[b_idx + 2])
                    b_row[b_idx + 0] = <uint16_t>roundf((er * ae + br * ab * inv_ae) / out_a_safe)
                    b_row[b_idx + 1] = <uint16_t>roundf((eg * ae + bg * ab * inv_ae) / out_a_safe)
                    b_row[b_idx + 2] = <uint16_t>roundf((eb * ae + bb * ab * inv_ae) / out_a_safe)
                else:
                    br = <float>(b_row[b_idx + 0])
                    b_row[b_idx + 0] = <uint16_t>roundf((er * ae + br * ab * inv_ae) / out_a_safe)

                b_row[b_idx + b_ch - 1] = <uint16_t>roundf(out_a * 65535.0)
            else:
                if b_colors == 3:
                    br = <float>(b_row[b_idx + 0])
                    bg = <float>(b_row[b_idx + 1])
                    bb = <float>(b_row[b_idx + 2])
                    b_row[b_idx + 0] = <uint16_t>roundf(er * ae + br * inv_ae)
                    b_row[b_idx + 1] = <uint16_t>roundf(eg * ae + bg * inv_ae)
                    b_row[b_idx + 2] = <uint16_t>roundf(eb * ae + bb * inv_ae)
                else:
                    br = <float>(b_row[b_idx + 0])
                    b_row[b_idx + 0] = <uint16_t>roundf(er * ae + br * inv_ae)


cdef void _blend_normal_f32(
    float[:, :, :] base,
    float[:, :, :] edit,
    float opacity,
) noexcept nogil:
    cdef int h = min(base.shape[0], edit.shape[0])
    cdef int w = min(base.shape[1], edit.shape[1])
    cdef int b_ch = base.shape[2]
    cdef int e_ch = edit.shape[2]

    cdef bint b_has_alpha = (b_ch == 2 or b_ch == 4)
    cdef bint e_has_alpha = (e_ch == 2 or e_ch == 4)
    cdef int b_colors = 1 if (b_ch == 1 or b_ch == 2) else 3
    cdef int e_colors = 1 if (e_ch == 1 or e_ch == 2) else 3

    cdef int y, x, b_idx, e_idx
    cdef float* b_row
    cdef const float* e_row
    cdef float ae_raw, ae, inv_ae, ab, out_a, out_a_safe
    cdef float er, eg, eb, br, bg, bb

    for y in prange(h, schedule='static'):
        b_row = &base[y, 0, 0]
        e_row = &edit[y, 0, 0]
        for x in range(w):
            b_idx = x * b_ch
            e_idx = x * e_ch

            ae_raw = e_row[e_idx + e_ch - 1] if e_has_alpha else 1.0
            ae = ae_raw * opacity
            if ae <= 0.0:
                continue

            inv_ae = 1.0 - ae
            if e_colors == 3:
                er = e_row[e_idx + 0]
                eg = e_row[e_idx + 1]
                eb = e_row[e_idx + 2]
                if b_colors == 1:
                    er = 0.299 * er + 0.587 * eg + 0.114 * eb
                    eg = er
                    eb = er
            else:
                er = e_row[e_idx + 0]
                eg = er
                eb = er

            if b_has_alpha:
                ab = b_row[b_idx + b_ch - 1]
                out_a = ae + ab * inv_ae
                out_a_safe = out_a if out_a > 0.0 else 1.0

                if b_colors == 3:
                    br = b_row[b_idx + 0]
                    bg = b_row[b_idx + 1]
                    bb = b_row[b_idx + 2]
                    b_row[b_idx + 0] = (er * ae + br * ab * inv_ae) / out_a_safe
                    b_row[b_idx + 1] = (eg * ae + bg * ab * inv_ae) / out_a_safe
                    b_row[b_idx + 2] = (eb * ae + bb * ab * inv_ae) / out_a_safe
                else:
                    br = b_row[b_idx + 0]
                    b_row[b_idx + 0] = (er * ae + br * ab * inv_ae) / out_a_safe

                b_row[b_idx + b_ch - 1] = out_a
            else:
                if b_colors == 3:
                    br = b_row[b_idx + 0]
                    bg = b_row[b_idx + 1]
                    bb = b_row[b_idx + 2]
                    b_row[b_idx + 0] = er * ae + br * inv_ae
                    b_row[b_idx + 1] = eg * ae + bg * inv_ae
                    b_row[b_idx + 2] = eb * ae + bb * inv_ae
                else:
                    br = b_row[b_idx + 0]
                    b_row[b_idx + 0] = er * ae + br * inv_ae


def blend_normal(
    pixel_t[:, :, :] base,
    pixel_t[:, :, :] edit,
    float opacity = 1.0,
):
    """Implementação unificada em Cython com Fused Types, Q16, ponteiros C e OpenMP."""
    if opacity <= 0.0:
        return

    with nogil:
        if pixel_t is uint8_t:
            _blend_normal_u8(base, edit, opacity)
        elif pixel_t is uint16_t:
            _blend_normal_u16(base, edit, opacity)
        elif pixel_t is float:
            _blend_normal_f32(base, edit, opacity)


cdef void _blend_normal_prgba_u8(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] edit,
    float opacity,
) noexcept nogil:
    cdef int h = min(base.shape[0], edit.shape[0])
    cdef int w = min(base.shape[1], edit.shape[1])
    cdef uint32_t op_256 = <uint32_t>(opacity * 256.0)
    if op_256 > 256:
        op_256 = 256

    cdef int y, x, idx
    cdef uint8_t* b_row
    cdef const uint8_t* e_row
    cdef uint32_t ae_raw, ae, inv_ae

    if op_256 >= 256:
        # Fast-Path: Opacidade total (1.0) - Apenas 1 multiplicação/shift por canal
        for y in prange(h, schedule='static'):
            b_row = &base[y, 0, 0]
            e_row = &edit[y, 0, 0]
            for x in range(w):
                idx = x << 2
                ae = e_row[idx + 3]
                if ae == 0:
                    continue

                if ae == 255:
                    (<uint32_t*>&b_row[idx])[0] = (<const uint32_t*>&e_row[idx])[0]
                    continue

                inv_ae = 255 - ae
                b_row[idx + 0] = <uint8_t>(e_row[idx + 0] + div255(b_row[idx + 0] * inv_ae))
                b_row[idx + 1] = <uint8_t>(e_row[idx + 1] + div255(b_row[idx + 1] * inv_ae))
                b_row[idx + 2] = <uint8_t>(e_row[idx + 2] + div255(b_row[idx + 2] * inv_ae))
                b_row[idx + 3] = <uint8_t>(ae + div255(b_row[idx + 3] * inv_ae))
    else:
        # Opacidade fracionária (< 1.0)
        for y in prange(h, schedule='static'):
            b_row = &base[y, 0, 0]
            e_row = &edit[y, 0, 0]
            for x in range(w):
                idx = x << 2
                ae_raw = e_row[idx + 3]
                if ae_raw == 0:
                    continue

                ae = div255(ae_raw * op_256)
                if ae == 0:
                    continue

                inv_ae = 255 - ae
                b_row[idx + 0] = <uint8_t>(div255(e_row[idx + 0] * op_256) + div255(b_row[idx + 0] * inv_ae))
                b_row[idx + 1] = <uint8_t>(div255(e_row[idx + 1] * op_256) + div255(b_row[idx + 1] * inv_ae))
                b_row[idx + 2] = <uint8_t>(div255(e_row[idx + 2] * op_256) + div255(b_row[idx + 2] * inv_ae))
                b_row[idx + 3] = <uint8_t>(ae + div255(b_row[idx + 3] * inv_ae))


cdef void _blend_prgba_over_opaque_u8(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] edit,
    float opacity,
) noexcept nogil:
    cdef int h = min(base.shape[0], edit.shape[0])
    cdef int w = min(base.shape[1], edit.shape[1])
    cdef int b_ch = base.shape[2]
    cdef uint32_t op_256 = <uint32_t>(opacity * 256.0)
    if op_256 > 256:
        op_256 = 256

    cdef int y, x, idx, b_idx, e_idx
    cdef uint8_t* b_row
    cdef const uint8_t* e_row
    cdef uint32_t ae_raw, ae, inv_ae

    if b_ch == 4:
        if op_256 >= 256:
            for y in prange(h, schedule='static'):
                b_row = &base[y, 0, 0]
                e_row = &edit[y, 0, 0]
                for x in range(w):
                    idx = x << 2
                    ae = e_row[idx + 3]
                    if ae == 0:
                        continue

                    if ae == 255:
                        b_row[idx + 0] = e_row[idx + 0]
                        b_row[idx + 1] = e_row[idx + 1]
                        b_row[idx + 2] = e_row[idx + 2]
                        b_row[idx + 3] = 255
                        continue

                    inv_ae = 255 - ae
                    b_row[idx + 0] = <uint8_t>(e_row[idx + 0] + div255(b_row[idx + 0] * inv_ae))
                    b_row[idx + 1] = <uint8_t>(e_row[idx + 1] + div255(b_row[idx + 1] * inv_ae))
                    b_row[idx + 2] = <uint8_t>(e_row[idx + 2] + div255(b_row[idx + 2] * inv_ae))
                    b_row[idx + 3] = 255
        else:
            for y in prange(h, schedule='static'):
                b_row = &base[y, 0, 0]
                e_row = &edit[y, 0, 0]
                for x in range(w):
                    idx = x << 2
                    ae_raw = e_row[idx + 3]
                    if ae_raw == 0:
                        continue

                    ae = div255(ae_raw * op_256)
                    if ae == 0:
                        continue

                    inv_ae = 255 - ae
                    b_row[idx + 0] = <uint8_t>(div255(e_row[idx + 0] * op_256) + div255(b_row[idx + 0] * inv_ae))
                    b_row[idx + 1] = <uint8_t>(div255(e_row[idx + 1] * op_256) + div255(b_row[idx + 1] * inv_ae))
                    b_row[idx + 2] = <uint8_t>(div255(e_row[idx + 2] * op_256) + div255(b_row[idx + 2] * inv_ae))
                    b_row[idx + 3] = 255
    elif b_ch == 3:
        if op_256 >= 256:
            for y in prange(h, schedule='static'):
                b_row = &base[y, 0, 0]
                e_row = &edit[y, 0, 0]
                for x in range(w):
                    b_idx = x * 3
                    e_idx = x << 2
                    ae = e_row[e_idx + 3]
                    if ae == 0:
                        continue

                    if ae == 255:
                        b_row[b_idx + 0] = e_row[e_idx + 0]
                        b_row[b_idx + 1] = e_row[e_idx + 1]
                        b_row[b_idx + 2] = e_row[e_idx + 2]
                        continue

                    inv_ae = 255 - ae
                    b_row[b_idx + 0] = <uint8_t>(e_row[e_idx + 0] + div255(b_row[b_idx + 0] * inv_ae))
                    b_row[b_idx + 1] = <uint8_t>(e_row[e_idx + 1] + div255(b_row[b_idx + 1] * inv_ae))
                    b_row[b_idx + 2] = <uint8_t>(e_row[e_idx + 2] + div255(b_row[b_idx + 2] * inv_ae))
        else:
            for y in prange(h, schedule='static'):
                b_row = &base[y, 0, 0]
                e_row = &edit[y, 0, 0]
                for x in range(w):
                    b_idx = x * 3
                    e_idx = x << 2
                    ae_raw = e_row[e_idx + 3]
                    if ae_raw == 0:
                        continue

                    ae = div255(ae_raw * op_256)
                    if ae == 0:
                        continue

                    inv_ae = 255 - ae
                    b_row[b_idx + 0] = <uint8_t>(div255(e_row[e_idx + 0] * op_256) + div255(b_row[b_idx + 0] * inv_ae))
                    b_row[b_idx + 1] = <uint8_t>(div255(e_row[e_idx + 1] * op_256) + div255(b_row[b_idx + 1] * inv_ae))
                    b_row[b_idx + 2] = <uint8_t>(div255(e_row[e_idx + 2] * op_256) + div255(b_row[b_idx + 2] * inv_ae))


def blend_normal_prgba(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] edit,
    float opacity = 1.0,
):
    """Kernel Cython ultra-rápido para mesclagem de PRGBA sobre PRGBA."""
    if opacity <= 0.0:
        return
    with nogil:
        _blend_normal_prgba_u8(base, edit, opacity)


def blend_prgba_over_opaque(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] edit,
    float opacity = 1.0,
):
    """Kernel Cython ultra-rápido para mesclagem de PRGBA sobre fundos opacos (RGB/RGBX)."""
    if opacity <= 0.0:
        return
    with nogil:
        _blend_prgba_over_opaque_u8(base, edit, opacity)


# =========================================================================
# 2. BLEND NORMAL LINEAR (Fisicamente Correto em Luz Linear)
# =========================================================================

cdef void _blend_normal_linear_u8(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] edit,
    float opacity,
) noexcept nogil:
    cdef int h = min(base.shape[0], edit.shape[0])
    cdef int w = min(base.shape[1], edit.shape[1])
    cdef int b_ch = base.shape[2]
    cdef int e_ch = edit.shape[2]

    cdef bint b_has_alpha = (b_ch == 2 or b_ch == 4)
    cdef bint e_has_alpha = (e_ch == 2 or e_ch == 4)
    cdef int b_colors = 1 if (b_ch == 1 or b_ch == 2) else 3
    cdef int e_colors = 1 if (e_ch == 1 or e_ch == 2) else 3

    cdef float inv_gamma = 1.0 / 2.2
    cdef int y, x, b_idx, e_idx
    cdef uint8_t* b_row
    cdef const uint8_t* e_row
    cdef uint8_t ae_raw, ab_raw
    cdef float ae, ab, inv_ae, out_a, out_a_safe
    cdef float er_lin, eg_lin, eb_lin, br_lin, bg_lin, bb_lin
    cdef float out_r_lin, out_g_lin, out_b_lin

    for y in prange(h, schedule='static'):
        b_row = &base[y, 0, 0]
        e_row = &edit[y, 0, 0]
        for x in range(w):
            b_idx = x * b_ch
            e_idx = x * e_ch

            ae_raw = e_row[e_idx + e_ch - 1] if e_has_alpha else 255
            if ae_raw == 0:
                continue

            ae = (<float>ae_raw / 255.0) * opacity
            if ae <= 0.0:
                continue

            if e_colors == 3:
                if b_colors == 1:
                    er_lin = SRGB_TO_LIN[rgb_to_gray_u8(e_row[e_idx + 0], e_row[e_idx + 1], e_row[e_idx + 2])]
                    eg_lin = er_lin
                    eb_lin = er_lin
                else:
                    er_lin = SRGB_TO_LIN[e_row[e_idx + 0]]
                    eg_lin = SRGB_TO_LIN[e_row[e_idx + 1]]
                    eb_lin = SRGB_TO_LIN[e_row[e_idx + 2]]
            else:
                er_lin = SRGB_TO_LIN[e_row[e_idx + 0]]
                eg_lin = er_lin
                eb_lin = er_lin

            inv_ae = 1.0 - ae

            if b_has_alpha:
                ab_raw = b_row[b_idx + b_ch - 1]
                ab = <float>ab_raw / 255.0
                out_a = ae + ab * inv_ae
                out_a_safe = out_a if out_a > 0.0 else 1.0

                if b_colors == 3:
                    br_lin = SRGB_TO_LIN[b_row[b_idx + 0]]
                    bg_lin = SRGB_TO_LIN[b_row[b_idx + 1]]
                    bb_lin = SRGB_TO_LIN[b_row[b_idx + 2]]

                    out_r_lin = (er_lin * ae + br_lin * ab * inv_ae) / out_a_safe
                    out_g_lin = (eg_lin * ae + bg_lin * ab * inv_ae) / out_a_safe
                    out_b_lin = (eb_lin * ae + bb_lin * ab * inv_ae) / out_a_safe

                    b_row[b_idx + 0] = lin_to_srgb_u8(out_r_lin)
                    b_row[b_idx + 1] = lin_to_srgb_u8(out_g_lin)
                    b_row[b_idx + 2] = lin_to_srgb_u8(out_b_lin)
                else:
                    br_lin = SRGB_TO_LIN[b_row[b_idx + 0]]
                    out_r_lin = (er_lin * ae + br_lin * ab * inv_ae) / out_a_safe
                    b_row[b_idx + 0] = lin_to_srgb_u8(out_r_lin)

                b_row[b_idx + b_ch - 1] = clamp_f_u8(out_a * 255.0)

            else:
                if b_colors == 3:
                    br_lin = SRGB_TO_LIN[b_row[b_idx + 0]]
                    bg_lin = SRGB_TO_LIN[b_row[b_idx + 1]]
                    bb_lin = SRGB_TO_LIN[b_row[b_idx + 2]]

                    out_r_lin = er_lin * ae + br_lin * inv_ae
                    out_g_lin = eg_lin * ae + bg_lin * inv_ae
                    out_b_lin = eb_lin * ae + bb_lin * inv_ae

                    b_row[b_idx + 0] = lin_to_srgb_u8(out_r_lin)
                    b_row[b_idx + 1] = lin_to_srgb_u8(out_g_lin)
                    b_row[b_idx + 2] = lin_to_srgb_u8(out_b_lin)
                else:
                    br_lin = SRGB_TO_LIN[b_row[b_idx + 0]]
                    out_r_lin = er_lin * ae + br_lin * inv_ae
                    b_row[b_idx + 0] = lin_to_srgb_u8(out_r_lin)


def blend_normal_linear(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] edit,
    float opacity = 1.0,
):
    """Implementação em Cython de blend normal linear com LUT sRGB -> Linear e OpenMP."""
    if opacity <= 0.0:
        return
    with nogil:
        _blend_normal_linear_u8(base, edit, opacity)


# =========================================================================
# 3. HARD MASKING (Máscara Binária / Fast-Path)
# =========================================================================

def hard_masking(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] overlay,
    float opacity = 1.0,
):
    """Implementação em Cython de hard masking com cópia direta e OpenMP."""
    if opacity <= 0.0:
        return

    cdef int h = min(base.shape[0], overlay.shape[0])
    cdef int w = min(base.shape[1], overlay.shape[1])
    cdef int b_ch = base.shape[2]
    cdef int o_ch = overlay.shape[2]

    cdef bint b_has_alpha = (b_ch == 2 or b_ch == 4)
    cdef bint o_has_alpha = (o_ch == 2 or o_ch == 4)
    cdef int color_channels = 1 if (o_ch == 1 or o_ch == 2) else 3

    cdef uint32_t op_256 = <uint32_t>(opacity * 256.0)
    if op_256 > 256:
        op_256 = 256

    cdef int y, x, c, b_idx, o_idx
    cdef uint8_t* b_row
    cdef const uint8_t* o_row
    cdef uint8_t o_alpha

    with nogil:
        # Fast-Path 1: RGBA -> RGBA
        if b_ch == 4 and o_ch == 4:
            if op_256 >= 256:
                for y in prange(h, schedule='static'):
                    b_row = &base[y, 0, 0]
                    o_row = &overlay[y, 0, 0]
                    for x in range(w):
                        b_idx = x << 2
                        o_idx = x << 2
                        if o_row[o_idx + 3] > 0:
                            (<uint32_t*>&b_row[b_idx])[0] = (<const uint32_t*>&o_row[o_idx])[0]
            else:
                for y in prange(h, schedule='static'):
                    b_row = &base[y, 0, 0]
                    o_row = &overlay[y, 0, 0]
                    for x in range(w):
                        b_idx = x << 2
                        o_idx = x << 2
                        o_alpha = o_row[o_idx + 3]
                        if o_alpha > 0:
                            b_row[b_idx + 0] = o_row[o_idx + 0]
                            b_row[b_idx + 1] = o_row[o_idx + 1]
                            b_row[b_idx + 2] = o_row[o_idx + 2]
                            b_row[b_idx + 3] = <uint8_t>div255(o_alpha * op_256)

        # Fast-Path 2: RGB -> RGB
        elif b_ch == 3 and o_ch == 3:
            if op_256 >= 256:
                for y in prange(h, schedule='static'):
                    memcpy(&base[y, 0, 0], &overlay[y, 0, 0], w * 3)
            else:
                for y in prange(h, schedule='static'):
                    b_row = &base[y, 0, 0]
                    o_row = &overlay[y, 0, 0]
                    for x in range(w * 3):
                        b_row[x] = <uint8_t>div255(o_row[x] * op_256 + b_row[x] * (256 - op_256))

        # Fast-Path 3: RGB -> RGBA
        elif b_ch == 4 and o_ch == 3:
            if op_256 >= 256:
                for y in prange(h, schedule='static'):
                    b_row = &base[y, 0, 0]
                    o_row = &overlay[y, 0, 0]
                    for x in range(w):
                        b_idx = x << 2
                        o_idx = x * 3
                        b_row[b_idx + 0] = o_row[o_idx + 0]
                        b_row[b_idx + 1] = o_row[o_idx + 1]
                        b_row[b_idx + 2] = o_row[o_idx + 2]
                        b_row[b_idx + 3] = 255
            else:
                for y in prange(h, schedule='static'):
                    b_row = &base[y, 0, 0]
                    o_row = &overlay[y, 0, 0]
                    for x in range(w):
                        b_idx = x << 2
                        o_idx = x * 3
                        b_row[b_idx + 0] = o_row[o_idx + 0]
                        b_row[b_idx + 1] = o_row[o_idx + 1]
                        b_row[b_idx + 2] = o_row[o_idx + 2]
                        b_row[b_idx + 3] = <uint8_t>op_256

        # Fast-Path 4: RGBA -> RGB
        elif b_ch == 3 and o_ch == 4:
            for y in prange(h, schedule='static'):
                b_row = &base[y, 0, 0]
                o_row = &overlay[y, 0, 0]
                for x in range(w):
                    b_idx = x * 3
                    o_idx = x << 2
                    if o_row[o_idx + 3] > 0:
                        b_row[b_idx + 0] = o_row[o_idx + 0]
                        b_row[b_idx + 1] = o_row[o_idx + 1]
                        b_row[b_idx + 2] = o_row[o_idx + 2]

        else:
            # Caminho geral cobrindo Grayscale e Gray-Alpha
            for y in prange(h, schedule='static'):
                b_row = &base[y, 0, 0]
                o_row = &overlay[y, 0, 0]

                if not o_has_alpha and not b_has_alpha and b_ch == o_ch:
                    memcpy(b_row, o_row, w * b_ch)
                elif o_has_alpha:
                    for x in range(w):
                        b_idx = x * b_ch
                        o_idx = x * o_ch
                        o_alpha = o_row[o_idx + o_ch - 1]
                        if o_alpha > 0:
                            for c in range(color_channels):
                                b_row[b_idx + c] = o_row[o_idx + c]

                            if b_has_alpha:
                                b_row[b_idx + b_ch - 1] = <uint8_t>div255(o_alpha * op_256) if op_256 < 256 else o_alpha
                else:
                    for x in range(w):
                        b_idx = x * b_ch
                        o_idx = x * o_ch
                        for c in range(color_channels):
                            b_row[b_idx + c] = o_row[o_idx + c]

                        if b_has_alpha:
                            b_row[b_idx + b_ch - 1] = 255 if op_256 >= 256 else <uint8_t>op_256


# =========================================================================
# 4. MIN-POOLING CONSERVADOR (Máscara de Oclusão / Early-Exit)
# =========================================================================

def min_pool_alpha(
    uint8_t[:, :, :] src,
    uint8_t[:, :] dst,
):
    """Calcula o min-pooling conservador do canal Alfa lendo diretamente a imagem 3D sem alocação."""
    cdef int src_h = src.shape[0]
    cdef int src_w = src.shape[1]
    cdef int ch = src.shape[2]
    cdef int dst_h = dst.shape[0]
    cdef int dst_w = dst.shape[1]
    cdef int alpha_idx = ch - 1

    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        return

    cdef int dy, dx, sy, sx
    cdef int y_start, y_end, x_start, x_end
    cdef uint8_t min_v, val
    cdef const uint8_t* src_alpha_base = &src[0, 0, alpha_idx]
    cdef const uint8_t* row_ptr
    cdef int stride_y = src.strides[0]
    cdef int stride_x = ch

    # Tabelas pré-computadas de limites (Zero divisões no loop 2D)
    cdef int x_start_lut[1024]
    cdef int x_end_lut[1024]
    cdef int y_start_lut[1024]
    cdef int y_end_lut[1024]

    cdef const uint8_t* p
    cdef const uint8_t* p_end

    with nogil:
        # Fast-Path: Formatos sem canal Alfa (RGB / Grayscale) são 100% opacos por definição
        if ch == 1 or ch == 3:
            for dy in range(dst_h):
                memset(&dst[dy, 0], 255, dst_w)
        else:
            # Precomputa limites verticais (Y)
            for dy in range(dst_h):
                y_start = (dy * src_h) // dst_h
                y_end = ((dy + 1) * src_h) // dst_h
                if y_end <= y_start:
                    y_end = y_start + 1
                if y_end > src_h:
                    y_end = src_h
                y_start_lut[dy] = y_start
                y_end_lut[dy] = y_end

            # Precomputa limites horizontais (X)
            for dx in range(dst_w):
                x_start = (dx * src_w) // dst_w
                x_end = ((dx + 1) * src_w) // dst_w
                if x_end <= x_start:
                    x_end = x_start + 1
                if x_end > src_w:
                    x_end = src_w
                x_start_lut[dx] = x_start
                x_end_lut[dx] = x_end

            # Varredura 2D com ponteiros diretos e Early-Exit
            for dy in range(dst_h):
                y_start = y_start_lut[dy]
                y_end = y_end_lut[dy]

                for dx in range(dst_w):
                    x_start = x_start_lut[dx]
                    x_end = x_end_lut[dx]

                    min_v = 255
                    for sy in range(y_start, y_end):
                        row_ptr = src_alpha_base + sy * stride_y
                        p = row_ptr + x_start * stride_x
                        p_end = row_ptr + x_end * stride_x
                        while p < p_end:
                            if p[0] < min_v:
                                min_v = p[0]
                                if min_v == 0:
                                    break
                            p += stride_x
                        if min_v == 0:
                            break

                    dst[dy, dx] = min_v
