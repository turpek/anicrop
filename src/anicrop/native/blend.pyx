# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

from libc.stdint cimport uint8_t, uint32_t
from libc.math cimport powf, roundf

cdef inline uint8_t clamp_u8(uint32_t v) noexcept nogil:
    return 255 if v > 255 else <uint8_t>v

cdef inline uint8_t clamp_f_u8(float v) noexcept nogil:
    if v <= 0.0:
        return 0
    if v >= 255.0:
        return 255
    return <uint8_t>roundf(v)

cdef inline uint8_t rgb_to_gray_u8(uint8_t r, uint8_t g, uint8_t b) noexcept nogil:
    # ITU-R BT.601: 0.299 * R + 0.587 * G + 0.114 * B
    return <uint8_t>((19595 * <uint32_t>r + 38470 * <uint32_t>g + 7471 * <uint32_t>b + 32768) >> 16)

# Tabela estatica pre-calculada sRGB -> Linear (256 valores)
cdef float SRGB_TO_LIN[256]

cdef void _init_lut() noexcept nogil:
    cdef int i
    for i in range(256):
        SRGB_TO_LIN[i] = powf(<float>i / 255.0, 2.2)

_init_lut()


# =========================================================================
# 1. BLEND NORMAL (Porter-Duff Over)
# =========================================================================

def blend_normal(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] edit,
    float opacity = 1.0
):
    """Implementacao em Cython de blend normal com Porter-Duff e aritmetica inteira."""
    if opacity <= 0.0:
        return

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

    cdef int y, x
    cdef uint32_t ae_raw, ae, inv_ae, ab, out_a
    cdef uint32_t er, eg, eb, b_r, b_g, b_b

    with nogil:
        # Fast-Path RGBA -> RGBA
        if b_ch == 4 and e_ch == 4:
            for y in range(h):
                for x in range(w):
                    ae_raw = edit[y, x, 3]
                    if ae_raw == 0:
                        continue

                    ae = (ae_raw * op_256) >> 8 if op_256 < 256 else ae_raw
                    if ae == 0:
                        continue

                    if ae == 255:
                        base[y, x, 0] = edit[y, x, 0]
                        base[y, x, 1] = edit[y, x, 1]
                        base[y, x, 2] = edit[y, x, 2]
                        base[y, x, 3] = 255
                        continue

                    ab = base[y, x, 3]
                    inv_ae = 255 - ae
                    out_a = ae + ((ab * inv_ae + 127) // 255)
                    if out_a == 0:
                        base[y, x, 3] = 0
                        continue

                    b_r = (base[y, x, 0] * ab * inv_ae + 127) // 255
                    b_g = (base[y, x, 1] * ab * inv_ae + 127) // 255
                    b_b = (base[y, x, 2] * ab * inv_ae + 127) // 255

                    base[y, x, 0] = clamp_u8((edit[y, x, 0] * ae + b_r + (out_a >> 1)) // out_a)
                    base[y, x, 1] = clamp_u8((edit[y, x, 1] * ae + b_g + (out_a >> 1)) // out_a)
                    base[y, x, 2] = clamp_u8((edit[y, x, 2] * ae + b_b + (out_a >> 1)) // out_a)
                    base[y, x, 3] = clamp_u8(out_a)

        else:
            # Caminho geral cobrindo RGBA, RGB, GRAY_ALPHA e GRAY
            for y in range(h):
                for x in range(w):
                    ae_raw = edit[y, x, e_ch - 1] if e_has_alpha else 255
                    if ae_raw == 0:
                        continue

                    ae = (ae_raw * op_256) >> 8 if op_256 < 256 else ae_raw
                    if ae == 0:
                        continue

                    if e_colors == 3:
                        er = edit[y, x, 0]
                        eg = edit[y, x, 1]
                        eb = edit[y, x, 2]
                        if b_colors == 1:
                            er = rgb_to_gray_u8(<uint8_t>er, <uint8_t>eg, <uint8_t>eb)
                            eg = er
                            eb = er
                    else:
                        er = edit[y, x, 0]
                        eg = er
                        eb = er

                    if b_has_alpha:
                        ab = base[y, x, b_ch - 1]
                        inv_ae = 255 - ae
                        out_a = ae + ((ab * inv_ae + 127) // 255)
                        if out_a == 0:
                            base[y, x, b_ch - 1] = 0
                            continue

                        if b_colors == 3:
                            b_r = (base[y, x, 0] * ab * inv_ae + 127) // 255
                            b_g = (base[y, x, 1] * ab * inv_ae + 127) // 255
                            b_b = (base[y, x, 2] * ab * inv_ae + 127) // 255

                            base[y, x, 0] = clamp_u8((er * ae + b_r + (out_a >> 1)) // out_a)
                            base[y, x, 1] = clamp_u8((eg * ae + b_g + (out_a >> 1)) // out_a)
                            base[y, x, 2] = clamp_u8((eb * ae + b_b + (out_a >> 1)) // out_a)
                        else:
                            b_g = (base[y, x, 0] * ab * inv_ae + 127) // 255
                            base[y, x, 0] = clamp_u8((er * ae + b_g + (out_a >> 1)) // out_a)

                        base[y, x, b_ch - 1] = clamp_u8(out_a)

                    else:
                        inv_ae = 255 - ae
                        if b_colors == 3:
                            base[y, x, 0] = clamp_u8((er * ae + base[y, x, 0] * inv_ae + 127) // 255)
                            base[y, x, 1] = clamp_u8((eg * ae + base[y, x, 1] * inv_ae + 127) // 255)
                            base[y, x, 2] = clamp_u8((eb * ae + base[y, x, 2] * inv_ae + 127) // 255)
                        else:
                            base[y, x, 0] = clamp_u8((er * ae + base[y, x, 0] * inv_ae + 127) // 255)


# =========================================================================
# 2. BLEND NORMAL LINEAR (Fisicamente Correto em Luz Linear)
# =========================================================================

def blend_normal_linear(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] edit,
    float opacity = 1.0
):
    """Implementacao em Cython de blend normal linear com LUT sRGB -> Linear."""
    if opacity <= 0.0:
        return

    cdef int h = min(base.shape[0], edit.shape[0])
    cdef int w = min(base.shape[1], edit.shape[1])
    cdef int b_ch = base.shape[2]
    cdef int e_ch = edit.shape[2]

    cdef bint b_has_alpha = (b_ch == 2 or b_ch == 4)
    cdef bint e_has_alpha = (e_ch == 2 or e_ch == 4)
    cdef int b_colors = 1 if (b_ch == 1 or b_ch == 2) else 3
    cdef int e_colors = 1 if (e_ch == 1 or e_ch == 2) else 3

    cdef float inv_gamma = 1.0 / 2.2
    cdef int y, x
    cdef uint8_t ae_raw, ab_raw
    cdef float ae, ab, inv_ae, out_a, out_a_safe
    cdef float er_lin, eg_lin, eb_lin, br_lin, bg_lin, bb_lin
    cdef float out_r_lin, out_g_lin, out_b_lin

    with nogil:
        for y in range(h):
            for x in range(w):
                ae_raw = edit[y, x, e_ch - 1] if e_has_alpha else 255
                if ae_raw == 0:
                    continue

                ae = (<float>ae_raw / 255.0) * opacity
                if ae <= 0.0:
                    continue

                if e_colors == 3:
                    if b_colors == 1:
                        er_lin = SRGB_TO_LIN[rgb_to_gray_u8(edit[y, x, 0], edit[y, x, 1], edit[y, x, 2])]
                        eg_lin = er_lin
                        eb_lin = er_lin
                    else:
                        er_lin = SRGB_TO_LIN[edit[y, x, 0]]
                        eg_lin = SRGB_TO_LIN[edit[y, x, 1]]
                        eb_lin = SRGB_TO_LIN[edit[y, x, 2]]
                else:
                    er_lin = SRGB_TO_LIN[edit[y, x, 0]]
                    eg_lin = er_lin
                    eb_lin = er_lin

                inv_ae = 1.0 - ae

                if b_has_alpha:
                    ab_raw = base[y, x, b_ch - 1]
                    ab = <float>ab_raw / 255.0
                    out_a = ae + ab * inv_ae
                    out_a_safe = out_a if out_a > 0.0 else 1.0

                    if b_colors == 3:
                        br_lin = SRGB_TO_LIN[base[y, x, 0]]
                        bg_lin = SRGB_TO_LIN[base[y, x, 1]]
                        bb_lin = SRGB_TO_LIN[base[y, x, 2]]

                        out_r_lin = (er_lin * ae + br_lin * ab * inv_ae) / out_a_safe
                        out_g_lin = (eg_lin * ae + bg_lin * ab * inv_ae) / out_a_safe
                        out_b_lin = (eb_lin * ae + bb_lin * ab * inv_ae) / out_a_safe

                        base[y, x, 0] = clamp_f_u8(powf(out_r_lin, inv_gamma) * 255.0)
                        base[y, x, 1] = clamp_f_u8(powf(out_g_lin, inv_gamma) * 255.0)
                        base[y, x, 2] = clamp_f_u8(powf(out_b_lin, inv_gamma) * 255.0)
                    else:
                        br_lin = SRGB_TO_LIN[base[y, x, 0]]
                        out_r_lin = (er_lin * ae + br_lin * ab * inv_ae) / out_a_safe
                        base[y, x, 0] = clamp_f_u8(powf(out_r_lin, inv_gamma) * 255.0)

                    base[y, x, b_ch - 1] = clamp_f_u8(out_a * 255.0)

                else:
                    if b_colors == 3:
                        br_lin = SRGB_TO_LIN[base[y, x, 0]]
                        bg_lin = SRGB_TO_LIN[base[y, x, 1]]
                        bb_lin = SRGB_TO_LIN[base[y, x, 2]]

                        out_r_lin = er_lin * ae + br_lin * inv_ae
                        out_g_lin = eg_lin * ae + bg_lin * inv_ae
                        out_b_lin = eb_lin * ae + bb_lin * inv_ae

                        base[y, x, 0] = clamp_f_u8(powf(out_r_lin, inv_gamma) * 255.0)
                        base[y, x, 1] = clamp_f_u8(powf(out_g_lin, inv_gamma) * 255.0)
                        base[y, x, 2] = clamp_f_u8(powf(out_b_lin, inv_gamma) * 255.0)
                    else:
                        br_lin = SRGB_TO_LIN[base[y, x, 0]]
                        out_r_lin = er_lin * ae + br_lin * inv_ae
                        base[y, x, 0] = clamp_f_u8(powf(out_r_lin, inv_gamma) * 255.0)


# =========================================================================
# 3. HARD MASKING (Mascara Binaria / Fast-Path)
# =========================================================================

def hard_masking(
    uint8_t[:, :, :] base,
    uint8_t[:, :, :] overlay,
    float opacity = 1.0
):
    """Implementacao em Cython de hard masking com copia direta em nivel de C."""
    if opacity <= 0.0:
        return

    cdef int h = min(base.shape[0], overlay.shape[0])
    cdef int w = min(base.shape[1], overlay.shape[1])
    cdef int b_ch = base.shape[2]
    cdef int o_ch = overlay.shape[2]

    cdef bint b_has_alpha = (b_ch == 2 or b_ch == 4)
    cdef bint o_has_alpha = (o_ch == 2 or o_ch == 4)
    cdef int color_channels = 1 if (o_ch == 1 or o_ch == 2) else 3

    cdef uint8_t alpha_val = <uint8_t>(255.0 * opacity) if opacity < 1.0 else 255
    cdef int y, x, c

    with nogil:
        if o_has_alpha:
            for y in range(h):
                for x in range(w):
                    if overlay[y, x, o_ch - 1] > 0:
                        for c in range(color_channels):
                            base[y, x, c] = overlay[y, x, c]

                        if b_has_alpha:
                            if opacity < 1.0:
                                base[y, x, b_ch - 1] = <uint8_t>(<float>overlay[y, x, o_ch - 1] * opacity)
                            else:
                                base[y, x, b_ch - 1] = overlay[y, x, o_ch - 1]
        else:
            for y in range(h):
                for x in range(w):
                    for c in range(color_channels):
                        base[y, x, c] = overlay[y, x, c]

                    if b_has_alpha:
                        base[y, x, b_ch - 1] = alpha_val
