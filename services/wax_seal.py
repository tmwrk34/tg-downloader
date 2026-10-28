"""
3D Wax Seal Renderer, physical parchment drop-shadow compositor,
and vectorized HSV noble color extractor for medieval quote scrolls.
"""

from __future__ import annotations

import colorsys
import hashlib
import io
import logging
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# Authentic medieval noble duotones (Base Wax Color, Contrasting Powder Highlight)
NOBLE_PAIRS: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    ((22, 62, 140), (238, 195, 50)),  # Royal Sapphire Blue Wax + Radiant Gold Powder
    ((142, 22, 28), (235, 185, 45)),  # Venetian Crimson Wax + Ancient Gold Powder
    ((20, 85, 45), (240, 190, 55)),  # Imperial Emerald Wax + Solar Gold Powder
    ((68, 28, 110), (238, 192, 50)),  # Royal Amethyst Wax + Venetian Gold Powder
    ((24, 25, 30), (220, 225, 235)),  # Obsidian Onyx Wax + Sterling Silver Powder
]


def extract_noble_colors(
    avatar_bytes: bytes | None,
    author_name: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Extract primary base wax color and contrasting metallic powder color from avatar
    using vectorized Pillow/NumPy HSV conversion, or fall back to a deterministic noble pair.
    """
    if avatar_bytes:
        try:
            img = (
                Image.open(io.BytesIO(avatar_bytes))
                .convert("RGB")
                .resize((128, 128), Image.Resampling.LANCZOS)
            )
            # Vectorized C-level HSV conversion in Pillow (0..255 per channel)
            hsv_arr = np.array(img.convert("HSV"), dtype=np.float32) / 255.0
            h, s, v = hsv_arr[:, :, 0], hsv_arr[:, :, 1], hsv_arr[:, :, 2]
            weight = s * v * (s > 0.16) * (v > 0.16)

            if weight.sum() > 1e-2:
                bins = 24
                hist, _ = np.histogram(h, bins=bins, range=(0.0, 1.0), weights=weight)
                best_h = float((np.argmax(hist) + 0.5) / bins)

                # Deep jewel-toned wax color
                r1, g1, b1 = colorsys.hsv_to_rgb(best_h, 0.88, 0.52)
                base_wax = (int(r1 * 255), int(g1 * 255), int(b1 * 255))

                # Radiant noble gold powder dusting as default contrasting metallic coat;
                # if avatar itself is primarily golden-yellow, use sterling silver powder
                if 0.10 <= best_h <= 0.18:
                    powder = (220, 225, 238)
                else:
                    powder = (238, 195, 50)

                return (base_wax, powder)
        except Exception as e:
            logger.debug("Failed extracting avatar colors: %s", e)

    idx = int(hashlib.md5(author_name.encode("utf-8")).hexdigest(), 16) % len(NOBLE_PAIRS)
    return NOBLE_PAIRS[idx]


def render_3d_wax_seal(
    avatar_bytes: bytes | None,
    author_name: str,
    base_color: tuple[int, int, int],
    powder_color: tuple[int, int, int],
    font_cormorant_path: str,
    size: int = 350,
    seed: int = 42,
) -> Image.Image:
    """Generate authentic 3D molten wax seal with organic rims, debossed valleys, and contrasting powder."""
    np.random.seed(seed)
    random.seed(seed)

    center = size / 2.0
    y_idx, x_idx = np.indices((size, size))
    dx = (x_idx - center).astype(np.float32)
    dy = (y_idx - center).astype(np.float32)
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    # 1. Organic molten wax puddle contour with viscous ripples
    base_r = size * 0.43
    puddle_r = base_r * (
        1.0
        + 0.07 * np.cos(3 * angle + random.uniform(0, math.pi))
        + 0.05 * np.cos(5 * angle + random.uniform(0, math.pi))
        + 0.04 * np.sin(2 * angle + random.uniform(0, math.pi))
        + 0.03 * np.cos(7 * angle + random.uniform(0, math.pi))
        + 0.02 * np.sin(11 * angle + random.uniform(0, math.pi))
    )

    edge_width = 8.0
    wax_mask = np.clip((puddle_r - dist) / edge_width, 0.0, 1.0)
    wax_mask = wax_mask * wax_mask * (3.0 - 2.0 * wax_mask)

    # 2. Height field Z(x, y)
    height = np.zeros((size, size), dtype=np.float32)

    meniscus = np.clip((puddle_r - dist) / (size * 0.12), 0.0, 1.0)
    height += np.sin(meniscus * math.pi * 0.5) * 0.55

    ridge_center = base_r * 0.76
    ridge_sigma = 18.0
    ridge = np.exp(-((dist - ridge_center) ** 2) / (2 * (ridge_sigma**2))) * 0.55
    height += ridge

    basin_r = base_r * 0.68
    in_basin = dist < basin_r
    basin_depth = np.clip((basin_r - dist) / 10.0, 0.0, 1.0)
    height -= basin_depth * 0.48

    ring_r = basin_r * 0.94
    ring = np.exp(-((dist - ring_r) ** 2) / (2 * (3.5**2))) * 0.22
    inner_ring_r = basin_r * 0.88
    inner_ring = np.exp(-((dist - inner_ring_r) ** 2) / (2 * (2.5**2))) * 0.18
    height += (ring + inner_ring) * in_basin

    # 3. Engraved Artwork in Basin (Avatar relief or noble monogram)
    engrave_depth = np.zeros((size, size), dtype=np.float32)

    if avatar_bytes:
        try:
            av_orig = Image.open(io.BytesIO(avatar_bytes)).convert("L")
            w_av, h_av = av_orig.size
            min_side = min(w_av, h_av)
            left = (w_av - min_side) // 2
            top = (h_av - min_side) // 2
            av_sq = av_orig.crop((left, top, left + min_side, top + min_side))

            box_dim = int(basin_r * 1.68)
            av_resized = av_sq.resize((box_dim, box_dim), Image.Resampling.LANCZOS)
            av_arr = np.array(av_resized, dtype=np.float32) / 255.0

            p2, p98 = np.percentile(av_arr, (2, 98))
            av_norm = np.clip((av_arr - p2) / (p98 - p2 + 1e-5), 0.0, 1.0)

            blur_img = av_resized.filter(ImageFilter.GaussianBlur(3))
            blur_arr = np.array(blur_img, dtype=np.float32) / 255.0
            high_pass = np.clip((av_norm - blur_arr) * 2.2, -1.0, 1.0)

            edges_img = av_resized.filter(ImageFilter.FIND_EDGES)
            edges_arr = np.array(edges_img, dtype=np.float32) / 255.0

            carve = (1.0 - av_norm) * 0.65 + edges_arr * 0.50 + np.clip(-high_pass, 0.0, 1.0) * 0.35
            carve = np.clip(carve, 0.0, 1.0)

            sub_mask = np.zeros((size, size), dtype=np.float32)
            px = int(center - box_dim // 2)
            py = int(center - box_dim // 2)
            sub_mask[py : py + box_dim, px : px + box_dim] = carve
            engrave_depth = sub_mask * (dist < (basin_r * 0.84))
        except Exception as e:
            logger.debug("Failed processing avatar relief for seal: %s", e)
            avatar_bytes = None

    if not avatar_bytes:
        mono_img = Image.new("L", (size, size), 0)
        d_m = ImageDraw.Draw(mono_img)
        initial = (author_name.strip()[:1] or "К").upper()
        f_size = int(basin_r * 1.15)
        font = ImageFont.truetype(font_cormorant_path, f_size)
        d_m.text((int(center), int(center) - 4), initial, font=font, fill=255, anchor="mm")
        for r_scale, width in ((0.85, 4), (0.77, 2)):
            d_m.ellipse(
                (
                    center - basin_r * r_scale,
                    center - basin_r * r_scale,
                    center + basin_r * r_scale,
                    center + basin_r * r_scale,
                ),
                outline=255,
                width=width,
            )
        mono_arr = np.array(mono_img, dtype=np.float32) / 255.0
        engrave_depth = mono_arr * (dist < (basin_r * 0.86))

    height -= engrave_depth * 0.32

    # Meniscus smoothing
    h_min, h_max = height.min(), height.max()
    h_u = np.clip((height - h_min) / (h_max - h_min + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    h_img = Image.fromarray(h_u, mode="L").filter(ImageFilter.GaussianBlur(1.5))
    h_smooth = np.array(h_img, dtype=np.float32) / 255.0 * (h_max - h_min) + h_min

    # 4. Normal map & 3D Blinn-Phong Lighting
    gy, gx = np.gradient(h_smooth * 35.0)
    nz = np.ones_like(gx)
    n_len = np.sqrt(gx**2 + gy**2 + nz**2)
    nx, ny, nz = -gx / n_len, -gy / n_len, nz / n_len

    lx, ly, lz = -0.55, -0.65, 0.75
    l_len = math.sqrt(lx**2 + ly**2 + lz**2)
    lx, ly, lz = lx / l_len, ly / l_len, lz / l_len
    diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)

    hx, hy, hz = lx, ly, lz + 1.0
    h_len = math.sqrt(hx**2 + hy**2 + hz**2)
    hx, hy, hz = hx / h_len, hy / h_len, hz / h_len
    ndoth = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0)
    specular = ndoth**22.0

    ao = np.clip(1.0 - (gx**2 + gy**2) * 0.04 - engrave_depth * 0.25, 0.35, 1.0)

    # 5. Contrasting powder dusting on raised un-engraved surfaces
    powder_noise = np.random.uniform(0.88, 1.12, (size, size))
    plateau_mask = np.clip(1.0 - engrave_depth * 2.2, 0.0, 1.0) * (dist < (basin_r * 0.90))
    powder_weight = plateau_mask * 0.92

    rgb = np.zeros((size, size, 3), dtype=np.float32)
    base_col = np.array(base_color, dtype=np.float32)
    powd_col = np.array(powder_color, dtype=np.float32)

    for c in range(3):
        wax = base_col[c] * (diffuse * 0.70 + 0.38) * ao
        powder = powd_col[c] * (diffuse * 0.85 + 0.30) * powder_noise
        surface = wax * (1.0 - powder_weight) + powder * powder_weight
        surface += specular * 165.0
        rgb[:, :, c] = np.clip(surface, 0.0, 255.0)

    seal_rgba = np.zeros((size, size, 4), dtype=np.uint8)
    seal_rgba[:, :, :3] = rgb.astype(np.uint8)
    seal_rgba[:, :, 3] = (wax_mask * 255.0).astype(np.uint8)
    return Image.fromarray(seal_rgba, mode="RGBA")


def composite_seal_and_shadows(
    bg: Image.Image,
    ink_layer: Image.Image,
    seal_img: Image.Image,
    stamp_x: int,
    stamp_y: int,
) -> Image.Image:
    """
    Composite multi-layer physical cast shadows (ambient penumbra, directional body,
    contact occlusion), ink layer, and the 3D wax seal onto the parchment background.
    """
    canvas_w, canvas_h = bg.size
    stamp_size = seal_img.size[0]
    sh_pad = 90
    sh_dim = stamp_size + sh_pad * 2
    wax_alpha = np.array(seal_img)[:, :, 3].astype(np.float32) / 255.0

    padded_wax = np.zeros((sh_dim, sh_dim), dtype=np.float32)
    padded_wax[sh_pad : sh_pad + stamp_size, sh_pad : sh_pad + stamp_size] = wax_alpha

    base_sh_x = stamp_x - sh_pad
    base_sh_y = stamp_y - sh_pad

    shadow_specs = (
        # (intensity_scale, blur_radius, rgb_color, offset_x, offset_y)
        (85, 32, (28, 18, 12, 255), 22, 32),  # Diffuse ambient penumbra
        (155, 16, (18, 12, 8, 255), 14, 20),  # Directional body shadow
        (210, 4, (10, 6, 4, 255), 3, 4),  # Deep contact occlusion
    )

    final_img = bg
    for scale, blur_r, rgba_col, off_x, off_y in shadow_specs:
        mask = Image.fromarray((padded_wax * scale).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(blur_r)
        )
        sh_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        sh_layer.paste(
            Image.new("RGBA", (sh_dim, sh_dim), rgba_col),
            (base_sh_x + off_x, base_sh_y + off_y),
            mask,
        )
        final_img = Image.alpha_composite(final_img, sh_layer)

    final_img = Image.alpha_composite(final_img, ink_layer)

    seal_placement = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    seal_placement.paste(seal_img, (stamp_x, stamp_y), seal_img)
    return Image.alpha_composite(final_img, seal_placement)
