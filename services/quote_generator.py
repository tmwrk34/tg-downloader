from __future__ import annotations

import asyncio
import colorsys
import hashlib
import logging
import math
import os
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# Template-specific safe zones on a 1792x2400 canvas
TEMPLATE_CONFIGS: dict[str, dict[str, int]] = {
    "Arrow": {
        "start_x": 370,
        "start_y": 860,  # Below the arrow spearhead at top-right
        "max_w": 960,
        "max_h": 720,
    },
    "Dagger": {
        "start_x": 400,
        "start_y": 700,  # Below the dagger hilt/blade
        "max_w": 950,
        "max_h": 880,
    },
    "Basic": {
        "start_x": 380,
        "start_y": 560,
        "max_w": 1040,
        "max_h": 1080,
    },
    "Ell": {
        "start_x": 380,
        "start_y": 660,
        "max_w": 950,
        "max_h": 850,
    },
    "Fired": {
        "start_x": 380,
        "start_y": 550,
        "max_w": 960,
        "max_h": 820,  # Above the burnt holes at bottom
    },
}

# Authentic medieval noble duotones (Base Wax Color, Contrasting Powder Highlight)
NOBLE_PAIRS: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    ((24, 65, 145), (235, 185, 45)),  # Sapphire Blue Wax + Radiant Gold Powder
    ((145, 24, 30), (230, 180, 50)),  # Venetian Crimson Wax + Ancient Gold Powder
    ((22, 88, 48), (240, 190, 55)),  # Forest Emerald Wax + Solar Gold Powder
    ((135, 75, 18), (245, 220, 120)),  # Burnished Amber Wax + Champagne Gold Powder
    ((22, 24, 30), (215, 220, 230)),  # Obsidian Black Wax + Sterling Silver Powder
]


class QuoteGeneratorService:
    """Authentic medieval parchment quote card generator with 16+5 noise-dispersion calligraphy."""

    TEMPLATES = ["Dagger", "Basic", "Arrow", "Ell", "Fired"]

    def __init__(
        self,
        media_dir: str = "media",
        fonts_dir: str = "assets/fonts",
        canvas_width: int = 1792,
        canvas_height: int = 2400,
    ) -> None:
        self.media_dir = media_dir
        self.fonts_dir = fonts_dir
        self.canvas_w = canvas_width
        self.canvas_h = canvas_height

        self._template_cache: dict[str, dict[str, Any]] = {}

        # Font file paths
        self.font_dropcap_path = os.path.join(fonts_dir, "BotanicalDropCaps.otf")
        self.font_marck_path = os.path.join(fonts_dir, "MarckScript.ttf")
        self.font_caveat_path = os.path.join(fonts_dir, "Caveat.ttf")
        self.font_bad_path = os.path.join(fonts_dir, "BadScript.ttf")
        self.font_cormorant_path = os.path.join(fonts_dir, "CormorantGaramond.ttf")

    def _get_template_data(self, template_name: str) -> dict[str, Any]:
        """Loads and pre-processes background, crease overlay, normal map, and fiber texture."""
        if template_name not in self.TEMPLATES:
            template_name = random.choice(self.TEMPLATES)

        if template_name in self._template_cache:
            return self._template_cache[template_name]

        # 1. Background image
        jpg_path = os.path.join(self.media_dir, f"{template_name}.jpg")
        png_path = os.path.join(self.media_dir, f"{template_name}.png")
        bg_path = jpg_path if os.path.exists(jpg_path) else png_path
        if not os.path.exists(bg_path):
            raise FileNotFoundError(f"Parchment background not found: {bg_path}")

        bg = (
            Image.open(bg_path)
            .convert("RGBA")
            .resize((self.canvas_w, self.canvas_h), Image.Resampling.LANCZOS)
        )

        # 2. Crease overlay
        cr_path = os.path.join(self.media_dir, f"{template_name} Crease Overlay.png")
        if os.path.exists(cr_path):
            cr = (
                Image.open(cr_path)
                .convert("L")
                .resize((self.canvas_w, self.canvas_h), Image.Resampling.BICUBIC)
            )
            cr_arr = np.array(cr, dtype=np.float32) / 255.0
        else:
            cr_arr = np.ones((self.canvas_h, self.canvas_w), dtype=np.float32)

        # 3. Normal map
        nr_path = os.path.join(self.media_dir, f"{template_name} normal.png")
        if os.path.exists(nr_path):
            nr = (
                Image.open(nr_path)
                .convert("RGB")
                .resize((self.canvas_w, self.canvas_h), Image.Resampling.BICUBIC)
            )
            nr_arr = (np.array(nr, dtype=np.float32) / 127.5) - 1.0  # [-1, 1] range

            # Normal tilt: sqrt(nx^2 + ny^2)
            tilt = np.sqrt(nr_arr[:, :, 0] ** 2 + nr_arr[:, :, 1] ** 2)

            # Normal gradients (crease ridges)
            gy1, gx1 = np.gradient(nr_arr[:, :, 0])
            gy2, gx2 = np.gradient(nr_arr[:, :, 1])
            n_grad = np.sqrt(gx1**2 + gy1**2 + gx2**2 + gy2**2)
            n_grad = n_grad / (np.percentile(n_grad, 99) + 1e-6)
        else:
            tilt = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
            n_grad = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)

        crease_darkness = 1.0 - cr_arr
        crease_intensity = np.clip(tilt * 0.45 + n_grad * 0.40 + crease_darkness * 0.50, 0.0, 1.0)

        # 4. High-pass extracted paper fiber texture from background photo
        bg_gray = np.array(bg.convert("L"), dtype=np.float32) / 255.0
        bg_blur = (
            np.array(bg.convert("L").filter(ImageFilter.GaussianBlur(5)), dtype=np.float32) / 255.0
        )
        paper_fiber = bg_gray - bg_blur
        paper_fiber = paper_fiber / (np.std(paper_fiber) + 1e-6)

        data = {
            "bg": bg,
            "crease_arr": cr_arr,
            "crease_intensity": crease_intensity,
            "paper_fiber": paper_fiber,
        }
        self._template_cache[template_name] = data
        return data

    @staticmethod
    def _generate_smooth_noise(width: int, height: int, scale: int, seed: int) -> np.ndarray:
        """Generates bicubic smooth gaussian-filtered coherent 2D noise."""
        np.random.seed(seed)
        gw = int(math.ceil(width / scale)) + 3
        gh = int(math.ceil(height / scale)) + 3
        raw = np.random.uniform(-1.0, 1.0, (gh, gw))
        img = Image.fromarray(((raw + 1.0) * 127.5).astype(np.uint8), mode="L")
        img = img.resize((width, height), Image.Resampling.BICUBIC)
        if scale > 6:
            img = img.filter(ImageFilter.GaussianBlur(radius=max(1, scale * 0.15)))
        arr = (np.array(img, dtype=np.float32) / 127.5) - 1.0
        return arr

    def _build_noise_and_dispersion_fields(
        self,
        base_seed: int,
        crease_intensity: np.ndarray,
        paper_fiber: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Builds 16-card 4-factor motor noise fields + 5-card normal-modulated ink texture dispersion."""
        scales_4factors = {
            "traj": [1200, 600, 280, 110],
            "slant": [500, 250, 120, 60],
            "scale": [400, 200, 90, 40],
            "press": [250, 120, 50, 20],
        }

        composites: dict[str, np.ndarray] = {}
        weights_4 = np.array([0.48, 0.28, 0.16, 0.08]).reshape((4, 1, 1))

        for p_idx, (p_k, p_sc) in enumerate(scales_4factors.items()):
            l_stack = [
                self._generate_smooth_noise(
                    self.canvas_w,
                    self.canvas_h,
                    sc,
                    seed=(base_seed + (p_idx + 1) * 1000 + (l_idx + 1) * 239) % 2147483647,
                )
                for l_idx, sc in enumerate(p_sc)
            ]
            st = np.array(l_stack)
            m = np.sum(st * weights_4, axis=0)
            d = np.std(st, axis=0)
            c = m * (0.85 + 0.35 * (d / (d.mean() + 1e-6)))
            composites[p_k] = np.clip(c / (np.percentile(np.abs(c), 99) + 1e-6), -1.0, 1.0)

        # 5-card ink-paper interaction noise pack
        scales_5cards = [120, 50, 20, 9, 4]
        noise_5cards = [
            self._generate_smooth_noise(
                self.canvas_w, self.canvas_h, sc, seed=(base_seed + 7000 + i * 443) % 2147483647
            )
            for i, sc in enumerate(scales_5cards)
        ]
        stack_5 = np.array(noise_5cards)
        raw_disp_5 = np.std(stack_5, axis=0)
        raw_mean_5 = np.mean(stack_5, axis=0)

        # Normal map modulated dispersion
        crease_weight = np.clip(
            crease_intensity / (np.percentile(crease_intensity, 98) + 1e-6), 0.0, 1.5
        )
        eff_dispersion = raw_disp_5 * (1.0 + 2.5 * crease_weight)

        # Texture alpha erosion modulated by 5-card dispersion
        tex_erosion = (
            paper_fiber * 0.65 * (eff_dispersion / (eff_dispersion.mean() + 1e-6))
            + raw_mean_5 * 0.35
        )

        # Ink transparency fade across folds
        crease_abrasion = np.clip(crease_weight * 0.60 * (0.7 + 0.3 * raw_disp_5), 0.0, 0.85)

        return composites, tex_erosion, crease_abrasion

    @staticmethod
    def extract_noble_colors(
        avatar_bytes: bytes | None, author_name: str
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Extracts primary base wax color and contrasting powder color from avatar, or picks medieval noble pair."""
        if avatar_bytes:
            try:
                import io

                img = (
                    Image.open(io.BytesIO(avatar_bytes))
                    .convert("RGB")
                    .resize((128, 128), Image.Resampling.LANCZOS)
                )
                arr = np.array(img, dtype=np.float32) / 255.0
                hsv = np.zeros_like(arr)
                for y in range(128):
                    for x in range(128):
                        hsv[y, x] = colorsys.rgb_to_hsv(arr[y, x, 0], arr[y, x, 1], arr[y, x, 2])

                h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
                weight = s * v * (s > 0.18) * (v > 0.18)

                if weight.sum() > 1e-2:
                    bins = 24
                    hist = np.zeros(bins)
                    for b in range(bins):
                        mask = (h >= b / bins) & (h < (b + 1) / bins)
                        hist[b] = weight[mask].sum()

                    best_h = (np.argmax(hist) + 0.5) / bins
                    r1, g1, b1 = colorsys.hsv_to_rgb(best_h, 0.85, 0.52)
                    contrast_h = (best_h + 0.5) % 1.0
                    r2, g2, b2 = colorsys.hsv_to_rgb(contrast_h, 0.75, 0.90)
                    return (
                        (int(r1 * 255), int(g1 * 255), int(b1 * 255)),
                        (int(r2 * 255), int(g2 * 255), int(b2 * 255)),
                    )
            except Exception as e:
                logger.debug("Failed extracting avatar colors: %s", e)

        idx = int(hashlib.md5(author_name.encode("utf-8")).hexdigest(), 16) % len(NOBLE_PAIRS)
        return NOBLE_PAIRS[idx]

    def _generate_3d_wax_seal(
        self,
        avatar_bytes: bytes | None,
        author_name: str,
        base_color: tuple[int, int, int],
        powder_color: tuple[int, int, int],
        size: int = 210,
        seed: int = 42,
    ) -> Image.Image:
        """Generates authentic 3D molten wax seal with organic rims, debossed valleys, and contrasting powder."""
        np.random.seed(seed)
        random.seed(seed)

        center = size / 2.0
        y_idx, x_idx = np.indices((size, size))
        dx = (x_idx - center).astype(np.float32)
        dy = (y_idx - center).astype(np.float32)
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)

        # Organic molten wax blob radius with harmonic waves
        base_r = size * 0.42
        blob_r = base_r * (
            1.0
            + 0.06 * np.cos(3 * angle + random.uniform(0, math.pi))
            + 0.05 * np.cos(5 * angle + random.uniform(0, math.pi))
            + 0.03 * np.cos(7 * angle + random.uniform(0, math.pi))
            + 0.04 * np.sin(2 * angle + random.uniform(0, math.pi))
        )

        rim_mask = np.clip((blob_r - dist) / 5.0, 0.0, 1.0)
        ridge_dist = base_r * 0.78
        ridge_h = np.exp(-((dist - ridge_dist) ** 2) / (2 * (14.0**2))) * 0.45
        basin_r = base_r * 0.70

        # Debossed emblem / monogram
        emblem_mask = np.zeros((size, size), dtype=np.float32)
        if avatar_bytes:
            try:
                import io

                av = (
                    Image.open(io.BytesIO(avatar_bytes))
                    .convert("L")
                    .resize((int(basin_r * 1.8), int(basin_r * 1.8)), Image.Resampling.LANCZOS)
                )
                av_arr = np.array(av, dtype=np.float32) / 255.0
                av_arr = (av_arr - av_arr.min()) / (av_arr.max() - av_arr.min() + 1e-6)
                pw, ph = av.size
                px = int(center - pw / 2)
                py = int(center - ph / 2)
                av_high = av_arr - np.array(av.filter(ImageFilter.GaussianBlur(3))) / 255.0
                av_emb = np.clip(av_high * 2.5 + 0.5, 0.0, 1.0)
                sub = np.zeros((size, size), dtype=np.float32)
                sub[py : py + ph, px : px + pw] = av_emb
                emblem_mask = sub * (dist < (basin_r * 0.92))
            except Exception:
                avatar_bytes = None

        if not avatar_bytes:
            mono_img = Image.new("L", (size, size), 255)
            d_m = ImageDraw.Draw(mono_img)
            initial = (author_name.strip()[:1] or "К").upper()
            f_size = int(basin_r * 1.1)
            font = ImageFont.truetype(self.font_cormorant_path, f_size)
            d_m.text((int(center), int(center) - 4), initial, font=font, fill=0, anchor="mm")
            d_m.ellipse(
                (
                    center - basin_r * 0.92,
                    center - basin_r * 0.92,
                    center + basin_r * 0.92,
                    center + basin_r * 0.92,
                ),
                outline=0,
                width=4,
            )
            d_m.ellipse(
                (
                    center - basin_r * 0.82,
                    center - basin_r * 0.82,
                    center + basin_r * 0.82,
                    center + basin_r * 0.82,
                ),
                outline=0,
                width=2,
            )
            mono_arr = np.array(mono_img, dtype=np.float32) / 255.0
            emblem_mask = (1.0 - mono_arr) * (dist < (basin_r * 0.95))

        H_seal = (rim_mask * 0.65 + ridge_h) - emblem_mask * 0.28
        H_seal = np.clip(H_seal, 0.0, 1.5)

        # 3D Normal & Lighting
        gy, gx = np.gradient(H_seal * 16.0)
        norm_len = np.sqrt(gx**2 + gy**2 + 1.0)
        nx, ny, nz = -gx / norm_len, -gy / norm_len, 1.0 / norm_len

        lx, ly, lz = -0.55, -0.55, 0.63
        l_len = math.sqrt(lx**2 + ly**2 + lz**2)
        lx, ly, lz = lx / l_len, ly / l_len, lz / l_len
        diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)

        hx, hy, hz = lx, ly, lz + 1.0
        h_len = math.sqrt(hx**2 + hy**2 + hz**2)
        specular = (
            np.clip(nx * (hx / h_len) + ny * (hy / h_len) + nz * (hz / h_len), 0.0, 1.0) ** 14.0
        )
        ao = np.clip(1.0 - (gx**2 + gy**2) * 0.08, 0.45, 1.0)

        powder_noise = np.random.uniform(0.85, 1.15, (size, size))
        powder_weight = np.clip((1.0 - emblem_mask * 1.8), 0.0, 1.0) * (dist < (basin_r * 0.88))

        rgb = np.zeros((size, size, 3), dtype=np.float32)
        base_col = np.array(base_color, dtype=np.float32)
        powd_col = np.array(powder_color, dtype=np.float32)

        for c in range(3):
            wax_col = base_col[c] * (diffuse * 0.70 + 0.35) * ao * (1.0 - emblem_mask * 0.35)
            p_col = powd_col[c] * (diffuse * 0.85 + 0.25) * powder_noise
            blended = wax_col * (1.0 - powder_weight) + p_col * powder_weight + specular * 140.0
            rgb[:, :, c] = np.clip(blended, 0.0, 255.0)

        seal_rgba = np.zeros((size, size, 4), dtype=np.uint8)
        seal_rgba[:, :, :3] = rgb.astype(np.uint8)
        seal_rgba[:, :, 3] = np.clip(rim_mask * 255.0, 0, 255).astype(np.uint8)
        seal_img = Image.fromarray(seal_rgba, mode="RGBA")

        # Soft cast shadow
        shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        sh_arr = (rim_mask * 110.0).astype(np.uint8)
        shadow.paste(
            Image.new("RGBA", (size, size), (15, 10, 8, 255)),
            (0, 0),
            Image.fromarray(sh_arr, mode="L"),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(8))

        final_stamp = Image.new("RGBA", (size + 20, size + 20), (0, 0, 0, 0))
        final_stamp.paste(shadow, (14, 14), shadow)
        final_stamp.paste(seal_img, (6, 6), seal_img)
        return final_stamp

    def _fit_text_to_bounds(
        self,
        text: str,
        font_path: str,
        dropcap_char: str,
        max_w: int,
        max_h: int,
    ) -> tuple[
        list[str], ImageFont.FreeTypeFont, int, int, ImageFont.FreeTypeFont, tuple[int, ...], int
    ]:
        """Dynamically scales font size to fit 100% of the quote within template bounds without truncation."""
        d_meas = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
        raw_words = text.replace("\n", " ").split()

        for sz in [74, 66, 58, 52, 46, 42, 38]:
            font = ImageFont.truetype(font_path, sz)
            line_h = int(sz * 1.30)
            dc_sz = int(sz * 2.3)
            f_dc = ImageFont.truetype(self.font_dropcap_path, dc_sz)
            dc_box = d_meas.textbbox((0, 0), dropcap_char, font=f_dc)
            indent_w = (dc_box[2] - dc_box[0]) + 30

            lines: list[str] = []
            curr: list[str] = []
            max_indented = max_w - indent_w

            for w in raw_words:
                line_idx = len(lines)
                cur_max = max_indented if line_idx < 2 else max_w
                test_line = " ".join(curr + [w])
                length = d_meas.textlength(test_line, font=font) * 0.84
                if length <= cur_max:
                    curr.append(w)
                else:
                    if curr:
                        lines.append(" ".join(curr))
                        curr = [w]
                    else:
                        lines.append(w)
            if curr:
                lines.append(" ".join(curr))

            total_h = len(lines) * line_h + 100
            if total_h <= max_h:
                return lines, font, sz, line_h, f_dc, dc_box, indent_w

        return lines, font, sz, line_h, f_dc, dc_box, indent_w

    def _render_rotated_glyph(
        self,
        ch: str,
        font: ImageFont.FreeTypeFont,
        angle_deg: float,
        pressure: float = 1.0,
        base_ink_rgb: np.ndarray | None = None,
    ) -> tuple[Image.Image, float, int, int]:
        """Renders single glyph with natural cursive line weight and baseline rotation."""
        if base_ink_rgb is None:
            # Rich deep black-sepia iron gall ink
            base_ink_rgb = np.array([12, 10, 8], dtype=np.float32)

        pad = 28
        pad_base = 80
        cw = 110 + pad * 2
        ch_h = 110 + pad * 2
        temp = Image.new("RGBA", (cw, ch_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(temp)

        # Natural hairline cursive without heavy artificial stroke
        d.text((pad, pad_base), ch, font=font, fill=(0, 0, 0, 255), anchor="ls")

        arr = np.array(temp)
        alpha = arr[:, :, 3].astype(np.float32) / 255.0
        ink_rendered = np.zeros((ch_h, cw, 4), dtype=np.uint8)

        for c in range(3):
            col = base_ink_rgb[c] * (1.15 - 0.25 * alpha * pressure)
            ink_rendered[:, :, c] = np.clip(col, 0, 255).astype(np.uint8)

        ink_rendered[:, :, 3] = np.clip(
            arr[:, :, 3] * np.clip(0.92 + pressure * 0.06, 0.86, 1.0),
            0,
            255,
        ).astype(np.uint8)

        pil_char = Image.fromarray(ink_rendered, mode="RGBA")
        bbox = d.textbbox((pad, pad_base), ch, font=font, anchor="ls")
        adv = (bbox[2] - bbox[0]) * 0.84

        # Rotate around exact baseline anchor point
        rotated = pil_char.rotate(
            angle_deg, resample=Image.Resampling.BILINEAR, center=(pad, pad_base)
        )
        return rotated, adv, pad, pad_base

    def render_quote(
        self,
        text: str,
        author_name: str,
        avatar_bytes: bytes | None = None,
        template_name: str | None = None,
    ) -> bytes:
        """Synchronously renders an authentic parchment quote card returning JPEG bytes."""
        clean_text = text.strip()
        if not clean_text:
            clean_text = "..."

        # If template not explicitly specified, pick randomly for variety
        if not template_name or template_name not in self.TEMPLATES:
            template_name = random.choice(self.TEMPLATES)

        cfg = TEMPLATE_CONFIGS.get(template_name, TEMPLATE_CONFIGS["Dagger"])

        tpl_data = self._get_template_data(template_name)
        bg = tpl_data["bg"].copy()
        cr_arr = tpl_data["crease_arr"]
        crease_intensity = tpl_data["crease_intensity"]
        paper_fiber = tpl_data["paper_fiber"]

        # Base seed for natural calligraphy variation
        base_seed = random.randint(1000, 999999)

        p_comps, tex_erosion, crease_abrasion = self._build_noise_and_dispersion_fields(
            base_seed=base_seed,
            crease_intensity=crease_intensity,
            paper_fiber=paper_fiber,
        )

        # Extract primary base wax color and contrasting powder highlight
        clean_author = author_name.strip() or "Анонім"
        base_wax_col, powder_col = self.extract_noble_colors(avatar_bytes, clean_author)

        # Extract dropcap letter (first valid alphanumeric character)
        dropcap_char = ""
        first_letter_idx = 0
        for idx, ch in enumerate(clean_text):
            if ch.isalnum():
                dropcap_char = ch.upper()
                first_letter_idx = idx
                break

        if not dropcap_char:
            dropcap_char = "А"
            body_text = clean_text
        else:
            body_text = clean_text[:first_letter_idx] + clean_text[first_letter_idx + 1 :]
            body_text = body_text.lstrip(" ,.-—")

        # Dynamic text auto-scaling to fit 100% of the quote without truncation
        lines, font_marck, font_sz, line_h, font_dropcap, dc_bbox, indent_w = (
            self._fit_text_to_bounds(
                body_text,
                self.font_marck_path,
                dropcap_char,
                max_w=cfg["max_w"],
                max_h=cfg["max_h"],
            )
        )

        layer = Image.new("RGBA", (self.canvas_w, self.canvas_h), (0, 0, 0, 0))
        d_layer = ImageDraw.Draw(layer)

        start_x = cfg["start_x"]
        start_y = cfg["start_y"]

        # Draw illuminated dropcap: Sapphire Blue base wax tone, aligned with Line 0 top
        dc_rgba = (base_wax_col[0], base_wax_col[1], base_wax_col[2], 255)
        dc_x = start_x
        dc_y = start_y - int(font_sz * 0.65) - dc_bbox[1]
        d_layer.text((dc_x, dc_y), dropcap_char, font=font_dropcap, fill=dc_rgba)

        # Base slope (3.2 degrees)
        base_slope_deg = 3.2
        base_tan = math.tan(math.radians(base_slope_deg))

        # Render cursive lines
        for line_idx, line in enumerate(lines):
            line_base_y = start_y + line_idx * line_h
            origin_x = start_x + indent_w if line_idx < 2 else start_x + 10
            curr_x = float(origin_x)
            words = line.split(" ")

            for word in words:
                for ch in word:
                    px = int(np.clip(curr_x, 0, self.canvas_w - 15))
                    py = int(np.clip(line_base_y, 0, self.canvas_h - 15))

                    dx = curr_x - origin_x
                    curve_wave = p_comps["traj"][py, px] * 7.5
                    y_baseline = line_base_y - (dx * base_tan) + curve_wave

                    # Numerical tangent angle
                    dx_step = 8.0
                    px_next = min(self.canvas_w - 1, int(curr_x + dx_step))
                    curve_wave_next = p_comps["traj"][py, px_next] * 7.5
                    dy = (-((dx + dx_step) * base_tan) + curve_wave_next) - (
                        -(dx * base_tan) + curve_wave
                    )
                    local_tangent_deg = np.clip(math.degrees(math.atan(dy / dx_step)), -4.5, 4.5)

                    total_angle = -local_tangent_deg + p_comps["slant"][py, px] * 1.5
                    sz = int(font_sz * (1.0 + p_comps["scale"][py, px] * 0.035))

                    press_val = p_comps["press"][py, px]
                    pressure = 1.02 + press_val * 0.12

                    if press_val < -0.30:
                        fpath = self.font_bad_path
                    elif press_val > 0.45:
                        fpath = self.font_caveat_path
                    else:
                        fpath = self.font_marck_path

                    font_ch = ImageFont.truetype(fpath, sz)
                    glyph_img, adv, pad, pad_base = self._render_rotated_glyph(
                        ch, font_ch, total_angle, pressure
                    )

                    layer.paste(
                        glyph_img, (int(curr_x - pad), int(y_baseline - pad_base)), glyph_img
                    )
                    curr_x += adv * math.cos(math.radians(local_tangent_deg))

                curr_x += 24.0

        # Author signature
        author_text = f"— {clean_author}"
        author_base_y = start_y + len(lines) * line_h + 65
        author_origin_x = start_x + 180
        curr_x = float(author_origin_x)

        font_author = ImageFont.truetype(self.font_marck_path, int(font_sz * 0.90))
        for ch in author_text:
            px = int(np.clip(curr_x, 0, self.canvas_w - 15))
            py = int(np.clip(author_base_y, 0, self.canvas_h - 15))
            dx = curr_x - author_origin_x
            y_baseline = author_base_y - (dx * base_tan) + p_comps["traj"][py, px] * 5.0
            angle = -(base_slope_deg + p_comps["slant"][py, px] * 1.2)
            glyph_img, adv, pad, pad_base = self._render_rotated_glyph(ch, font_author, angle, 1.04)
            layer.paste(glyph_img, (int(curr_x - pad), int(y_baseline - pad_base)), glyph_img)
            curr_x += adv * 0.90

        # 3D Molten Wax Seal with Gold Powder Dusting beside author signature
        stamp_size = 200
        stamp_img = self._generate_3d_wax_seal(
            avatar_bytes,
            clean_author,
            base_wax_col,
            powder_col,
            size=stamp_size,
            seed=base_seed,
        )
        stamp_x = int(min(curr_x + 35, self.canvas_w - stamp_size - 180))
        stamp_y = int(author_base_y - 70)
        layer.paste(stamp_img, (stamp_x, stamp_y), stamp_img)

        # AUTHENTIC TEXTURE & CREASE INTERACTION
        arr = np.array(layer, dtype=np.float32)
        alpha = arr[:, :, 3] / 255.0

        # Natural paper transparency: fibers show through moderately (70% .. 95% opacity)
        alpha_creased = alpha * (1.0 - crease_abrasion * 0.30)
        alpha_textured = alpha_creased * np.clip(1.0 + tex_erosion * 0.28, 0.70, 1.15)
        edge = np.clip(alpha * 2.0, 0.0, 1.0) * np.clip((1.0 - alpha) * 2.0, 0.0, 1.0) * 4.0
        final_alpha = np.clip(alpha_textured + tex_erosion * 0.25 * edge, 0.0, 1.0)

        # Color: deep black ink with paper crease absorption
        for c in range(3):
            arr[:, :, c] = np.clip(arr[:, :, c] * (0.85 + 0.15 * final_alpha), 0, 255)
            arr[:, :, c] = np.clip(arr[:, :, c] * (cr_arr * 0.25 + 0.75), 0, 255)

        arr[:, :, 3] = final_alpha * 255.0
        layer = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

        # Composite with parchment background
        final_img = Image.alpha_composite(bg, layer)

        # Save to JPEG bytes
        import io

        buf = io.BytesIO()
        final_img.convert("RGB").save(buf, format="JPEG", quality=93, optimize=True)
        return buf.getvalue()

    async def async_render_quote(
        self,
        text: str,
        author_name: str,
        avatar_bytes: bytes | None = None,
        template_name: str | None = None,
    ) -> bytes:
        """Asynchronously executes quote rendering in threadpool without blocking event loop."""
        return await asyncio.to_thread(
            self.render_quote,
            text,
            author_name,
            avatar_bytes,
            template_name,
        )
