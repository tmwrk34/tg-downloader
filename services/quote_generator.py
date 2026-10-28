from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from services.wax_seal import (
    NOBLE_PAIRS,
    composite_seal_and_shadows,
    extract_noble_colors,
    render_3d_wax_seal,
)

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
        """Load and pre-process background, crease overlay, normal map, and fiber texture."""
        if template_name not in self.TEMPLATES:
            template_name = random.choice(self.TEMPLATES)

        if template_name in self._template_cache:
            return self._template_cache[template_name]

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

        nr_path = os.path.join(self.media_dir, f"{template_name} normal.png")
        if os.path.exists(nr_path):
            nr = (
                Image.open(nr_path)
                .convert("RGB")
                .resize((self.canvas_w, self.canvas_h), Image.Resampling.BICUBIC)
            )
            nr_arr = (np.array(nr, dtype=np.float32) / 127.5) - 1.0
            tilt = np.sqrt(nr_arr[:, :, 0] ** 2 + nr_arr[:, :, 1] ** 2)
            gy1, gx1 = np.gradient(nr_arr[:, :, 0])
            gy2, gx2 = np.gradient(nr_arr[:, :, 1])
            n_grad = np.sqrt(gx1**2 + gy1**2 + gx2**2 + gy2**2)
            n_grad = n_grad / (np.percentile(n_grad, 99) + 1e-6)
        else:
            tilt = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
            n_grad = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)

        crease_darkness = 1.0 - cr_arr
        crease_intensity = np.clip(tilt * 0.45 + n_grad * 0.40 + crease_darkness * 0.50, 0.0, 1.0)

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
        """Generate bicubic smooth gaussian-filtered coherent 2D noise."""
        np.random.seed(seed)
        gw = int(math.ceil(width / scale)) + 3
        gh = int(math.ceil(height / scale)) + 3
        raw = np.random.uniform(-1.0, 1.0, (gh, gw))
        img = Image.fromarray(((raw + 1.0) * 127.5).astype(np.uint8), mode="L")
        img = img.resize((width, height), Image.Resampling.BICUBIC)
        if scale > 6:
            img = img.filter(ImageFilter.GaussianBlur(radius=max(1, scale * 0.15)))
        return (np.array(img, dtype=np.float32) / 127.5) - 1.0

    def _build_noise_and_dispersion_fields(
        self,
        base_seed: int,
        crease_intensity: np.ndarray,
        paper_fiber: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Build 16-card 4-factor motor noise fields + 5-card normal-modulated ink texture dispersion."""
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

        crease_weight = np.clip(
            crease_intensity / (np.percentile(crease_intensity, 98) + 1e-6), 0.0, 1.5
        )
        eff_dispersion = raw_disp_5 * (1.0 + 2.5 * crease_weight)

        tex_erosion = (
            paper_fiber * 0.65 * (eff_dispersion / (eff_dispersion.mean() + 1e-6))
            + raw_mean_5 * 0.35
        )
        crease_abrasion = np.clip(crease_weight * 0.60 * (0.7 + 0.3 * raw_disp_5), 0.0, 0.85)

        return composites, tex_erosion, crease_abrasion

    @staticmethod
    def extract_noble_colors(
        avatar_bytes: bytes | None, author_name: str
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Delegate to vectorized HSV noble color extractor."""
        return extract_noble_colors(avatar_bytes, author_name)

    def _generate_3d_wax_seal(
        self,
        avatar_bytes: bytes | None,
        author_name: str,
        base_color: tuple[int, int, int],
        powder_color: tuple[int, int, int],
        size: int = 350,
        seed: int = 42,
    ) -> Image.Image:
        """Delegate 3D wax seal rendering to `services.wax_seal`."""
        return render_3d_wax_seal(
            avatar_bytes=avatar_bytes,
            author_name=author_name,
            base_color=base_color,
            powder_color=powder_color,
            font_cormorant_path=self.font_cormorant_path,
            size=size,
            seed=seed,
        )

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
        """Dynamically scale font size from 116px down to 36px to fit 100% of quote within template bounds."""
        d_meas = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
        raw_words = text.replace("\n", " ").split()

        for sz in [116, 104, 94, 84, 76, 68, 60, 52, 46, 40, 36]:
            font = ImageFont.truetype(font_path, sz)
            line_h = int(sz * 1.34)
            dc_sz = int(sz * 2.2)
            f_dc = ImageFont.truetype(self.font_dropcap_path, dc_sz)
            dc_box = d_meas.textbbox((0, 0), dropcap_char, font=f_dc)
            indent_w = (dc_box[2] - dc_box[0]) + 35

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
        """Render single glyph with natural cursive line weight and baseline rotation."""
        if base_ink_rgb is None:
            base_ink_rgb = np.array([12, 10, 8], dtype=np.float32)

        font_sz = getattr(font, "size", 100)
        pad = int(font_sz * 0.8)
        pad_base = int(font_sz * 1.5)
        cw = int(font_sz * 2.5) + pad * 2
        ch_h = int(font_sz * 2.5) + pad * 2
        temp = Image.new("RGBA", (cw, ch_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(temp)

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

        rotated = pil_char.rotate(
            angle_deg, resample=Image.Resampling.BILINEAR, center=(pad, pad_base)
        )
        return rotated, adv, pad, pad_base

    @staticmethod
    def _extract_dropcap_and_body(clean_text: str) -> tuple[str, str]:
        """Split quote text into illuminated dropcap letter and remaining body text."""
        for idx, ch in enumerate(clean_text):
            if ch.isalnum():
                body = clean_text[:idx] + clean_text[idx + 1 :]
                return ch.upper(), body.lstrip(" ,.-—")
        return "А", clean_text

    def render_quote(
        self,
        text: str,
        author_name: str,
        avatar_bytes: bytes | None = None,
        template_name: str | None = None,
    ) -> bytes:
        """Synchronously render an authentic parchment quote card returning JPEG bytes."""
        clean_text = text.strip() or "..."

        if not template_name or template_name not in self.TEMPLATES:
            template_name = random.choice(self.TEMPLATES)

        cfg = TEMPLATE_CONFIGS.get(template_name, TEMPLATE_CONFIGS["Dagger"])
        tpl_data = self._get_template_data(template_name)
        bg = tpl_data["bg"].copy()
        cr_arr = tpl_data["crease_arr"]
        crease_intensity = tpl_data["crease_intensity"]
        paper_fiber = tpl_data["paper_fiber"]

        base_seed = random.randint(1000, 999999)
        p_comps, tex_erosion, crease_abrasion = self._build_noise_and_dispersion_fields(
            base_seed=base_seed,
            crease_intensity=crease_intensity,
            paper_fiber=paper_fiber,
        )

        clean_author = author_name.strip() or "Анонім"
        base_wax_col, powder_col = self.extract_noble_colors(avatar_bytes, clean_author)
        dropcap_char, body_text = self._extract_dropcap_and_body(clean_text)

        lines, _font_marck, font_sz, line_h, font_dropcap, dc_bbox, indent_w = (
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

        stamp_size = 350
        total_quote_h = len(lines) * line_h + 160
        vert_offset = int(max(0, cfg["max_h"] - total_quote_h) * 0.35)
        start_x = cfg["start_x"]
        start_y = cfg["start_y"] + vert_offset

        # Draw illuminated dropcap
        dc_rgba = (base_wax_col[0], base_wax_col[1], base_wax_col[2], 255)
        dc_x = start_x
        dc_y = start_y - int(font_sz * 0.65) - dc_bbox[1]
        d_layer.text((dc_x, dc_y), dropcap_char, font=font_dropcap, fill=dc_rgba)

        base_slope_deg = 3.2
        base_tan = math.tan(math.radians(base_slope_deg))

        for line_idx, line in enumerate(lines):
            line_base_y = start_y + line_idx * line_h
            origin_x = start_x + indent_w if line_idx < 2 else start_x + 10
            curr_x = float(origin_x)

            for word in line.split(" "):
                for ch in word:
                    px = int(np.clip(curr_x, 0, self.canvas_w - 15))
                    py = int(np.clip(line_base_y, 0, self.canvas_h - 15))

                    dx = curr_x - origin_x
                    curve_wave = p_comps["traj"][py, px] * 7.5
                    y_baseline = line_base_y - (dx * base_tan) + curve_wave

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

        # Apply paper fiber texture & crease absorption to calligraphy ink layer
        arr = np.array(layer, dtype=np.float32)
        alpha = arr[:, :, 3] / 255.0
        alpha_creased = alpha * (1.0 - crease_abrasion * 0.30)
        alpha_textured = alpha_creased * np.clip(1.0 + tex_erosion * 0.28, 0.70, 1.15)
        edge = np.clip(alpha * 2.0, 0.0, 1.0) * np.clip((1.0 - alpha) * 2.0, 0.0, 1.0) * 4.0
        final_alpha = np.clip(alpha_textured + tex_erosion * 0.25 * edge, 0.0, 1.0)

        for c in range(3):
            arr[:, :, c] = np.clip(arr[:, :, c] * (0.85 + 0.15 * final_alpha), 0, 255)
            arr[:, :, c] = np.clip(arr[:, :, c] * (cr_arr * 0.25 + 0.75), 0, 255)

        arr[:, :, 3] = final_alpha * 255.0
        ink_layer = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

        # Generate 3D Wax Seal and composite with multi-layer drop shadows
        seal_img = self._generate_3d_wax_seal(
            avatar_bytes,
            clean_author,
            base_wax_col,
            powder_col,
            size=stamp_size,
            seed=base_seed,
        )
        stamp_x = int(min(curr_x + 35, self.canvas_w - stamp_size - 130))
        stamp_y = int(author_base_y - int(stamp_size * 0.40))
        if stamp_x < curr_x + 10:
            stamp_x = int(author_origin_x + 60)
            stamp_y = int(author_base_y + 35)

        final_img = composite_seal_and_shadows(
            bg=bg,
            ink_layer=ink_layer,
            seal_img=seal_img,
            stamp_x=stamp_x,
            stamp_y=stamp_y,
        )

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
        """Asynchronously execute quote rendering in threadpool without blocking event loop."""
        return await asyncio.to_thread(
            self.render_quote,
            text,
            author_name,
            avatar_bytes,
            template_name,
        )


__all__ = [
    "NOBLE_PAIRS",
    "TEMPLATE_CONFIGS",
    "QuoteGeneratorService",
]
