from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# Fallback noble medieval pigment palette (RGB)
MEDIEVAL_PALETTE = [
    (28, 70, 140),  # Royal Lapis Lazuli Blue
    (142, 32, 38),  # Venetian Crimson Red
    (26, 92, 58),  # Forest Emerald Green
    (130, 78, 22),  # Burnished Amber Gold
    (88, 40, 108),  # Imperial Purple
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
            template_name = "Dagger"

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
        # 4 handwriting factors x 4 scales (16 maps)
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
    def extract_dominant_color(avatar_bytes: bytes | None, seed_str: str) -> tuple[int, int, int]:
        """Extracts dominant noble color from avatar, or picks deterministic medieval palette fallback."""
        if not avatar_bytes:
            idx = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16) % len(MEDIEVAL_PALETTE)
            return MEDIEVAL_PALETTE[idx]

        try:
            import io

            img = Image.open(io.BytesIO(avatar_bytes)).convert("RGB")
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32)
            # Find pixels with good saturation: max - min > 40
            rgb_max = arr.max(axis=-1)
            rgb_min = arr.min(axis=-1)
            sat = rgb_max - rgb_min
            valid_mask = (sat > 35) & (rgb_max > 50) & (rgb_min < 220)

            if np.any(valid_mask):
                valid_pixels = arr[valid_mask]
                mean_col = valid_pixels.mean(axis=0)
                # Boost saturation slightly
                mean_col = np.clip(mean_col * 1.1, 0, 255).astype(int)
                return (int(mean_col[0]), int(mean_col[1]), int(mean_col[2]))
        except Exception as e:
            logger.debug("Failed extracting avatar color: %s", e)

        idx = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16) % len(MEDIEVAL_PALETTE)
        return MEDIEVAL_PALETTE[idx]

    def _render_avatar_signet(
        self,
        avatar_bytes: bytes | None,
        author_name: str,
        dom_color: tuple[int, int, int],
        size: int = 185,
    ) -> Image.Image:
        """Generates an authentic 3D wax/ink signet stamp with user avatar or calligraphic monogram."""
        stamp_rgba = np.zeros((size, size, 4), dtype=np.uint8)
        dom_rgb = np.array(dom_color, dtype=np.float32)

        if avatar_bytes:
            try:
                import io

                av_img = Image.open(io.BytesIO(avatar_bytes)).convert("RGB")
                av_res = av_img.resize((size, size), Image.Resampling.LANCZOS)
                av_arr = np.clip(
                    (np.array(av_res.convert("L"), dtype=np.float32) - 60) * 1.8, 0, 255
                )
                intensity = (255.0 - av_arr) / 255.0
            except Exception:
                intensity = np.zeros((size, size), dtype=np.float32)
        else:
            # Monogram fallback
            mono_img = Image.new("L", (size, size), 255)
            d_mono = ImageDraw.Draw(mono_img)
            f_mono = ImageFont.truetype(self.font_cormorant_path, int(size * 0.55))
            initial = (author_name.strip()[:1] or "К").upper()
            d_mono.text((size // 2, size // 2), initial, font=f_mono, fill=40, anchor="mm")
            intensity = (255.0 - np.array(mono_img, dtype=np.float32)) / 255.0

        for c in range(3):
            stamp_rgba[:, :, c] = np.clip(dom_rgb[c] * (1.15 - 0.40 * intensity), 0, 255).astype(
                np.uint8
            )

        # Circular seal mask with double medieval border
        stamp_mask = Image.new("L", (size, size), 0)
        d_m = ImageDraw.Draw(stamp_mask)
        d_m.ellipse((4, 4, size - 4, size - 4), fill=255)
        d_m.ellipse((10, 10, size - 10, size - 10), outline=0, width=3)
        d_m.ellipse((3, 3, size - 3, size - 3), outline=255, width=4)

        mask_arr = np.array(stamp_mask, dtype=np.float32) / 255.0
        stamp_rgba[:, :, 3] = np.clip(
            intensity * mask_arr * 230 + (mask_arr > 0.8) * 210,
            0,
            255,
        ).astype(np.uint8)

        return Image.fromarray(stamp_rgba, mode="RGBA")

    def _wrap_text_for_parchment(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_full_w: int = 980,
        max_indented_w: int = 740,
        max_lines: int = 8,
    ) -> list[str]:
        """Wraps text taking into account the illuminated drop-cap indent for the first 2 lines."""
        raw_words = text.replace("\n", " ").split()
        if not raw_words:
            return ["..."]

        lines: list[str] = []
        curr_words: list[str] = []

        d = ImageDraw.Draw(Image.new("RGBA", (10, 10)))

        for word in raw_words:
            line_idx = len(lines)
            max_w = max_indented_w if line_idx < 2 else max_full_w
            test_line = " ".join(curr_words + [word])
            w = d.textlength(test_line, font=font) * 0.84

            if w <= max_w:
                curr_words.append(word)
            else:
                if curr_words:
                    lines.append(" ".join(curr_words))
                    curr_words = [word]
                    if len(lines) >= max_lines:
                        break
                else:
                    lines.append(word)
                    if len(lines) >= max_lines:
                        break

        if curr_words and len(lines) < max_lines:
            lines.append(" ".join(curr_words))

        if len(lines) >= max_lines and len(raw_words) > sum(len(ln.split()) for ln in lines):
            lines[-1] = lines[-1].rstrip(".,!?;: ") + "…"

        return lines

    def _render_rotated_glyph(
        self,
        ch: str,
        font: ImageFont.FreeTypeFont,
        angle_deg: float,
        pressure: float = 1.0,
        base_ink_rgb: np.ndarray | None = None,
    ) -> tuple[Image.Image, float, int, int]:
        """Renders single glyph with liquid ink physics and rotation anchored on the font baseline."""
        if base_ink_rgb is None:
            base_ink_rgb = np.array([28, 20, 16], dtype=np.float32)

        pad = 28
        pad_base = 80
        cw = 110 + pad * 2
        ch_h = 110 + pad * 2
        temp = Image.new("RGBA", (cw, ch_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(temp)
        d.text((pad, pad_base), ch, font=font, fill=(0, 0, 0, 255), anchor="ls")

        arr = np.array(temp)
        alpha = arr[:, :, 3].astype(np.float32) / 255.0
        ink_rendered = np.zeros((ch_h, cw, 4), dtype=np.uint8)

        for c in range(3):
            col = base_ink_rgb[c] * (1.20 - 0.35 * alpha * pressure)
            ink_rendered[:, :, c] = np.clip(col, 0, 255).astype(np.uint8)

        ink_rendered[:, :, 3] = np.clip(
            arr[:, :, 3] * np.clip(0.92 + pressure * 0.06, 0.88, 1.0),
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
        """Synchronously renders an authentic parchment quote card returning PNG bytes."""
        clean_text = text.strip()
        if not clean_text:
            clean_text = "..."

        if not template_name or template_name not in self.TEMPLATES:
            # Deterministically choose template from text hash
            h_val = int(hashlib.md5(clean_text.encode("utf-8")).hexdigest(), 16)
            template_name = self.TEMPLATES[h_val % len(self.TEMPLATES)]

        tpl_data = self._get_template_data(template_name)
        bg = tpl_data["bg"].copy()
        cr_arr = tpl_data["crease_arr"]
        crease_intensity = tpl_data["crease_intensity"]
        paper_fiber = tpl_data["paper_fiber"]

        # Base seed from quote content for reproducible handwriting trajectory
        base_seed = int(hashlib.md5((clean_text + author_name).encode("utf-8")).hexdigest()[:8], 16)

        p_comps, tex_erosion, crease_abrasion = self._build_noise_and_dispersion_fields(
            base_seed=base_seed,
            crease_intensity=crease_intensity,
            paper_fiber=paper_fiber,
        )

        # Dominant color
        dom_color = self.extract_dominant_color(avatar_bytes, clean_text)

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

        # Fonts
        font_dropcap = ImageFont.truetype(self.font_dropcap_path, 230)
        base_font_sz = 74 if len(body_text) < 220 else 64
        line_h = 96 if base_font_sz == 74 else 84
        font_marck = ImageFont.truetype(self.font_marck_path, base_font_sz)

        # Wrap text
        lines = self._wrap_text_for_parchment(
            body_text,
            font_marck,
            max_full_w=980,
            max_indented_w=740,
            max_lines=7,
        )

        layer = Image.new("RGBA", (self.canvas_w, self.canvas_h), (0, 0, 0, 0))
        d_layer = ImageDraw.Draw(layer)

        start_x = 420
        start_y = 660

        # Draw illuminated dropcap
        dc_rgba = (dom_color[0], dom_color[1], dom_color[2], 255)
        d_layer.text((start_x, start_y - 20), dropcap_char, font=font_dropcap, fill=dc_rgba)

        # Base slope (3.2 degrees)
        base_slope_deg = 3.2
        base_tan = math.tan(math.radians(base_slope_deg))

        # Render cursive lines
        for line_idx, line in enumerate(lines):
            line_base_y = start_y + line_idx * line_h
            origin_x = start_x + 235 if line_idx < 2 else start_x + 14
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
                    sz = int(base_font_sz * (1.0 + p_comps["scale"][py, px] * 0.035))

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

        # Author signature & seal
        clean_author = author_name.strip() or "Анонім"
        author_text = f"— {clean_author}"
        author_base_y = start_y + len(lines) * line_h + 75
        author_origin_x = start_x + 200
        curr_x = float(author_origin_x)

        font_author = ImageFont.truetype(self.font_marck_path, int(base_font_sz * 0.88))
        for ch in author_text:
            px = int(np.clip(curr_x, 0, self.canvas_w - 15))
            py = int(np.clip(author_base_y, 0, self.canvas_h - 15))
            dx = curr_x - author_origin_x
            y_baseline = author_base_y - (dx * base_tan) + p_comps["traj"][py, px] * 5.0
            angle = -(base_slope_deg + p_comps["slant"][py, px] * 1.2)
            glyph_img, adv, pad, pad_base = self._render_rotated_glyph(ch, font_author, angle, 1.04)
            layer.paste(glyph_img, (int(curr_x - pad), int(y_baseline - pad_base)), glyph_img)
            curr_x += adv * 0.90

        # Signet stamp
        stamp_size = 185
        stamp_img = self._render_avatar_signet(
            avatar_bytes, clean_author, dom_color, size=stamp_size
        )
        stamp_x = int(min(curr_x + 35, self.canvas_w - stamp_size - 180))
        stamp_y = int(author_base_y - 65)

        # Soft shadow under stamp
        sh = Image.new("RGBA", stamp_img.size, (0, 0, 0, 0))
        d_sh = ImageDraw.Draw(sh)
        d_sh.ellipse((4, 4, stamp_size - 4, stamp_size - 4), fill=(20, 15, 10, 80))
        sh = sh.filter(ImageFilter.GaussianBlur(6))
        layer.paste(sh, (stamp_x + 4, stamp_y + 4), sh)
        layer.paste(stamp_img, (stamp_x, stamp_y), stamp_img)

        # APPLY 5-CARD NORMAL DISPERSION & INK TEXTURE ABSORPTION
        arr = np.array(layer, dtype=np.float32)
        alpha = arr[:, :, 3] / 255.0

        # 1. Base alpha reduced directly across fold lines (abrasion on creases)
        alpha_creased = alpha * (1.0 - crease_abrasion * 0.70)

        # 2. Texture applied by the 5-card dispersion
        tex_modulation = tex_erosion * 0.85
        alpha_textured = alpha_creased * np.clip(1.0 + tex_modulation, 0.10, 1.30)

        # 3. Micro-crack edge roughness
        edge = np.clip(alpha * 2.0, 0.0, 1.0) * np.clip((1.0 - alpha) * 2.0, 0.0, 1.0) * 4.0
        edge_crack = tex_modulation * 0.50 * edge * (1.0 + crease_intensity)
        final_alpha = np.clip(alpha_textured + edge_crack, 0.0, 1.0)

        # 4. Color fading & fold bleaching
        for c in range(3):
            arr[:, :, c] = np.clip(
                arr[:, :, c] * (0.8 + 0.35 * final_alpha) + crease_abrasion * 35.0, 0, 255
            )
            arr[:, :, c] = np.clip(arr[:, :, c] * (cr_arr * 0.35 + 0.65), 0, 255)

        arr[:, :, 3] = final_alpha * 255.0
        layer = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

        # Composite with parchment background
        final_img = Image.alpha_composite(bg, layer)

        # Save to PNG bytes
        import io

        buf = io.BytesIO()
        final_img.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
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
