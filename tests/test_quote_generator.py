from __future__ import annotations

import io

import pytest
from PIL import Image

from services.quote_generator import NOBLE_PAIRS, QuoteGeneratorService


@pytest.fixture
def quote_service() -> QuoteGeneratorService:
    return QuoteGeneratorService()


def test_quote_service_init(quote_service: QuoteGeneratorService):
    assert quote_service.canvas_w == 1792
    assert quote_service.canvas_h == 2400
    assert len(quote_service.TEMPLATES) == 5


def test_noble_colors_fallback(quote_service: QuoteGeneratorService):
    base_col, powder_col = quote_service.extract_noble_colors(None, "Test Author")
    assert isinstance(base_col, tuple)
    assert len(base_col) == 3
    assert (base_col, powder_col) in NOBLE_PAIRS


def test_noble_colors_from_avatar(quote_service: QuoteGeneratorService):
    # Create a small saturated blue test avatar
    img = Image.new("RGB", (64, 64), (20, 60, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    avatar_bytes = buf.getvalue()

    base_col, powder_col = quote_service.extract_noble_colors(avatar_bytes, "Hero")
    assert base_col[2] > base_col[0]  # Blue channel should dominate
    assert isinstance(powder_col, tuple)
    assert len(powder_col) == 3


def test_text_wrapping(quote_service: QuoteGeneratorService):
    text = "Це довгий тестовий рядок для перевірки правильного розбиття слів на кілька рядків у манускрипті."
    lines, font, font_sz, line_h, f_dc, dc_box, indent_w = quote_service._fit_text_to_bounds(
        text=text,
        font_path=quote_service.font_marck_path,
        dropcap_char="Ц",
        max_w=980,
        max_h=800,
    )
    assert len(lines) > 1
    assert font_sz > 0
    assert indent_w > 0


def test_render_quote_dagger(quote_service: QuoteGeneratorService):
    data = quote_service.render_quote(
        text="Слава хоробрим мандрівникам!",
        author_name="Геральт",
        avatar_bytes=None,
        template_name="Dagger",
    )
    assert isinstance(data, bytes)
    assert len(data) > 50000
    assert data[:2] == b"\xff\xd8"  # JPEG SOI marker


@pytest.mark.parametrize("template", ["Arrow", "Basic", "Ell", "Fired"])
def test_render_quote_all_templates(quote_service: QuoteGeneratorService, template: str):
    data = quote_service.render_quote(
        text="Тут тихо грає лютня.",
        author_name="Любисток",
        avatar_bytes=None,
        template_name=template,
    )
    assert isinstance(data, bytes)
    assert len(data) > 50000
    assert data[:2] == b"\xff\xd8"


@pytest.mark.asyncio
async def test_async_render_quote(quote_service: QuoteGeneratorService):
    data = await quote_service.async_render_quote(
        text="Асинхронний сувій знань.",
        author_name="Мерлін",
    )
    assert isinstance(data, bytes)
    assert len(data) > 50000
    assert data[:2] == b"\xff\xd8"


@pytest.mark.parametrize(
    ("show_seal", "show_dropcap"),
    [
        (False, True),  # /qs
        (True, False),  # /qf
        (False, False),  # /qsf
    ],
)
def test_render_quote_modifiers(
    quote_service: QuoteGeneratorService, show_seal: bool, show_dropcap: bool
):
    data = quote_service.render_quote(
        text="Лист без зайвих оздоблень, чисто суть.",
        author_name="Писар",
        template_name="Dagger",
        show_seal=show_seal,
        show_dropcap=show_dropcap,
    )
    assert isinstance(data, bytes)
    assert len(data) > 30000
    assert data[:2] == b"\xff\xd8"
