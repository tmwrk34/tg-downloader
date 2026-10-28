from __future__ import annotations

import io

import pytest
from PIL import Image

from services.quote_generator import MEDIEVAL_PALETTE, QuoteGeneratorService


@pytest.fixture
def quote_service() -> QuoteGeneratorService:
    return QuoteGeneratorService()


def test_quote_service_init(quote_service: QuoteGeneratorService):
    assert quote_service.canvas_w == 1792
    assert quote_service.canvas_h == 2400
    assert len(quote_service.TEMPLATES) == 5


def test_dominant_color_fallback(quote_service: QuoteGeneratorService):
    col = quote_service.extract_dominant_color(None, "Test Author")
    assert isinstance(col, tuple)
    assert len(col) == 3
    assert col in MEDIEVAL_PALETTE


def test_dominant_color_from_avatar(quote_service: QuoteGeneratorService):
    # Create a small saturated blue test avatar
    img = Image.new("RGB", (64, 64), (20, 60, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    avatar_bytes = buf.getvalue()

    col = quote_service.extract_dominant_color(avatar_bytes, "Hero")
    assert col[2] > col[0]  # Blue channel should dominate


def test_text_wrapping(quote_service: QuoteGeneratorService):
    from PIL import ImageFont

    font = ImageFont.truetype(quote_service.font_marck_path, 74)
    text = "Це довгий тестовий рядок для перевірки правильного розбиття слів на кілька рядків у манускрипті."
    lines = quote_service._wrap_text_for_parchment(text, font, max_full_w=980, max_indented_w=740)
    assert len(lines) > 1
    # First line should be wrapped
    assert len(lines[0]) > 0


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
