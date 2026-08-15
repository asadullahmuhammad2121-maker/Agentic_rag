"""Helpers for building minimal PDF fixtures in tests."""

from __future__ import annotations

from io import BytesIO


def build_pdf_bytes(pages: list[str]) -> bytes:
    """
    Build a minimal PDF containing one text page per entry in ``pages``.

    Uses a tiny handcrafted PDF structure so tests do not need external assets.
    """
    if not pages:
        # Valid PDF with zero pages is awkward; return a truncated/corrupt header instead
        # when callers want an empty document — prefer explicit corrupt helpers.
        pages = [""]

    objects: list[bytes] = []
    # 1: Catalog, 2: Pages tree — filled after page objects are known.
    page_object_numbers: list[int] = []

    # Reserve object numbers: 1=catalog, 2=pages, then for each page: page, content, font
    next_obj = 3
    page_specs: list[tuple[int, int, int, str]] = []
    for text in pages:
        page_obj = next_obj
        content_obj = next_obj + 1
        font_obj = next_obj + 2
        next_obj += 3
        page_object_numbers.append(page_obj)
        page_specs.append((page_obj, content_obj, font_obj, text))

    kids = " ".join(f"{n} 0 R" for n in page_object_numbers)
    objects.append(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    objects.append(
        f"2 0 obj<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>endobj\n".encode()
    )

    for page_obj, content_obj, font_obj, text in page_specs:
        safe = _escape_pdf_text(text)
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        objects.append(
            (
                f"{page_obj} 0 obj<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 612 792] "
                f"/Contents {content_obj} 0 R "
                f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
                f">>endobj\n"
            ).encode()
        )
        objects.append(
            (f"{content_obj} 0 obj<< /Length {len(stream)} >>stream\n").encode()
            + stream
            + b"\nendstream\nendobj\n"
        )
        objects.append(
            (
                f"{font_obj} 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
            ).encode()
        )

    body = BytesIO()
    body.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj in objects:
        offsets.append(body.tell())
        body.write(obj)

    xref_pos = body.tell()
    body.write(f"xref\n0 {len(offsets)}\n".encode())
    body.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.write(f"{offset:010d} 00000 n \n".encode())
    body.write(
        (f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n").encode()
    )
    return body.getvalue()


def build_corrupt_pdf_bytes() -> bytes:
    """Return bytes that look like a PDF header but are not a valid PDF."""
    return b"%PDF-1.4\nthis is not a valid pdf structure\n%%EOF\n"


def build_empty_pdf_bytes() -> bytes:
    """Return an empty byte string."""
    return b""


def _escape_pdf_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\r", " ")
        .replace("\n", " ")
    )
