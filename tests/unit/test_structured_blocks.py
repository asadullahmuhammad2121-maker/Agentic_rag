"""Unit tests for structured block detection during chunking."""

from __future__ import annotations

from app.services.chunking.structured_blocks import (
    is_list_line,
    split_structured_text,
)


def test_is_list_line_detects_numbered_bullet_and_label_items() -> None:
    assert is_list_line("1. First step")
    assert is_list_line("2) Second step")
    assert is_list_line("- Bullet item")
    assert is_list_line("* Another bullet")
    assert is_list_line("Ingest: Load documents into the pipeline")
    assert not is_list_line("Plain paragraph sentence.")
    assert not is_list_line("")


def test_split_structured_text_keeps_heading_with_numbered_list() -> None:
    text = (
        "Intro paragraph about retrieval augmented generation.\n\n"
        "Core Pipeline\n"
        "1. Ingest source documents\n"
        "2. Store vector embeddings\n"
        "3. Retrieve relevant chunks\n"
        "4. Augment the prompt\n"
        "5. Generate the grounded answer\n\n"
        "Closing paragraph about evaluation."
    )
    blocks = split_structured_text(text)
    joined = "\n".join(block for block, _start, _end, _is_list in blocks)
    assert "Core Pipeline" in joined
    assert "1. Ingest source documents" in joined
    assert "5. Generate the grounded answer" in joined

    pipeline_blocks = [
        block
        for block, _start, _end, is_list in blocks
        if is_list and "Core Pipeline" in block and "1. Ingest" in block
    ]
    assert len(pipeline_blocks) == 1
    pipeline_block = pipeline_blocks[0]
    for stage in ("2. Store", "3. Retrieve", "4. Augment", "5. Generate"):
        assert stage in pipeline_block


def test_split_structured_text_keeps_trailing_heading_on_same_line() -> None:
    text = (
        "RAG systems follow a repeatable pipeline. Core Pipeline\n"
        "1. Ingest\n"
        "2. Store\n"
        "3. Retrieve\n"
        "4. Augment\n"
        "5. Generate"
    )
    blocks = split_structured_text(text)
    pipeline_blocks = [
        block
        for block, _start, _end, _is_list in blocks
        if "Core Pipeline" in block and "1. Ingest" in block
    ]
    assert len(pipeline_blocks) == 1
    assert "5. Generate" in pipeline_blocks[0]


def test_split_structured_text_keeps_label_colon_list() -> None:
    text = (
        "Processing stages\n"
        "Alpha: First operation\n"
        "Beta: Second operation\n"
        "Gamma: Third operation"
    )
    blocks = split_structured_text(text)
    list_blocks = [
        block
        for block, _start, _end, is_list in blocks
        if is_list and "Alpha:" in block and "Gamma:" in block
    ]
    assert len(list_blocks) == 1
    assert "Processing stages" in list_blocks[0]


def test_split_structured_text_keeps_pdf_style_number_label_list() -> None:
    text = (
        "Intro paragraph.\n\n"
        "Core Pipeline\n"
        "1\n"
        "Ingest: Documents are split into chunks and converted into vector embeddings.\n"
        "2\n"
        "Store: Embeddings are saved in a vector database.\n"
        "3\n"
        "Retrieve: A user query is embedded and matched against stored vectors.\n"
        "4\n"
        "Augment: Retrieved chunks are inserted into the prompt as context.\n"
        "5\n"
        "Generate: The language model produces a response grounded in that context.\n\n"
        "Closing paragraph."
    )
    blocks = split_structured_text(text)
    pipeline_blocks = [
        block
        for block, _start, _end, is_list in blocks
        if is_list and "Core Pipeline" in block and "Ingest:" in block
    ]
    assert len(pipeline_blocks) == 1
    pipeline_block = pipeline_blocks[0]
    for stage in ("Store:", "Retrieve:", "Augment:", "Generate:"):
        assert stage in pipeline_block


def test_single_pdf_style_number_label_does_not_form_list_block() -> None:
    text = (
        "Notes section\n"
        "1\n"
        "Alpha: Only one paired item here.\n\n"
        "Normal prose continues afterward."
    )
    blocks = split_structured_text(text)
    list_blocks = [block for block, _start, _end, is_list in blocks if is_list]
    assert list_blocks == []


def test_pdf_style_list_includes_wrapped_description_lines() -> None:
    text = (
        "Process Overview\n"
        "1\n"
        "Alpha: First step with a long description that continues on the next line\n"
        "without a colon.\n"
        "2\n"
        "Beta: Second step completes the list."
    )
    blocks = split_structured_text(text)
    list_blocks = [
        block
        for block, _start, _end, is_list in blocks
        if is_list and "Process Overview" in block
    ]
    assert len(list_blocks) == 1
    assert "without a colon." in list_blocks[0]
    assert "Beta:" in list_blocks[0]
