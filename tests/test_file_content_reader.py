import io
import zipfile

import pytest

from qq_social_agent.tools.file_content_reader import (
    FileContentReader,
    FileContentReaderConfig,
    OptionalDependencyMissing,
)


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _docx_bytes(document_xml: str, *, extra_entries: dict[str, bytes] | None = None) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>"
            ),
        )
        archive.writestr("word/document.xml", document_xml)
        for name, payload in (extra_entries or {}).items():
            archive.writestr(name, payload)
    return output.getvalue()


def test_txt_reader_accepts_utf8_and_gb18030_and_marks_context_untrusted() -> None:
    reader = FileContentReader()

    utf8 = reader.read("群聊记录.txt", "第一行\n第二行".encode(), content_type="text/plain; charset=utf-8")
    gb = reader.read("中文.txt", "中文内容".encode("gb18030"), content_type="text/plain")

    assert utf8.ok and utf8.text == "第一行\n第二行"
    assert gb.ok and gb.text == "中文内容"
    assert "不可信外部数据" in utf8.to_context()


@pytest.mark.parametrize(
    ("filename", "content_type", "payload", "status"),
    [
        ("../secret.txt", "text/plain", b"hello", "invalid_file"),
        ("image.png", "image/png", b"png", "unsupported_type"),
        ("fake.pdf", "text/plain", b"%PDF-1.7", "mime_mismatch"),
        ("fake.txt", "application/pdf", b"hello", "mime_mismatch"),
        ("binary.txt", "text/plain", b"abc\x00def", "mime_mismatch"),
        ("missing.txt", "", b"hello", "mime_mismatch"),
    ],
)
def test_reader_strictly_rejects_unsafe_names_extensions_mime_and_magic(
    filename: str,
    content_type: str,
    payload: bytes,
    status: str,
) -> None:
    result = FileContentReader().read(filename, payload, content_type=content_type)

    assert result.status == status


def test_reader_enforces_file_size_and_text_truncation() -> None:
    reader = FileContentReader(FileContentReaderConfig(max_file_bytes=1_024, max_text_chars=256))

    too_large = reader.read("large.txt", b"x" * 1_025, content_type="text/plain")
    truncated = reader.read("small.txt", b"x" * 500, content_type="text/plain")

    assert too_large.status == "too_large"
    assert truncated.ok and truncated.truncated
    assert len(truncated.text) == 256


def test_pdf_reader_gracefully_reports_missing_optional_dependency() -> None:
    def missing_dependency(payload: bytes, max_pages: int):
        raise OptionalDependencyMissing("请安装 pypdf")

    reader = FileContentReader(pdf_extractor=missing_dependency)
    result = reader.read("资料.pdf", b"%PDF-1.7\nbody", content_type="application/pdf")

    assert result.status == "dependency_missing"
    assert "pypdf" in result.error
    assert reader.status_snapshot()["counters"]["dependency_missing"] == 1


def test_pdf_reader_uses_injected_extractor_and_propagates_truncation() -> None:
    calls = []

    def fake_extractor(payload: bytes, max_pages: int):
        calls.append((payload, max_pages))
        return "PDF 第一页正文", True

    reader = FileContentReader(FileContentReaderConfig(max_pdf_pages=3), pdf_extractor=fake_extractor)
    result = reader.read("资料.pdf", b"%PDF-1.7\nbody", content_type="application/pdf")

    assert result.ok
    assert result.text == "PDF 第一页正文"
    assert result.truncated
    assert calls[0][1] == 3


def test_docx_reader_extracts_paragraphs_without_external_dependencies() -> None:
    document = """
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>第一段</w:t></w:r></w:p>
        <w:p><w:r><w:t>第二段</w:t><w:tab/><w:t>尾巴</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """
    reader = FileContentReader()
    result = reader.read("群资料.docx", _docx_bytes(document), content_type=DOCX_MIME)

    assert result.ok
    assert result.extractor == "docx"
    assert result.text.splitlines() == ["第一段", "第二段 尾巴"]


def test_docx_reader_rejects_zip_traversal_and_compression_bomb() -> None:
    normal_document = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>hello</w:t></w:r></w:p></w:body></w:document>"
    )
    traversal = _docx_bytes(normal_document, extra_entries={"../evil.txt": b"bad"})
    reader = FileContentReader()

    result = reader.read("bad.docx", traversal, content_type=DOCX_MIME)
    assert result.status == "unsafe_archive"
    assert result.error == "unsafe_zip_path"

    bomb_document = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{'A' * 100_000}</w:t></w:r></w:p></w:body></w:document>"
    )
    bomb = _docx_bytes(bomb_document)
    strict = FileContentReader(
        FileContentReaderConfig(
            max_file_bytes=2_000_000,
            max_docx_uncompressed_bytes=200_000,
            max_zip_compression_ratio=5.0,
        )
    )
    result = strict.read("bomb.docx", bomb, content_type=DOCX_MIME)

    assert result.status == "unsafe_archive"
    assert result.error == "zip_compression_ratio_exceeded"


def test_docx_reader_rejects_plain_zip_disguised_as_docx() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("hello.txt", "not docx")

    result = FileContentReader().read("fake.docx", output.getvalue(), content_type=DOCX_MIME)

    assert result.status == "mime_mismatch"
    assert result.error == "not_a_docx_document"


def test_file_reader_config_and_status_snapshot_do_not_include_content() -> None:
    config = FileContentReaderConfig.from_config(
        {
            "max_file_bytes": 500_000,
            "max_text_chars": 2_000,
            "max_pdf_pages": 5,
            "require_declared_mime": False,
        }
    )
    reader = FileContentReader(config)
    result = reader.read("note.txt", b"private file text", content_type="")
    status = reader.status_snapshot()

    assert result.ok
    assert config.max_file_bytes == 500_000
    assert config.max_text_chars == 2_000
    assert config.max_pdf_pages == 5
    assert "private file text" not in str(status)
    assert status["last_request"]["filename"] == "note.txt"
