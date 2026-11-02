"""文档解析器测试：覆盖 TXT/MD/CSV/XLSX/DOCX 构造样本。"""

import io

import pytest

from app.core.exceptions import UnsupportedFileTypeError
from app.rag.document_parser import parse_document


class TestUnsupported:
    def test_unknown_extension(self):
        with pytest.raises(UnsupportedFileTypeError):
            parse_document("malware.exe", b"data")

    def test_no_extension(self):
        with pytest.raises(UnsupportedFileTypeError):
            parse_document("README", b"data")


class TestTextMarkdown:
    def test_utf8_txt(self):
        doc = parse_document("a.txt", "第一段。\n\n第二段。".encode("utf-8"))
        assert doc.file_type == "text"
        assert len(doc.pages) == 2

    def test_gbk_fallback(self):
        content = "中文 GBK 编码内容测试"
        doc = parse_document("a.txt", content.encode("gbk"))
        assert content in doc.pages[0].text

    def test_markdown(self):
        doc = parse_document("a.md", "# 标题\n\n正文段落".encode("utf-8"))
        assert doc.file_type == "markdown"

    def test_empty_file_rejected(self):
        with pytest.raises(ValueError):
            parse_document("a.txt", b"   \n\n  ")


class TestCsv:
    def test_csv_rows(self):
        csv_data = "姓名,部门,金额\n张三,研发部,1500\n李四,市场部,2300\n"
        doc = parse_document("data.csv", csv_data.encode("utf-8"))
        assert doc.file_type == "csv"
        assert len(doc.pages) == 2
        assert "张三" in doc.pages[0].text
        assert "1500" in doc.pages[0].text

    def test_nan_filtered(self):
        csv_data = "A,B\n1,\n,"
        doc = parse_document("t.csv", csv_data.encode("utf-8"))
        # 全 nan 行被过滤
        assert all("nan" not in p.text for p in doc.pages)


class TestExcel:
    def test_xlsx_roundtrip(self):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")

        df = pd.DataFrame({"产品": ["笔记本", "显示器"], "单价": [5999, 1299]})
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        doc = parse_document("products.xlsx", buf.getvalue())

        assert doc.file_type == "excel"
        joined = "".join(p.text for p in doc.pages)
        assert "笔记本" in joined
        assert "5999" in joined


class TestDocx:
    def test_docx_paragraphs(self):
        docx = pytest.importorskip("docx")

        document = docx.Document()
        document.add_paragraph("公司考勤管理制度")
        document.add_paragraph("第一条 员工应按时上下班。")
        document.add_paragraph("第二条 迟到超过30分钟按旷工处理。")
        buf = io.BytesIO()
        document.save(buf)

        parsed = parse_document("考勤制度.docx", buf.getvalue())
        assert parsed.file_type == "docx"
        joined = "".join(p.text for p in parsed.pages)
        assert "考勤管理制度" in joined
        assert "旷工" in joined


class TestPdf:
    def test_pdf_parser_invoked(self):
        # 极简 PDF：不验证内容，仅验证不抛异常类型错误
        pypdf = pytest.importorskip("pypdf")
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        writer.write(buf)

        # 空白 PDF 无文本 -> ValueError（未解析出内容）
        with pytest.raises(ValueError):
            parse_document("blank.pdf", buf.getvalue())
