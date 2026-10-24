#!/usr/bin/env python3
"""
Hybrid CV/Resume Parser - single-file, local-only implementation.

Required packages:
    pip install pdfplumber PyMuPDF python-docx rapidfuzz

Recommended NLP and web packages:
    pip install spacy fastapi "uvicorn[standard]" python-multipart
    python -m spacy download en_core_web_sm

Optional OCR packages:
    pip install pytesseract pdf2image pillow
    Ubuntu/Debian: sudo apt-get install tesseract-ocr poppler-utils
    macOS:         brew install tesseract poppler

Optional transformer NER:
    pip install transformers torch
    Set CV_PARSER_HF_MODEL to a local model path. The parser never downloads a
    model automatically and never sends resume content to an external service.

USAGE
-----
Function:
    from cv_parser import parse_cv
    result = parse_cv("resume.pdf")

CLI, single file:
    python cv_parser.py resume.pdf --output result.json

CLI, folder (one JSON per PDF/DOCX):
    python cv_parser.py ./resumes --output ./parsed

Web upload:
    python cv_parser.py --serve --host 127.0.0.1 --port 8000

ARCHITECTURE
------------
DocumentLoader validates signatures and extracts PDF/DOCX text and layout
hints. TextCleaner normalizes Unicode and removes repeated headers/footers.
SectionDetector combines heading aliases, fuzzy matching, and layout cues.
RuleBasedExtractor handles deterministic fields and a curated skills lexicon.
NERExtractor optionally adds local spaCy/Hugging Face entities. EntryParser
builds education, employment, project, and certification records. HybridMerger
deduplicates and validates output. CVParser orchestrates the pipeline.

Priority: regex wins for emails, phones, URLs, dates, grades, and credential
IDs; rules determine section boundaries; NER assists with names, organizations,
and locations; conservative validation rejects headings/job titles as names.

KNOWN LIMITATIONS
-----------------
Text extraction from highly graphical, nested, or handwriting-heavy resumes may
lose reading order. OCR quality depends on scan resolution and local language
packs. Rule-based organization/title splitting is necessarily heuristic.
Accuracy can be improved with labeled multilingual resume data, a fine-tuned
layout-aware model, custom spaCy components, and human-reviewed skill/title
taxonomies. Confidence values are heuristic indicators, not probabilities.

EXAMPLE OUTPUT (abridged values, complete schema is always returned)
--------------------------------------------------------------------
{
  "personal_information": {
    "full_name": "Ayesha Khan", "first_name": "Ayesha",
    "last_name": "Khan", "emails": ["ayesha@example.com"],
    "phones": ["+92 300 1234567"],
    "location": {"city": "Islamabad", "state": null,
                 "country": "Pakistan", "raw": "Islamabad, Pakistan"},
    "links": {"linkedin": "https://linkedin.com/in/ayesha",
              "github": null, "portfolio": null, "other": []},
    "date_of_birth": null, "nationality": null, "gender": null,
    "marital_status": null
  },
  "summary": "Data analyst with five years of experience.",
  "skills": {"technical_skills": ["Python", "SQL"],
             "soft_skills": ["Communication"],
             "tools_and_technologies": ["Power BI"],
             "all_skills": ["Python", "SQL", "Communication", "Power BI"]},
  "education": [], "qualifications": [], "experience": [],
  "total_experience_years": 5.0, "projects": [], "certifications": [],
  "languages": [], "awards_and_achievements": [], "publications": [],
  "volunteer_experience": [], "interests": [], "references": [],
  "additional_sections": {},
  "metadata": {"file_name": "resume.pdf", "file_type": "pdf",
               "page_count": 1, "parser_version": "1.0.0",
               "methods_used": ["pdfplumber", "rules", "spacy"],
               "ocr_used": false, "confidence_scores": {},
               "warnings": [], "errors": []}
}
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence
from urllib.parse import urlparse
from xml.etree import ElementTree as ET


VERSION = "1.0.0"
LOGGER = logging.getLogger("cv_parser")


@dataclass
class LineHint:
    """One extracted text line and lightweight layout evidence."""

    text: str
    page: int = 1
    order: int = 0
    bold: bool = False
    font_size: Optional[float] = None
    x0: Optional[float] = None
    y0: Optional[float] = None
    source: str = "body"


@dataclass
class LoadedDocument:
    lines: list[LineHint] = field(default_factory=list)
    file_type: Optional[str] = None
    page_count: Optional[int] = None
    methods: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ocr_used: bool = False

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text.strip())


class DocumentLoader:
    """Validate and extract text from PDF/DOCX without external services."""

    PDF_MAGIC = b"%PDF-"
    DOCX_REQUIRED = {"[Content_Types].xml", "word/document.xml"}

    def validate(self, file_path: Path) -> tuple[Optional[str], Optional[str]]:
        if not file_path.exists() or not file_path.is_file():
            return None, "File does not exist or is not a regular file"
        if file_path.stat().st_size == 0:
            return None, "File is empty"
        suffix = file_path.suffix.lower()
        if suffix not in {".pdf", ".docx"}:
            return None, "Unsupported file type; only PDF and DOCX are accepted"
        try:
            with file_path.open("rb") as handle:
                magic = handle.read(8)
            if suffix == ".pdf":
                if not magic.startswith(self.PDF_MAGIC):
                    return None, "Extension is .pdf but file content is not a PDF"
                return "pdf", None
            if not zipfile.is_zipfile(file_path):
                return None, "Extension is .docx but file content is not a valid DOCX container"
            with zipfile.ZipFile(file_path) as archive:
                names = set(archive.namelist())
                if not self.DOCX_REQUIRED.issubset(names):
                    return None, "DOCX is missing required document components"
            return "docx", None
        except (OSError, zipfile.BadZipFile) as exc:
            return None, f"Could not validate file: {exc}"

    def load(self, file_path: str | Path) -> LoadedDocument:
        path = Path(file_path)
        result = LoadedDocument()
        file_type, error = self.validate(path)
        result.file_type = file_type or path.suffix.lower().lstrip(".") or None
        if error:
            result.errors.append(error)
            return result
        try:
            return self._load_pdf(path) if file_type == "pdf" else self._load_docx(path)
        except PermissionError:
            result.errors.append("Permission denied while reading file")
        except Exception as exc:  # final safety boundary by design
            LOGGER.exception("Document extraction failed")
            message = str(exc).lower()
            if "password" in message or "encrypt" in message:
                result.errors.append("Document is password-protected or encrypted")
            else:
                result.errors.append(f"Document is corrupted or unreadable: {exc}")
        return result

    def _load_pdf(self, path: Path) -> LoadedDocument:
        result = LoadedDocument(file_type="pdf")
        extraction_errors: list[str] = []
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(str(path)) as pdf:
                result.page_count = len(pdf.pages)
                for page_no, page in enumerate(pdf.pages, 1):
                    words = page.extract_words(
                        use_text_flow=False,
                        keep_blank_chars=False,
                        extra_attrs=["fontname", "size"],
                    )
                    if words:
                        result.lines.extend(self._pdf_words_to_lines(words, page_no, page.width))
                    else:
                        text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                        for index, line in enumerate(text.splitlines()):
                            result.lines.append(LineHint(line, page_no, index))
            result.methods.append("pdfplumber")
        except ImportError:
            extraction_errors.append("pdfplumber is not installed")
        except Exception as exc:
            extraction_errors.append(f"pdfplumber failed: {exc}")

        if len(result.text.strip()) < 40:
            try:
                import fitz  # type: ignore

                fitz_lines: list[LineHint] = []
                document = fitz.open(str(path))
                if document.needs_pass:
                    result.errors.append("PDF is password-protected")
                    return result
                result.page_count = document.page_count
                for page_no, page in enumerate(document, 1):
                    blocks = sorted(page.get_text("blocks"), key=lambda b: (round(b[1], 1), b[0]))
                    order = 0
                    for block in blocks:
                        for line in str(block[4]).splitlines():
                            if line.strip():
                                fitz_lines.append(LineHint(line, page_no, order, x0=block[0], y0=block[1]))
                                order += 1
                if len("\n".join(x.text for x in fitz_lines)) > len(result.text):
                    result.lines = fitz_lines
                result.methods.append("PyMuPDF")
            except ImportError:
                extraction_errors.append("PyMuPDF is not installed")
            except Exception as exc:
                extraction_errors.append(f"PyMuPDF failed: {exc}")

        if len(result.text.strip()) < 40:
            self._ocr_pdf(path, result)
        if not result.lines and not result.errors:
            result.errors.append("No extractable text found in PDF")
        result.warnings.extend(extraction_errors)
        return result

    @staticmethod
    def _pdf_words_to_lines(words: list[dict[str, Any]], page: int, width: float) -> list[LineHint]:
        """Reconstruct lines, sorting likely columns from left to right."""
        if not words:
            return []
        mid = width / 2
        left = [w for w in words if float(w.get("x0", 0)) < mid and float(w.get("x1", 0)) < mid + 20]
        right = [w for w in words if float(w.get("x0", 0)) >= mid - 20]
        overlap = len({id(x) for x in left} & {id(x) for x in right})
        column_mode = len(left) > 8 and len(right) > 8 and overlap == 0
        groups = [left, right] if column_mode else [words]
        output: list[LineHint] = []
        order = 0
        for group in groups:
            rows: list[list[dict[str, Any]]] = []
            for word in sorted(group, key=lambda w: (round(float(w.get("top", 0)) / 3) * 3, float(w.get("x0", 0)))):
                top = float(word.get("top", 0))
                target = next((row for row in rows if abs(float(row[0].get("top", 0)) - top) <= 3), None)
                if target is None:
                    target = []
                    rows.append(target)
                target.append(word)
            for row in sorted(rows, key=lambda r: float(r[0].get("top", 0))):
                row.sort(key=lambda w: float(w.get("x0", 0)))
                text = " ".join(str(w.get("text", "")) for w in row).strip()
                if not text:
                    continue
                fonts = [float(w.get("size", 0) or 0) for w in row]
                bold = any("bold" in str(w.get("fontname", "")).lower() for w in row)
                output.append(LineHint(text, page, order, bold, max(fonts) if fonts else None,
                                       float(row[0].get("x0", 0)), float(row[0].get("top", 0))))
                order += 1
        return output

    def _ocr_pdf(self, path: Path, result: LoadedDocument) -> None:
        try:
            import pytesseract  # type: ignore
            from pdf2image import convert_from_path  # type: ignore

            images = convert_from_path(str(path), dpi=250)
            ocr_lines: list[LineHint] = []
            for page_no, image in enumerate(images, 1):
                text = pytesseract.image_to_string(image)
                for order, line in enumerate(text.splitlines()):
                    if line.strip():
                        ocr_lines.append(LineHint(line, page_no, order, source="ocr"))
            if ocr_lines:
                result.lines = ocr_lines
                result.page_count = len(images)
                result.ocr_used = True
                result.methods.append("OCR:pytesseract")
                result.warnings.append("Scanned PDF detected; OCR used")
            else:
                result.errors.append("Scanned PDF detected but OCR produced no text")
        except ImportError:
            result.errors.append(
                "PDF has little or no extractable text; install pytesseract, pdf2image, "
                "Tesseract, and Poppler to enable OCR"
            )
        except Exception as exc:
            result.errors.append(f"Scanned PDF detected but OCR failed: {exc}")

    def _load_docx(self, path: Path) -> LoadedDocument:
        result = LoadedDocument(file_type="docx", page_count=None)
        try:
            from docx import Document  # type: ignore
            from docx.table import Table  # type: ignore
            from docx.text.paragraph import Paragraph  # type: ignore
        except ImportError:
            result.errors.append("python-docx is required to parse DOCX files")
            return result
        try:
            document = Document(str(path))
        except Exception as exc:
            msg = "DOCX is password-protected" if "password" in str(exc).lower() else f"DOCX is corrupted: {exc}"
            result.errors.append(msg)
            return result

        order = 0
        body = document.element.body
        for child in body.iterchildren():
            if child.tag.endswith("}p"):
                paragraph = Paragraph(child, document)
                hint = self._paragraph_hint(paragraph, order)
                if hint.text.strip():
                    result.lines.append(hint)
                    order += 1
            elif child.tag.endswith("}tbl"):
                table = Table(child, document)
                for row in table.rows:
                    cells = [self._cell_text(cell) for cell in row.cells]
                    text = " | ".join(x for x in cells if x)
                    if text.strip():
                        result.lines.append(LineHint(text, order=order, source="table"))
                        order += 1

        for section in document.sections:
            for source_name, part in (("header", section.header), ("footer", section.footer)):
                for paragraph in part.paragraphs:
                    if paragraph.text.strip():
                        hint = self._paragraph_hint(paragraph, order)
                        hint.source = source_name
                        result.lines.append(hint)
                        order += 1

        textbox_seen: set[str] = set()
        namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        for node in document.element.xpath(".//w:txbxContent"):
            texts = [t.text or "" for t in node.findall(".//w:t", namespaces)]
            text = " ".join(x.strip() for x in texts if x.strip()).strip()
            if text and text not in textbox_seen:
                textbox_seen.add(text)
                result.lines.append(LineHint(text, order=order, source="textbox"))
                order += 1
        result.methods.append("python-docx")
        if not result.lines:
            result.errors.append("No text found in DOCX")
        return result

    @staticmethod
    def _cell_text(cell: Any) -> str:
        return " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())

    @staticmethod
    def _paragraph_hint(paragraph: Any, order: int) -> LineHint:
        runs = [r for r in paragraph.runs if r.text.strip()]
        bold = bool(runs) and sum(1 for r in runs if r.bold) >= max(1, math.ceil(len(runs) / 2))
        sizes = [r.font.size.pt for r in runs if r.font.size is not None]
        return LineHint(paragraph.text, order=order, bold=bold, font_size=max(sizes) if sizes else None)


class TextCleaner:
    """Normalize extracted text while preserving line boundaries."""

    LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})
    BULLETS = re.compile(r"^[\s\u2022\u2023\u25E6\u2043\u2219\u25AA\u25CF\u00B7\uf0b7●▪◦]+")
    PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?$", re.I)

    def clean(self, document: LoadedDocument) -> list[LineHint]:
        lines: list[LineHint] = []
        for hint in document.lines:
            text = unicodedata.normalize("NFKC", hint.text.translate(self.LIGATURES))
            text = text.replace("\u200b", "").replace("\xa0", " ")
            text = re.sub(r"[\t\r ]+", " ", text).strip()
            text = self.BULLETS.sub("- ", text)
            if not text or self.PAGE_NUMBER.fullmatch(text):
                continue
            lines.append(LineHint(text, hint.page, hint.order, hint.bold, hint.font_size,
                                  hint.x0, hint.y0, hint.source))

        page_count = max((line.page for line in lines), default=1)
        if page_count > 1:
            normalized = Counter(self._repeat_key(line.text) for line in lines)
            threshold = max(2, math.ceil(page_count * 0.6))
            repeated = {key for key, count in normalized.items() if count >= threshold and len(key) < 100}
            lines = [line for line in lines if self._repeat_key(line.text) not in repeated]

        joined: list[LineHint] = []
        for line in lines:
            if (joined and joined[-1].page == line.page and joined[-1].text.endswith("-")
                    and re.match(r"^[a-z]", line.text)):
                joined[-1].text = joined[-1].text[:-1] + line.text
            else:
                joined.append(line)
        for i, line in enumerate(joined):
            line.order = i
        return joined

    @staticmethod
    def _repeat_key(text: str) -> str:
        return re.sub(r"\d+", "#", text.casefold()).strip()


class SectionDetector:
    """Map heading variants to canonical sections."""

    ALIASES: dict[str, set[str]] = {
        "summary": {"summary", "professional summary", "profile", "professional profile", "objective",
                    "career objective", "about me", "executive summary", "personal statement"},
        "skills": {"skills", "key skills", "core skills", "competencies", "core competencies",
                   "technical skills", "expertise", "areas of expertise", "tools", "technologies"},
        "education": {"education", "academic background", "academic qualifications", "education history",
                      "educational background", "academics"},
        "qualifications": {"qualifications", "professional qualifications", "licenses", "memberships",
                           "licenses and memberships", "professional memberships"},
        "experience": {"experience", "work experience", "professional experience", "employment history",
                       "career history", "work history", "employment", "career experience"},
        "projects": {"projects", "key projects", "personal projects", "academic projects", "selected projects"},
        "certifications": {"certifications", "certificates", "certification", "courses and certifications",
                           "training", "professional development"},
        "languages": {"languages", "language skills", "linguistic skills"},
        "awards_and_achievements": {"awards", "achievements", "honors", "honours", "awards and achievements",
                                    "accomplishments"},
        "publications": {"publications", "research", "papers", "articles"},
        "volunteer_experience": {"volunteer experience", "volunteering", "community service", "voluntary work"},
        "interests": {"interests", "hobbies", "hobbies and interests", "personal interests"},
        "references": {"references", "referees"},
    }
    REVERSE = {alias: key for key, aliases in ALIASES.items() for alias in aliases}

    def __init__(self) -> None:
        try:
            from rapidfuzz import fuzz  # type: ignore
            self._fuzz = fuzz
        except ImportError:
            self._fuzz = None

    @staticmethod
    def _heading_text(text: str) -> str:
        text = re.sub(r"[:|\-–—]+$", "", text.strip())
        return re.sub(r"[^a-z0-9& ]", "", text.casefold()).strip()

    def classify_heading(self, line: LineHint, median_font: float) -> tuple[Optional[str], float, Optional[str]]:
        raw = line.text.strip()
        clean = self._heading_text(raw)
        word_count = len(clean.split())
        if not clean or word_count > 7 or len(clean) > 65:
            return None, 0.0, None
        if clean in self.REVERSE:
            return self.REVERSE[clean], 0.99, raw
        candidate_heading = (
            line.bold or raw.endswith(":") or (raw.isupper() and len(raw) > 2)
            or (line.font_size is not None and line.font_size >= median_font * 1.12)
        )
        best_key, best_score = None, 0.0
        if self._fuzz:
            for alias, key in self.REVERSE.items():
                score = float(self._fuzz.ratio(clean, alias)) / 100
                if score > best_score:
                    best_key, best_score = key, score
        if best_score >= 0.88 and (candidate_heading or best_score >= 0.94):
            return best_key, best_score, raw
        if candidate_heading and 1 <= word_count <= 5 and not self._looks_like_content(raw):
            return "additional", 0.60, raw.rstrip(":")
        return None, 0.0, None

    @staticmethod
    def _looks_like_content(text: str) -> bool:
        return bool(re.search(r"@|https?://|\b\d{4}\b|\+?\d[\d ()-]{6,}", text))

    def split(self, lines: list[LineHint]) -> tuple[dict[str, list[LineHint]], dict[str, list[LineHint]], dict[str, float]]:
        sizes = sorted(x.font_size for x in lines if x.font_size)
        median = sizes[len(sizes) // 2] if sizes else 10.0
        sections: dict[str, list[LineHint]] = defaultdict(list)
        additional: dict[str, list[LineHint]] = defaultdict(list)
        scores: dict[str, float] = {}
        current = "preamble"
        current_extra: Optional[str] = None
        for line in lines:
            category, score, heading = self.classify_heading(line, median)
            if category:
                if category == "additional":
                    current, current_extra = "additional", heading
                    scores[f"additional:{heading}"] = score
                else:
                    current, current_extra = category, None
                    scores[category] = max(scores.get(category, 0), score)
                continue
            if current == "additional" and current_extra:
                additional[current_extra].append(line)
            else:
                sections[current].append(line)
        return dict(sections), dict(additional), scores


class DateParser:
    MONTHS = {name.casefold(): i for i, name in enumerate(calendar.month_name) if name}
    MONTHS.update({name.casefold(): i for i, name in enumerate(calendar.month_abbr) if name})
    MONTHS.update({"sept": 9})
    CURRENT = {"present", "current", "now", "till date", "ongoing", "to date"}
    DATE_TOKEN = r"(?:[A-Za-z]{3,9}[,.]?\s+\d{4}|(?:0?[1-9]|1[0-2])[/-]\d{4}|\d{4}(?:[/-](?:0?[1-9]|1[0-2]))?|(?:Spring|Summer|Fall|Autumn|Winter)\s+\d{4}|Present|Current|Now|Till\s+date|Ongoing|To\s+date)"
    RANGE_RE = re.compile(rf"(?P<start>{DATE_TOKEN})\s*(?:-|–|—|to|until|through)\s*(?P<end>{DATE_TOKEN})", re.I)
    SINGLE_RE = re.compile(DATE_TOKEN, re.I)

    @classmethod
    def normalize(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        clean = re.sub(r"[,\s]+", " ", value.strip()).casefold()
        if clean in cls.CURRENT:
            return "present"
        year_match = re.search(r"\b(19|20)\d{2}\b", clean)
        if not year_match:
            return None
        year = int(year_match.group())
        numeric = re.search(r"\b(0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b", clean)
        if numeric:
            return f"{year:04d}-{int(numeric.group(1)):02d}"
        for month, number in cls.MONTHS.items():
            if re.search(rf"\b{re.escape(month)}\b", clean):
                return f"{year:04d}-{number:02d}"
        season_month = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}
        for season, number in season_month.items():
            if season in clean:
                return f"{year:04d}-{number:02d}"
        return str(year)

    @classmethod
    def extract_range(cls, text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        match = cls.RANGE_RE.search(text)
        if match:
            raw = match.group(0)
            return cls.normalize(match.group("start")), cls.normalize(match.group("end")), raw
        matches = cls.SINGLE_RE.findall(text)
        if matches:
            raw = str(matches[0])
            normalized = cls.normalize(raw)
            return normalized, normalized, raw
        return None, None, None


class RuleBasedExtractor:
    """Deterministic extraction and curated skill matching."""

    EMAIL_RE = re.compile(r"(?<![\w.+-])([\w.+-]+@[\w-]+(?:\.[\w-]+)+)", re.I)
    URL_RE = re.compile(r"(?:(?:https?://|www\.)[^\s<>{}\[\]]+|(?:linkedin\.com|github\.com)/[^\s<>{}\[\]]+)", re.I)
    PHONE_RE = re.compile(r"(?<!\w)(?:\+|00)?\d{1,3}[\s.-]?(?:\(?\d{2,4}\)?[\s.-]?){1,3}\d{3,5}(?!\w)")
    GPA_RE = re.compile(r"\b(?:GPA|CGPA)\s*[:=-]?\s*([0-9](?:\.[0-9]{1,2})?(?:\s*/\s*[0-9](?:\.0)?)?)|\b(\d{1,3}(?:\.\d+)?)\s*%", re.I)
    CREDENTIAL_RE = re.compile(r"\b(?:credential|certificate|license)\s*(?:id|no\.?|number)?\s*[:#-]\s*([\w.-]+)", re.I)
    LABEL_RE = re.compile(r"^(date of birth|dob|nationality|gender|marital status)\s*[:|-]\s*(.+)$", re.I)

    TECHNICAL = {
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "PHP", "Ruby", "Swift", "Kotlin",
        "Scala", "Rust", "Go", "R", "MATLAB", "SQL", "HTML", "CSS", "Bash", "PowerShell", "Dart",
        "React", "Angular", "Vue.js", "Next.js", "Node.js", "Express.js", "Django", "Flask", "FastAPI",
        "Spring Boot", ".NET", "Laravel", "Ruby on Rails", "Flutter", "React Native", "Tailwind CSS",
        "PostgreSQL", "MySQL", "SQL Server", "Oracle", "MongoDB", "Redis", "SQLite", "Elasticsearch",
        "DynamoDB", "Firebase", "Snowflake", "BigQuery", "Databricks", "AWS", "Azure", "Google Cloud",
        "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitHub Actions", "GitLab CI", "Linux",
        "Git", "REST API", "GraphQL", "Microservices", "Machine Learning", "Deep Learning", "NLP",
        "Computer Vision", "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy", "Spark", "Hadoop",
        "Airflow", "Kafka", "MLOps", "Data Analysis", "Data Engineering", "Data Science", "ETL", "Statistics",
        "Cybersecurity", "Penetration Testing", "Network Security", "DevOps", "CI/CD", "Agile", "Scrum",
        "SEO", "SEM", "Google Analytics", "Content Marketing", "Performance Marketing", "A/B Testing",
        "Financial Analysis", "Forecasting", "Risk Management", "Supply Chain", "CRM", "ERP", "SAP",
        "Dynamics 365", "Salesforce", "Figma", "UI/UX", "Graphic Design", "AutoCAD", "SolidWorks",
    }
    TOOLS = {
        "Microsoft Excel", "Microsoft Word", "PowerPoint", "Power BI", "Tableau", "Looker", "Jira",
        "Confluence", "Trello", "Asana", "Notion", "Slack", "Adobe Photoshop", "Adobe Illustrator",
        "Adobe XD", "Canva", "Postman", "Visual Studio Code", "IntelliJ IDEA", "Eclipse", "Jupyter",
        "Google Workspace", "Microsoft Office", "HubSpot", "Mailchimp", "Google Ads", "Meta Ads",
    }
    SOFT = {
        "Communication", "Leadership", "Teamwork", "Problem Solving", "Critical Thinking", "Creativity",
        "Time Management", "Adaptability", "Collaboration", "Negotiation", "Presentation", "Decision Making",
        "Attention to Detail", "Conflict Resolution", "Project Management", "Stakeholder Management",
        "Customer Service", "Analytical Thinking", "Interpersonal Skills", "Mentoring", "Strategic Planning",
    }
    SKILL_LOOKUP = {s.casefold(): s for s in TECHNICAL | TOOLS | SOFT}
    SHORT_SKILLS = {"c", "r", "go"}

    DEGREE_RE = re.compile(
        r"\b(?:Ph\.?D|Doctorate|M\.?Phil|MBA|MSc|M\.S\.?|Master(?:'s)?|BSc|B\.S\.?|B\.Tech|B\.E\.?|"
        r"Bachelor(?:'s)?|Associate(?:'s)?|Diploma|FSc|FA|ICS|ICom|Intermediate|A[- ]Levels?|O[- ]Levels?|Matric(?:ulation)?)\b",
        re.I,
    )
    JOB_RE = re.compile(
        r"\b(?:intern|trainee|assistant|associate|specialist|analyst|consultant|coordinator|officer|engineer|"
        r"developer|designer|administrator|executive|manager|director|head|lead|supervisor|architect|scientist|"
        r"teacher|lecturer|professor|accountant|recruiter|writer|editor|founder|president|salesperson)\b",
        re.I,
    )
    CERT_PROVIDER_RE = re.compile(r"\b(?:Microsoft|Google|Amazon|AWS|Cisco|Oracle|IBM|Coursera|Udemy|edX|PMI|CompTIA|Meta)\b", re.I)

    def extract_contacts(self, text: str) -> dict[str, Any]:
        emails = self._dedupe(self.EMAIL_RE.findall(text))
        urls = self._dedupe(self._clean_url(x) for x in self.URL_RE.findall(text))
        phones: list[str] = []
        for match in self.PHONE_RE.finditer(text):
            value = match.group().strip(" .,-")
            digits = re.sub(r"\D", "", value)
            if 7 <= len(digits) <= 15 and not re.fullmatch(r"(?:19|20)\d{2}(?:\D.*)?", value):
                phones.append(value)
        phones = self._dedupe(phones)
        links: dict[str, Any] = {"linkedin": None, "github": None, "portfolio": None, "other": []}
        for url in urls:
            host = urlparse(url).netloc.casefold()
            if "linkedin.com" in host and links["linkedin"] is None:
                links["linkedin"] = url
            elif "github.com" in host and links["github"] is None:
                links["github"] = url
            elif links["portfolio"] is None and not any(x in host for x in ("facebook.com", "twitter.com", "x.com", "instagram.com")):
                links["portfolio"] = url
            else:
                links["other"].append(url)
        return {"emails": emails, "phones": phones, "links": links}

    @staticmethod
    def _clean_url(value: str) -> str:
        value = value.rstrip(".,;:)")
        return value if value.startswith(("http://", "https://")) else "https://" + value

    def extract_sensitive_labels(self, lines: Sequence[LineHint]) -> dict[str, Optional[str]]:
        values = {"date_of_birth": None, "nationality": None, "gender": None, "marital_status": None}
        mapping = {"date of birth": "date_of_birth", "dob": "date_of_birth", "nationality": "nationality",
                   "gender": "gender", "marital status": "marital_status"}
        for line in lines:
            match = self.LABEL_RE.match(line.text)
            if match:
                values[mapping[match.group(1).casefold()]] = match.group(2).strip()
        return values

    def extract_skills(self, text: str, in_skill_section: bool = False) -> dict[str, list[str]]:
        lower = text.casefold()
        found: list[str] = []
        for key, display in self.SKILL_LOOKUP.items():
            if key in self.SHORT_SKILLS and not in_skill_section:
                continue
            pattern = rf"(?<![\w+#.]){re.escape(key)}(?![\w+#.])"
            if re.search(pattern, lower):
                found.append(display)
        if in_skill_section:
            for chunk in re.split(r"[,;|•·]|\s{2,}", text):
                value = chunk.strip(" -:.")
                if 1 < len(value) < 40 and len(value.split()) <= 4 and not DateParser.SINGLE_RE.search(value):
                    if value.casefold() in self.SKILL_LOOKUP:
                        found.append(self.SKILL_LOOKUP[value.casefold()])
        found = self._dedupe(found)
        technical = [x for x in found if x in self.TECHNICAL]
        tools = [x for x in found if x in self.TOOLS]
        soft = [x for x in found if x in self.SOFT]
        return {"technical_skills": technical, "soft_skills": soft,
                "tools_and_technologies": tools, "all_skills": found}

    @staticmethod
    def _dedupe(values: Iterable[str]) -> list[str]:
        seen, result = set(), []
        for value in values:
            clean = re.sub(r"\s+", " ", str(value)).strip()
            key = clean.casefold()
            if clean and key not in seen:
                seen.add(key)
                result.append(clean)
        return result


class NERExtractor:
    """Optional local NER with spaCy first and local-only HF support."""

    def __init__(self, warnings: list[str], methods: list[str]) -> None:
        self.spacy_nlp = None
        self.hf_pipeline = None
        try:
            import spacy  # type: ignore

            model = "en_core_web_trf" if spacy.util.is_package("en_core_web_trf") else "en_core_web_sm"
            try:
                self.spacy_nlp = spacy.load(model)
                methods.append(f"spaCy:{model}")
            except OSError:
                warnings.append("spaCy is installed but no English model is available; rules-only NER used")
        except ImportError:
            warnings.append("spaCy is unavailable; rules-only NER used")

        local_model = os.getenv("CV_PARSER_HF_MODEL")
        if local_model:
            try:
                from transformers import pipeline  # type: ignore
                self.hf_pipeline = pipeline("token-classification", model=local_model,
                                            aggregation_strategy="simple", local_files_only=True)
                methods.append("HuggingFaceNER:local")
            except Exception as exc:
                warnings.append(f"Local Hugging Face NER model could not be loaded: {exc}")

    def entities(self, text: str) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {"PERSON": [], "ORG": [], "GPE": [], "LOC": [], "DATE": []}
        if self.spacy_nlp and text.strip():
            doc = self.spacy_nlp(text[:100_000])
            for entity in doc.ents:
                if entity.label_ in output:
                    output[entity.label_].append(entity.text.strip())
        if self.hf_pipeline and text.strip():
            try:
                for entity in self.hf_pipeline(text[:20_000]):
                    label = str(entity.get("entity_group", "")).upper()
                    mapped = {"PER": "PERSON", "PERSON": "PERSON", "ORG": "ORG",
                              "LOC": "LOC", "GPE": "GPE", "DATE": "DATE"}.get(label)
                    if mapped:
                        output[mapped].append(str(entity.get("word", "")).strip())
            except Exception as exc:
                LOGGER.warning("HF inference failed: %s", exc)
        for key, values in output.items():
            output[key] = RuleBasedExtractor._dedupe(values)
        return output


class EntryParser:
    """Parse canonical section bodies into structured records."""

    def __init__(self, rules: RuleBasedExtractor, ner: NERExtractor) -> None:
        self.rules, self.ner = rules, ner

    @staticmethod
    def split_entries(lines: Sequence[LineHint], kind: str) -> list[list[str]]:
        clean = [x.text.strip() for x in lines if x.text.strip()]
        if not clean:
            return []
        entries: list[list[str]] = []
        current: list[str] = []
        for line in clean:
            date_range = bool(DateParser.RANGE_RE.search(line))
            cue = bool(RuleBasedExtractor.DEGREE_RE.search(line) if kind == "education"
                       else RuleBasedExtractor.JOB_RE.search(line) if kind == "experience"
                       else False)
            separator = bool(re.fullmatch(r"[-–—_ ]{3,}", line))
            if separator:
                if current:
                    entries.append(current)
                    current = []
                continue
            if current and ((date_range and any(DateParser.RANGE_RE.search(x) for x in current))
                            or (cue and len(current) >= 3 and not current[-1].startswith("- "))):
                entries.append(current)
                current = []
            current.append(line)
        if current:
            entries.append(current)
        return entries

    def education(self, lines: Sequence[LineHint]) -> list[dict[str, Any]]:
        output = []
        for chunk in self.split_entries(lines, "education"):
            raw = "\n".join(chunk)
            entities = self.ner.entities(raw)
            start, end, date_raw = DateParser.extract_range(raw)
            degree_line = next((x for x in chunk if self.rules.DEGREE_RE.search(x)), None)
            degree, field = self._degree_and_field(degree_line)
            institution = next(iter(entities["ORG"]), None)
            if not institution:
                institution = next((x for x in chunk if re.search(r"\b(university|college|institute|school|academy)\b", x, re.I)), None)
            grade_match = self.rules.GPA_RE.search(raw)
            grade = grade_match.group(0) if grade_match else None
            location = next(iter(entities["GPE"] + entities["LOC"]), None)
            description = "\n".join(x for x in chunk if x not in {degree_line, institution} and not DateParser.RANGE_RE.search(x)) or None
            output.append({"degree": degree, "field_of_study": field, "institution": institution,
                           "location": location, "start_date": start, "end_date": end,
                           "date_raw": date_raw, "grade": grade, "description": description,
                           "raw_text": raw})
        return self._nonempty(output, ("degree", "institution", "raw_text"))

    @staticmethod
    def _degree_and_field(line: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if not line:
            return None, None
        match = RuleBasedExtractor.DEGREE_RE.search(line)
        if not match:
            return None, None
        degree = match.group(0)
        remainder = re.sub(re.escape(degree), "", line, flags=re.I).strip(" ,:-|—–")
        remainder = re.sub(r"^(?:in|of)\s+", "", remainder, flags=re.I)
        return degree, remainder or None

    def experience(self, lines: Sequence[LineHint]) -> list[dict[str, Any]]:
        output = []
        for chunk in self.split_entries(lines, "experience"):
            raw = "\n".join(chunk)
            entities = self.ner.entities(raw)
            start, end, date_raw = DateParser.extract_range(raw)
            title = next((self._without_dates(x) for x in chunk if self.rules.JOB_RE.search(x)), None)
            title = title.strip(" |-—–,") if title else None
            company = next(iter(entities["ORG"]), None)
            if not company:
                company = self._company_heuristic(chunk, title)
            location = next(iter(entities["GPE"] + entities["LOC"]), None)
            employment = self._employment_type(raw)
            responsibilities = [re.sub(r"^-\s*", "", x).strip() for x in chunk if x.startswith("- ")]
            if not responsibilities:
                excluded = {x for x in (title, company) if x}
                responsibilities = [x for x in chunk[1:] if x not in excluded and not DateParser.RANGE_RE.search(x)]
            skills = self.rules.extract_skills(raw)
            current = (end == "present")
            output.append({"job_title": title, "company": company, "location": location,
                           "employment_type": employment, "start_date": start, "end_date": end,
                           "date_raw": date_raw, "is_current": current,
                           "duration_months": self._duration_months(start, end),
                           "responsibilities": RuleBasedExtractor._dedupe(responsibilities),
                           "technologies": skills["technical_skills"] + skills["tools_and_technologies"],
                           "raw_text": raw})
        return self._nonempty(output, ("job_title", "company", "date_raw"))

    def projects(self, lines: Sequence[LineHint]) -> list[dict[str, Any]]:
        output = []
        for chunk in self.split_entries(lines, "projects"):
            raw = "\n".join(chunk)
            start, end, _ = DateParser.extract_range(raw)
            urls = self.rules.extract_contacts(raw)["links"]
            all_urls = [x for x in [urls["github"], urls["portfolio"], *urls["other"]] if x]
            first = re.sub(r"^-\s*", "", chunk[0]).strip() if chunk else None
            role_match = re.search(r"\brole\s*[:|-]\s*(.+)", raw, re.I)
            skills = self.rules.extract_skills(raw)
            description = "\n".join(chunk[1:]) or None
            output.append({"name": first, "description": description,
                           "role": role_match.group(1).strip() if role_match else None,
                           "technologies": skills["technical_skills"] + skills["tools_and_technologies"],
                           "start_date": start, "end_date": end,
                           "url": all_urls[0] if all_urls else None, "raw_text": raw})
        return self._nonempty(output, ("name", "description"))

    def certifications(self, lines: Sequence[LineHint]) -> list[dict[str, Any]]:
        output = []
        for chunk in self.split_entries(lines, "certifications"):
            raw = "\n".join(chunk)
            start, end, _ = DateParser.extract_range(raw)
            cred = self.rules.CREDENTIAL_RE.search(raw)
            entities = self.ner.entities(raw)
            issuer = next(iter(entities["ORG"]), None)
            if not issuer:
                provider = self.rules.CERT_PROVIDER_RE.search(raw)
                issuer = provider.group(0) if provider else None
            contacts = self.rules.extract_contacts(raw)["links"]
            urls = [x for x in [contacts["portfolio"], contacts["linkedin"], *contacts["other"]] if x]
            name = re.sub(DateParser.RANGE_RE, "", chunk[0]).strip(" |-—–,") if chunk else None
            output.append({"name": name or None, "issuer": issuer, "issue_date": start,
                           "expiry_date": end if end != start else None,
                           "credential_id": cred.group(1) if cred else None,
                           "url": urls[0] if urls else None, "raw_text": raw})
        return self._nonempty(output, ("name", "issuer", "credential_id"))

    @staticmethod
    def list_section(lines: Sequence[LineHint]) -> list[str]:
        values: list[str] = []
        for line in lines:
            parts = re.split(r"\s*[;|•·]\s*|,\s+(?=[A-Z])", re.sub(r"^-\s*", "", line.text))
            values.extend(x.strip() for x in parts if x.strip())
        return RuleBasedExtractor._dedupe(values)

    @staticmethod
    def languages(lines: Sequence[LineHint]) -> list[dict[str, Optional[str]]]:
        levels = r"native|fluent|professional|advanced|intermediate|basic|elementary|beginner|conversational|proficient"
        output = []
        for value in EntryParser.list_section(lines):
            match = re.match(rf"(.+?)(?:\s*[-:|(]\s*({levels})\)?)?$", value, re.I)
            if match:
                output.append({"language": match.group(1).strip(),
                               "proficiency": match.group(2).title() if match.group(2) else None})
        return output

    @staticmethod
    def _company_heuristic(lines: Sequence[str], title: Optional[str]) -> Optional[str]:
        for line in lines[:4]:
            clean = re.sub(DateParser.RANGE_RE, "", line).strip(" |-—–,")
            if clean and clean != title and not RuleBasedExtractor.JOB_RE.search(clean) and len(clean.split()) <= 10:
                return clean
        return None

    @staticmethod
    def _employment_type(text: str) -> Optional[str]:
        types = {"full time": "Full-time", "full-time": "Full-time", "part time": "Part-time",
                 "part-time": "Part-time", "internship": "Internship", "intern": "Internship",
                 "freelance": "Freelance", "contract": "Contract", "temporary": "Temporary",
                 "volunteer": "Volunteer", "self-employed": "Self-employed"}
        lower = text.casefold()
        return next((display for key, display in types.items() if re.search(rf"\b{re.escape(key)}\b", lower)), None)

    @staticmethod
    def _without_dates(text: str) -> str:
        return DateParser.RANGE_RE.sub("", text)

    @staticmethod
    def _duration_months(start: Optional[str], end: Optional[str]) -> Optional[int]:
        interval = HybridMerger.to_month_interval(start, end)
        return interval[1] - interval[0] + 1 if interval else None

    @staticmethod
    def _nonempty(items: list[dict[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
        return [x for x in items if any(x.get(field) for field in fields)]


class HybridMerger:
    """Validate conflicts, deduplicate values, and calculate derived fields."""

    @staticmethod
    def to_month_interval(start: Optional[str], end: Optional[str]) -> Optional[tuple[int, int]]:
        def month_index(value: Optional[str], is_end: bool = False) -> Optional[int]:
            if not value:
                return None
            if value == "present":
                now = datetime.now()
                return now.year * 12 + now.month - 1
            match = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", value)
            if not match:
                return None
            year = int(match.group(1))
            month = int(match.group(2) or (12 if is_end else 1))
            return year * 12 + month - 1
        s, e = month_index(start), month_index(end, True)
        return (s, e) if s is not None and e is not None and e >= s else None

    @classmethod
    def total_experience(cls, experience: Sequence[dict[str, Any]]) -> Optional[float]:
        intervals = [cls.to_month_interval(x.get("start_date"), x.get("end_date")) for x in experience]
        valid = sorted(x for x in intervals if x)
        if not valid:
            return None
        merged: list[list[int]] = []
        for start, end in valid:
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        months = sum(end - start + 1 for start, end in merged)
        return round(months / 12, 2)


def empty_result(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    return {
        "personal_information": {
            "full_name": None, "first_name": None, "last_name": None,
            "emails": [], "phones": [],
            "location": {"city": None, "state": None, "country": None, "raw": None},
            "links": {"linkedin": None, "github": None, "portfolio": None, "other": []},
            "date_of_birth": None, "nationality": None, "gender": None, "marital_status": None,
        },
        "summary": None,
        "skills": {"technical_skills": [], "soft_skills": [], "tools_and_technologies": [], "all_skills": []},
        "education": [], "qualifications": [], "experience": [], "total_experience_years": None,
        "projects": [], "certifications": [], "languages": [], "awards_and_achievements": [],
        "publications": [], "volunteer_experience": [], "interests": [], "references": [],
        "additional_sections": {},
        "metadata": {
            "file_name": path.name or None, "file_type": path.suffix.lower().lstrip(".") or None,
            "page_count": None, "parsed_at": datetime.now().astimezone().isoformat(),
            "parser_version": VERSION, "methods_used": [], "ocr_used": False,
            "confidence_scores": {}, "warnings": [], "errors": [],
        },
    }


class CVParser:
    """Main hybrid parser orchestration class."""

    def __init__(self) -> None:
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.detector = SectionDetector()
        self.rules = RuleBasedExtractor()

    def parse(self, file_path: str | Path) -> dict[str, Any]:
        result = empty_result(file_path)
        loaded = self.loader.load(file_path)
        metadata = result["metadata"]
        metadata.update({"file_type": loaded.file_type, "page_count": loaded.page_count,
                         "methods_used": loaded.methods, "ocr_used": loaded.ocr_used,
                         "warnings": loaded.warnings, "errors": loaded.errors})
        if loaded.errors and not loaded.lines:
            return result
        lines = self.cleaner.clean(loaded)
        if not lines:
            metadata["errors"].append("No usable text remained after normalization")
            return result
        metadata["methods_used"].append("rules")
        ner = NERExtractor(metadata["warnings"], metadata["methods_used"])
        entry_parser = EntryParser(self.rules, ner)
        sections, additional, section_scores = self.detector.split(lines)
        full_text = "\n".join(x.text for x in lines)
        contacts = self.rules.extract_contacts(full_text)
        result["personal_information"]["emails"] = contacts["emails"]
        result["personal_information"]["phones"] = contacts["phones"]
        result["personal_information"]["links"] = contacts["links"]
        result["personal_information"].update(self.rules.extract_sensitive_labels(lines))
        self._extract_identity(result, lines, sections.get("preamble", []), ner)
        result["summary"] = self._section_text(sections.get("summary", []))
        skill_text = self._section_text(sections.get("skills", [])) or ""
        result["skills"] = self.rules.extract_skills(skill_text, in_skill_section=True)
        contextual_skills = self.rules.extract_skills(full_text, in_skill_section=False)
        for key in result["skills"]:
            result["skills"][key] = self.rules._dedupe(result["skills"][key] + contextual_skills[key])
        result["education"] = entry_parser.education(sections.get("education", []))
        result["qualifications"] = entry_parser.list_section(sections.get("qualifications", []))
        result["experience"] = entry_parser.experience(sections.get("experience", []))
        result["projects"] = entry_parser.projects(sections.get("projects", []))
        result["certifications"] = entry_parser.certifications(sections.get("certifications", []))
        result["languages"] = entry_parser.languages(sections.get("languages", []))
        for key in ("awards_and_achievements", "publications", "volunteer_experience", "interests", "references"):
            result[key] = entry_parser.list_section(sections.get(key, []))
        result["additional_sections"] = {
            heading: {"content": self._section_text(body), "raw_text": "\n".join(x.text for x in body)}
            for heading, body in additional.items()
        }
        for heading in additional:
            metadata["warnings"].append(f"Section '{heading}' not recognized; stored in additional_sections")
        result["total_experience_years"] = HybridMerger.total_experience(result["experience"])
        metadata["confidence_scores"] = self._confidence(result, section_scores)
        self._warnings(result)
        metadata["warnings"] = self.rules._dedupe(metadata["warnings"])
        metadata["methods_used"] = self.rules._dedupe(metadata["methods_used"])
        return result

    def _extract_identity(self, result: dict[str, Any], lines: Sequence[LineHint],
                          preamble: Sequence[LineHint], ner: NERExtractor) -> None:
        top = list(preamble[:12]) or list(lines[:12])
        top_text = "\n".join(x.text for x in top)
        entities = ner.entities(top_text)
        rejected = re.compile(r"\b(?:curriculum vitae|resume|cv|profile|summary|engineer|manager|developer|"
                              r"specialist|consultant|director|designer|analyst|address|phone|email)\b", re.I)
        candidates = [x for x in entities["PERSON"] if 1 < len(x.split()) <= 5 and not rejected.search(x)]
        for line in top[:6]:
            text = line.text.strip()
            if (2 <= len(text.split()) <= 5 and re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ' .-]+", text)
                    and not rejected.search(text) and not self.rules.DEGREE_RE.search(text)):
                candidates.append(text)
        email_user = result["personal_information"]["emails"][0].split("@")[0] if result["personal_information"]["emails"] else ""
        def score(name: str) -> float:
            words = [x.casefold() for x in re.findall(r"[A-Za-z]+", name)]
            email_score = sum(1 for x in words if len(x) > 2 and x in email_user.casefold()) * 0.2
            line_score = 0.2 if any(x.text.strip().casefold() == name.casefold() and (x.bold or x.font_size) for x in top) else 0
            return 0.55 + email_score + line_score
        name = max(self.rules._dedupe(candidates), key=score, default=None)
        pi = result["personal_information"]
        pi["full_name"] = name
        if name:
            parts = name.split()
            pi["first_name"], pi["last_name"] = parts[0], parts[-1] if len(parts) > 1 else None
        location_raw = next((x for x in entities["GPE"] + entities["LOC"] if x.casefold() != (name or "").casefold()), None)
        if location_raw:
            parts = [x.strip() for x in location_raw.split(",")]
            pi["location"] = {"city": parts[0] or None,
                              "state": parts[1] if len(parts) > 2 else None,
                              "country": parts[-1] if len(parts) > 1 else None,
                              "raw": location_raw}

    @staticmethod
    def _section_text(lines: Sequence[LineHint]) -> Optional[str]:
        text = "\n".join(re.sub(r"^-\s*", "", x.text) for x in lines).strip()
        return text or None

    @staticmethod
    def _confidence(result: dict[str, Any], heading_scores: dict[str, float]) -> dict[str, float]:
        pi = result["personal_information"]
        personal = 0.2 + (0.3 if pi["full_name"] else 0) + (0.25 if pi["emails"] else 0) + (0.15 if pi["phones"] else 0)
        scores = {
            "personal_information": round(min(personal, 1), 2),
            "summary": round(heading_scores.get("summary", 0.35 if result["summary"] else 0), 2),
            "skills": round(max(heading_scores.get("skills", 0), 0.65 if result["skills"]["all_skills"] else 0), 2),
            "education": round(max(heading_scores.get("education", 0), 0.55 if result["education"] else 0), 2),
            "experience": round(max(heading_scores.get("experience", 0), 0.55 if result["experience"] else 0), 2),
            "projects": round(max(heading_scores.get("projects", 0), 0.55 if result["projects"] else 0), 2),
            "certifications": round(max(heading_scores.get("certifications", 0), 0.55 if result["certifications"] else 0), 2),
        }
        return scores

    @staticmethod
    def _warnings(result: dict[str, Any]) -> None:
        warnings = result["metadata"]["warnings"]
        pi = result["personal_information"]
        if not pi["emails"]:
            warnings.append("No email found")
        if not pi["full_name"]:
            warnings.append("Candidate name could not be identified confidently")
        if not result["education"]:
            warnings.append("No structured education entries found")
        if not result["experience"]:
            warnings.append("No structured work experience entries found")


def parse_cv(file_path: str | Path) -> dict[str, Any]:
    """Public function: parse a PDF or DOCX resume into the documented schema."""
    try:
        return CVParser().parse(file_path)
    except Exception as exc:  # caller-facing guarantee: malformed input never crashes
        LOGGER.exception("Unexpected parser failure")
        result = empty_result(file_path)
        result["metadata"]["errors"].append(f"Unexpected parser error: {exc}")
        return result


UPLOAD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid CV Parser</title><style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:4rem auto;padding:0 1rem;color:#172033}
.card{border:1px solid #d7dce5;border-radius:14px;padding:2rem;box-shadow:0 8px 28px #15213b12}
h1{margin-top:0}input,button{font:inherit}input{display:block;margin:1rem 0;width:100%}
button{background:#2457d6;color:white;border:0;border-radius:8px;padding:.75rem 1rem;cursor:pointer}
small{color:#5d687c}</style></head><body><div class="card"><h1>Hybrid CV Parser</h1>
<p>Upload a PDF or DOCX resume. Processing stays on this machine.</p>
<form action="/parse" method="post" enctype="multipart/form-data">
<input type="file" name="file" accept=".pdf,.docx" required><button type="submit">Parse CV</button></form>
<p><small>Maximum upload size: 20 MB.</small></p></div></body></html>"""


def create_app() -> Any:
    """Create the optional FastAPI application without requiring it for CLI use."""
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile  # type: ignore
        from fastapi.responses import HTMLResponse, JSONResponse  # type: ignore
    except ImportError as exc:
        raise RuntimeError("FastAPI mode requires: pip install fastapi uvicorn python-multipart") from exc

    app = FastAPI(title="Hybrid CV Parser", version=VERSION)

    @app.get("/", response_class=HTMLResponse)
    async def upload_page() -> str:
        return UPLOAD_HTML

    @app.post("/parse")
    async def parse_endpoint(file: UploadFile = File(...)) -> Any:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".docx"}:
            raise HTTPException(status_code=415, detail="Only PDF and DOCX files are supported")
        max_size = 20 * 1024 * 1024
        content = await file.read(max_size + 1)
        if len(content) > max_size:
            raise HTTPException(status_code=413, detail="File exceeds the 20 MB limit")
        with tempfile.TemporaryDirectory(prefix="cv-parser-") as temp_dir:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(file.filename or f"upload{suffix}").name)
            path = Path(temp_dir) / safe_name
            path.write_bytes(content)
            parsed = parse_cv(path)
        status = 422 if parsed["metadata"]["errors"] else 200
        return JSONResponse(content=parsed, status_code=status)

    return app


def _write_json(data: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_folder(folder: Path, output: Optional[Path]) -> int:
    files = sorted(x for x in folder.rglob("*") if x.is_file() and x.suffix.lower() in {".pdf", ".docx"})
    if not files:
        LOGGER.error("No PDF or DOCX files found in %s", folder)
        return 1
    target = output or folder / "parsed"
    target.mkdir(parents=True, exist_ok=True)
    failures = 0
    for source in files:
        result = parse_cv(source)
        relative = source.relative_to(folder)
        digest = hashlib.sha1(str(relative).encode()).hexdigest()[:8]
        destination = target / relative.parent / f"{source.stem}_{digest}.json"
        _write_json(result, destination)
        print(f"{source} -> {destination}")
        failures += bool(result["metadata"]["errors"])
    return 2 if failures else 0


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse PDF/DOCX resumes into structured JSON")
    parser.add_argument("path", nargs="?", help="Resume file or folder of resumes")
    parser.add_argument("--output", "-o", help="Output JSON file or folder")
    parser.add_argument("--serve", action="store_true", help="Start the local FastAPI upload interface")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable detailed logs")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_cli().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.serve:
        try:
            import uvicorn  # type: ignore
            uvicorn.run(create_app(), host=args.host, port=args.port)
            return 0
        except Exception as exc:
            LOGGER.error("Could not start web interface: %s", exc)
            return 2
    if not args.path:
        build_cli().print_help()
        return 1
    source = Path(args.path).expanduser()
    output = Path(args.output).expanduser() if args.output else None
    if source.is_dir():
        return _parse_folder(source, output)
    result = parse_cv(source)
    if output:
        _write_json(result, output)
        print(output)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["metadata"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
