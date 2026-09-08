"""
Turn MEMO.md and DECK.md into the artefacts the brief actually asks for:
a 2-page memo and a 6-slide deck.

Markdown stays the single source of truth; this only renders it. The script
asserts the page and slide counts at the end, so the brief's limits are
verified rather than estimated.

    python build_pdfs.py
"""
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent

# Helvetica has no Indian Rupee glyph (U+20B9) and renders it as a black box.
# DejaVu Sans does, and ships with matplotlib, so register it and switch to it
# for that one character only -- the rest of the typography stays Helvetica.
RUPEE_FONT = "Helvetica"


def _register_rupee_font():
    global RUPEE_FONT
    try:
        import matplotlib
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont as RLTTFont
        base = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        for name, fname in [("DejaVuSans", "DejaVuSans.ttf"),
                            ("DejaVuSans-Bold", "DejaVuSans-Bold.ttf")]:
            p = base / fname
            if p.exists():
                pdfmetrics.registerFont(RLTTFont(name, str(p)))
        pdfmetrics.getFont("DejaVuSans")
        RUPEE_FONT = "DejaVuSans"
    except Exception:
        RUPEE_FONT = None  # fall back to writing "Rs"


_register_rupee_font()

INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#c0392b")
GREY = colors.HexColor("#6b6b6b")
RULE = colors.HexColor("#d5d5d5")
BAND = colors.HexColor("#f2f2f2")


# ---------------------------------------------------------------- inline markdown
def inline(text: str) -> str:
    """Convert the inline markdown we actually use into reportlab's mini-HTML."""
    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    t = t.replace("—", "\u2014").replace("→", "\u2192").replace("←", "\u2190")
    if "\u20b9" in t:
        t = (t.replace("\u20b9", f"<font face='{RUPEE_FONT}'>\u20b9</font>")
             if RUPEE_FONT else t.replace("\u20b9", "Rs "))
    return t


def split_row(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


# ---------------------------------------------------------------- block parser
def parse_blocks(md: str):
    """Yield ('h1'|'h2'|'h3'|'p'|'bullet'|'table'|'image'|'rule', payload)."""
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if not s:
            i += 1
            continue
        if s.startswith("![") :
            m = re.search(r"\((.+?)\)", s)
            yield ("image", m.group(1) if m else "")
            i += 1
            continue
        if re.fullmatch(r"-{3,}", s):
            yield ("rule", None)
            i += 1
            continue
        if s.startswith("### "):
            yield ("h3", s[4:])
            i += 1
            continue
        if s.startswith("## "):
            yield ("h2", s[3:])
            i += 1
            continue
        if s.startswith("# "):
            yield ("h1", s[2:])
            i += 1
            continue
        if s.startswith("|"):
            rows, header = [], None
            while i < len(lines) and lines[i].strip().startswith("|"):
                if is_sep(lines[i]):
                    header = rows.pop() if rows else None
                else:
                    rows.append(split_row(lines[i]))
                i += 1
            yield ("table", (header, rows))
            continue
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
            items = []
            while i < len(lines):
                cur = lines[i].strip()
                if re.match(r"^[-*]\s+", cur):
                    items.append(re.sub(r"^[-*]\s+", "", cur))
                elif re.match(r"^\d+\.\s+", cur):
                    items.append(re.sub(r"^\d+\.\s+", "", cur))
                elif cur and not cur.startswith(("|", "#", "!")) and items:
                    items[-1] += " " + cur           # wrapped continuation
                else:
                    break
                i += 1
            yield ("bullet", items)
            continue
        # paragraph: consume until blank or a new block starts
        para = [s]
        i += 1
        while i < len(lines):
            cur = lines[i].strip()
            if not cur or cur.startswith(("|", "#", "!", "- ", "* ")) or re.fullmatch(r"-{3,}", cur):
                break
            para.append(cur)
            i += 1
        yield ("p", " ".join(para))


# ---------------------------------------------------------------- memo
MEMO_STYLES = dict(
    h1=ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14.5, leading=17,
                      textColor=INK, spaceAfter=3),
    h2=ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=9.8, leading=12,
                      textColor=ACCENT, spaceBefore=6.5, spaceAfter=2.5),
    p=ParagraphStyle("p", fontName="Helvetica", fontSize=8.1, leading=10.3,
                     textColor=INK, alignment=TA_LEFT, spaceAfter=3.6),
    meta=ParagraphStyle("meta", fontName="Helvetica", fontSize=7.6, leading=9.6,
                        textColor=GREY, spaceAfter=4),
    bullet=ParagraphStyle("bullet", fontName="Helvetica", fontSize=8.1, leading=10.3,
                          textColor=INK, leftIndent=8.5, bulletIndent=1.5,
                          spaceAfter=2.6),
    cell=ParagraphStyle("cell", fontName="Helvetica", fontSize=7.5, leading=9.2,
                        textColor=INK),
    cellh=ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=7.5,
                         leading=9.2, textColor=INK),
)


def memo_table(header, rows, width):
    st = MEMO_STYLES
    data, raw_rows = [], []
    if header:
        data.append([Paragraph(inline(c), st["cellh"]) for c in header])
        raw_rows.append(header)
    for r in rows:
        data.append([Paragraph(inline(c), st["cell"]) for c in r])
        raw_rows.append(r)
    ncol = max(len(r) for r in raw_rows)
    data = [r + [Paragraph("", st["cell"])] * (ncol - len(r)) for r in data]
    widths = column_widths(raw_rows, ncol, width,
                           font_size=st["cell"].fontSize, pad=4.0)
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BAND),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.6, GREY)]
    t.setStyle(TableStyle(style))
    return t


def build_memo(md_path: Path, out: Path):
    st = MEMO_STYLES
    doc = BaseDocTemplate(str(out), pagesize=A4,
                          leftMargin=13 * mm, rightMargin=13 * mm,
                          topMargin=11 * mm, bottomMargin=11 * mm,
                          title="Captain onboarding memo", author="Data Science")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="memo", frames=[frame])])

    flow, first_para = [], True
    for kind, payload in parse_blocks(md_path.read_text(encoding="utf-8")):
        if kind == "h1":
            flow.append(Paragraph(inline(payload), st["h1"]))
        elif kind == "h2":
            flow.append(Paragraph(inline(payload).upper(), st["h2"]))
        elif kind == "h3":
            flow.append(Paragraph(inline(payload), st["h2"]))
        elif kind == "p":
            # the To/From/Basis block right after the title is metadata
            style = st["meta"] if first_para else st["p"]
            flow.append(Paragraph(inline(payload), style))
            first_para = False
        elif kind == "bullet":
            for it in payload:
                flow.append(Paragraph(inline(it), st["bullet"], bulletText="\u2022"))
            flow.append(Spacer(1, 1.6))
        elif kind == "table":
            header, rows = payload
            flow.append(Spacer(1, 1.6))
            flow.append(memo_table(header, rows, doc.width))
            flow.append(Spacer(1, 3.4))
    doc.build(flow)
    return out


# ---------------------------------------------------------------- deck
# Base sizes are presentation-scale. Each slide is then measured and its
# typography scaled to fill the page without overflowing it, so a sparse slide
# reads large and a dense one still fits. See fit_slide().
def deck_styles(s: float = 1.0):
    return dict(
        kicker=ParagraphStyle("kicker", fontName="Helvetica-Bold", fontSize=10 * s,
                              leading=12 * s, textColor=ACCENT, spaceAfter=4 * s),
        title=ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=23 * s,
                             leading=27 * s, textColor=INK, spaceAfter=11 * s),
        sub=ParagraphStyle("sub", fontName="Helvetica-Bold", fontSize=14.5 * s,
                           leading=18 * s, textColor=INK, spaceAfter=9 * s),
        p=ParagraphStyle("p", fontName="Helvetica", fontSize=12 * s,
                         leading=15.5 * s, textColor=INK, spaceAfter=7 * s),
        bullet=ParagraphStyle("bullet", fontName="Helvetica", fontSize=12 * s,
                              leading=15.5 * s, textColor=INK, leftIndent=13 * s,
                              bulletIndent=2 * s, spaceAfter=5 * s),
        cell=ParagraphStyle("cell", fontName="Helvetica", fontSize=10.2 * s,
                            leading=12.6 * s, textColor=INK),
        cellh=ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=10.2 * s,
                             leading=12.6 * s, textColor=INK),
        pad=4.2 * s,
    )


def column_widths(raw_rows, ncol, total, font_size=10.2, pad=6.0):
    """
    Allocate table width by how much text each column actually holds.

    Equal columns waste space on a '#' column and cramp a prose one. But
    weighting by length alone can starve a column below its longest single word,
    and reportlab then hyphen-lessly splits it mid-word ("app rovals"). So every
    column first gets enough room for its longest unbreakable token, and only
    the remainder is shared out by text volume.
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth

    def clean(s):
        return re.sub(r"[*`]", "", s)

    mins, weights = [], []
    for c in range(ncol):
        cells = [clean(r[c]) for r in raw_rows if c < len(r)] or [""]
        words = [w for cell in cells for w in cell.split()] or [""]
        longest = max(words, key=len)
        # measure in bold: headers and emphasis are the widest case
        mins.append(stringWidth(longest, "Helvetica-Bold", font_size) + 2 * pad + 2)
        lens = [len(x) for x in cells]
        weights.append(max(1.0, (max(lens) + sum(lens) / len(lens)) / 2))

    if sum(mins) >= total:                       # cannot honour minima at this size
        k = total / sum(mins)
        return [m * k for m in mins]

    w = [total * x / sum(weights) for x in weights]
    w = [max(w[i], mins[i]) for i in range(ncol)]
    over = sum(w) - total
    if over > 0:
        slack = [w[i] - mins[i] for i in range(ncol)]
        if sum(slack) > 0:
            w = [w[i] - over * slack[i] / sum(slack) for i in range(ncol)]
    else:
        left = -over
        w = [w[i] + left * weights[i] / sum(weights) for i in range(ncol)]
    return w


def deck_table(header, rows, width, st):
    data, raw_rows = [], []
    if header:
        data.append([Paragraph(inline(c), st["cellh"]) for c in header])
        raw_rows.append(header)
    for r in rows:
        data.append([Paragraph(inline(c), st["cell"]) for c in r])
        raw_rows.append(r)
    ncol = max(len(r) for r in raw_rows)
    data = [r + [Paragraph("", st["cell"])] * (ncol - len(r)) for r in data]
    widths = column_widths(raw_rows, ncol, width,
                           font_size=st["cell"].fontSize, pad=st["pad"] + 1)
    t = Table(data, colWidths=widths, hAlign="LEFT")
    pad = st["pad"]
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LEFTPADDING", (0, 0), (-1, -1), pad + 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), pad + 1),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BAND),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.6, GREY)]
    t.setStyle(TableStyle(style))
    return t


def build_deck(md_path: Path, out: Path):
    page = landscape(A4)
    doc = BaseDocTemplate(str(out), pagesize=page,
                          leftMargin=14 * mm, rightMargin=14 * mm,
                          topMargin=11 * mm, bottomMargin=9 * mm,
                          title="Captain onboarding deck", author="Data Science")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawRightString(page[0] - 14 * mm, 5.5 * mm, f"{canvas.getPageNumber()} / 6")
        canvas.drawString(14 * mm, 5.5 * mm, "Captain Acquisition, Supply \u2014 30 June 2026")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="slide", frames=[frame], onPage=footer)])

    blocks = list(parse_blocks(md_path.read_text(encoding="utf-8")))
    # split into slides on each '## Slide' heading; drop the cover preamble
    slides, cur = [], None
    for kind, payload in blocks:
        if kind == "h2" and payload.lower().startswith("slide"):
            if cur is not None:
                slides.append(cur)
            cur = [(kind, payload)]
        elif cur is not None and kind != "rule":
            cur.append((kind, payload))
    if cur is not None:
        slides.append(cur)

    flow = []
    for n, blocks_ in enumerate(slides):
        flow.extend(fit_slide(blocks_, doc.width, doc.height))
        if n < len(slides) - 1:
            from reportlab.platypus import PageBreak
            flow.append(PageBreak())
    doc.build(flow)
    return out, len(slides)


def slide_flowables(blocks_, width, height, scale):
    """Render one slide's blocks at a given typographic scale."""
    from reportlab.lib.utils import ImageReader
    st = deck_styles(scale)
    out = []
    n_img = sum(1 for k, _ in blocks_ if k == "image")
    for kind, payload in blocks_:
        if kind == "h2":
            num, _, rest = payload.partition("\u2014")
            out.append(Paragraph(inline(num.strip().upper()), st["kicker"]))
            if rest.strip():
                out.append(Paragraph(inline(rest.strip()), st["title"]))
        elif kind == "h3":
            out.append(Paragraph(inline(payload), st["sub"]))
        elif kind == "p":
            out.append(Paragraph(inline(payload), st["p"]))
        elif kind == "bullet":
            for it in payload:
                out.append(Paragraph(inline(it), st["bullet"], bulletText="\u2022"))
            out.append(Spacer(1, 3 * scale))
        elif kind == "table":
            header, rows = payload
            out.append(Spacer(1, 2 * scale))
            out.append(deck_table(header, rows, width, st))
            out.append(Spacer(1, 6 * scale))
        elif kind == "image":
            p = ROOT / payload
            if not p.exists():
                raise FileNotFoundError(f"chart missing: {p} (run run_all.py first)")
            iw, ih = ImageReader(str(p)).getSize()
            budget = 0.60 if n_img == 1 else 0.40
            maxw, maxh = width * 0.97, height * budget * scale
            k = min(maxw / iw, maxh / ih)
            out.append(Spacer(1, 3 * scale))
            out.append(Image(str(p), iw * k, ih * k, hAlign="CENTER"))
            out.append(Spacer(1, 6 * scale))
    return out


def _measure(flowables, width, height) -> float:
    total = 0.0
    for f in flowables:
        try:
            _, h = f.wrap(width, height)
        except Exception:
            h = 0
        total += h + getattr(f, "getSpaceBefore", lambda: 0)() \
                   + getattr(f, "getSpaceAfter", lambda: 0)()
    return total


def fit_slide(blocks_, width, height):
    """
    Pick the largest typographic scale whose content still fits on one slide.

    Without this, a light slide occupies a third of the page and reads like a
    document, while a dense one silently spills onto a seventh page and breaks
    the brief's six-slide limit.
    """
    lo, hi = 0.45, 1.30
    best = None
    for _ in range(18):
        mid = (lo + hi) / 2
        fl = slide_flowables(blocks_, width, height, mid)
        if _measure(fl, width, height) <= height * 0.985:
            best = fl
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.005:
            break
    if best is None:                       # even the smallest scale overflows
        best = slide_flowables(blocks_, width, height, 0.45)
    # nudge short slides down from the very top edge for balance
    used = _measure(best, width, height)
    slack = height - used
    if slack > height * 0.10:
        best = [Spacer(1, min(slack * 0.22, 34))] + best
    return best


# ---------------------------------------------------------------- page counting
def page_count(pdf: Path) -> int:
    """Count pages without a PDF library: /Type /Page objects in the raw file."""
    raw = pdf.read_bytes()
    n = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    if n == 0:  # some writers use /Count in the page tree
        m = re.findall(rb"/Count\s+(\d+)", raw)
        n = max(int(x) for x in m) if m else 0
    return n


def main():
    charts = ROOT / "outputs" / "charts"
    if not charts.exists():
        print("outputs/charts missing - run `python run_all.py` first.")
        return 1

    deliv = ROOT / "deliverables"
    memo = build_memo(deliv / "MEMO.md", deliv / "MEMO.pdf")
    deck, n_slides = build_deck(deliv / "DECK.md", deliv / "DECK.pdf")

    mp, dp = page_count(memo), page_count(deck)
    print(f"  MEMO.pdf   {mp} page(s)   limit 2   {'OK' if mp <= 2 else 'OVER LIMIT'}")
    print(f"  DECK.pdf   {dp} slide(s)  limit 6   {'OK' if dp <= 6 else 'OVER LIMIT'}")
    print(f"  slides parsed from DECK.md: {n_slides}")

    if mp > 2 or dp > 6:
        print("\nFAIL: a deliverable is over the limit set in the brief.")
        return 1
    print("\nBoth deliverables are within the brief's limits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
