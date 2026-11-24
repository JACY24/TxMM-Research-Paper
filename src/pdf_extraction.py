#!/usr/bin/env python3
"""
Extract raw text from PDFs.

- Text only: images are never extracted.
- Headers/footers/page numbers are skipped (standalone page numbers, plus text
  that repeats in the top/bottom margin across pages).
- Multi-column layouts are read column by column, also when the number of
  columns changes within a page (title + intro paragraph + 4 columns, a boxed
  side panel, a full-width caption between two columns, ...).

How the reading order is determined (recursive "XY-cut" on word positions):
  Starting with all words on the page, a region is split recursively:
    * a full-height empty vertical strip (a gutter) splits it into a left and a
      right part (left is read first), or
    * the largest empty horizontal strip splits it into a top and a bottom part.
  A gutter is preferred unless a much larger horizontal gap exists. Every part is
  analysed on its own, so each part of a page can have its own column layout.
  Text is therefore read per column: column 1 top to bottom, then column 2, ...

Backends: PyMuPDF (pip install pymupdf) if available, else pdfplumber
(pip install pdfplumber). Both give word positions; the layout logic is shared.

Usage:
    python extract_pdf_text.py file.pdf                  # print to stdout
    python extract_pdf_text.py file.pdf -o out/          # write out/file.txt
    python extract_pdf_text.py folder/ -o out/           # all PDFs in folder
    python extract_pdf_text.py file.pdf --debug          # show the cuts it makes
"""

import argparse
import re
import statistics
import sys
from collections import Counter, namedtuple
from pathlib import Path

Word = namedtuple("Word", "x0 y0 x1 y1 text")

PAGE_NUM_RE = re.compile(
    r"^\s*(?:page|pagina|pag|blz|p|s|seite)?\.?\s*[-–—•·]?\s*"
    r"(?:\d{1,4}|[ivxlcdm]{1,7})\s*[-–—•·.]?"
    r"(?:\s*(?:of|van|/)\s*\d{1,4})?\s*$",
    re.IGNORECASE,
)

sys.setrecursionlimit(20000)


# --------------------------------------------------------------------------
# Basic geometry helpers
# --------------------------------------------------------------------------
class Line:
    """A visual line: words that share (roughly) the same baseline."""

    def __init__(self, words):
        self.words = sorted(words, key=lambda w: w.x0)
        self.x0 = min(w.x0 for w in self.words)
        self.x1 = max(w.x1 for w in self.words)
        self.y0 = min(w.y0 for w in self.words)
        self.y1 = max(w.y1 for w in self.words)
        self.yc = sum((w.y0 + w.y1) / 2 for w in self.words) / len(self.words)
        self.text = " ".join(w.text for w in self.words)


def normalize(text):
    """Normalize so 'Page 3' and 'Page 4' compare equal."""
    text = re.sub(r"\d+", "#", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def cluster_lines(words, tol=0.4):
    """Group words into visual lines by vertical centre."""
    groups = []
    for w in sorted(words, key=lambda w: (w.y0 + w.y1) / 2):
        yc = (w.y0 + w.y1) / 2
        h = max(w.y1 - w.y0, 1e-6)
        if groups and abs(yc - groups[-1]["yc"]) <= tol * max(h, groups[-1]["h"]):
            g = groups[-1]
            g["words"].append(w)
            g["sum"] += yc
            g["yc"] = g["sum"] / len(g["words"])
            g["h"] = max(g["h"], h)
        else:
            groups.append({"words": [w], "sum": yc, "yc": yc, "h": h})
    return [Line(g["words"]) for g in groups]


def split_line(line, gap):
    """Split a visual line where there is a horizontal gap >= gap."""
    segs, end = [[line.words[0]]], line.words[0].x1
    for w in line.words[1:]:
        if w.x0 - end >= gap:
            segs.append([w])
        else:
            segs[-1].append(w)
        end = max(end, w.x1)
    return [Line(s) for s in segs]


def free_gaps(intervals):
    """Empty stretches between the union of the given (start, end) intervals."""
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]


# --------------------------------------------------------------------------
# Reading order: recursive XY-cut
# --------------------------------------------------------------------------
class XYCutter:
    def __init__(self, min_gutter=None, v_bias=0.5, debug=None):
        self.min_gutter = min_gutter  # points; None = automatic
        self.v_bias = v_bias  # a gutter wins if width >= v_bias * best horizontal gap
        self.debug = debug

    def run(self, words):
        leaves = []
        self._cut(words, leaves, 0)
        return leaves  # list of lists of Line, in reading order

    def _cut(self, words, leaves, depth):
        if not words:
            return
        lines = cluster_lines(words)
        if len(lines) <= 1:
            leaves.append(lines)
            return

        heights = [w.y1 - w.y0 for w in words]
        med_h = statistics.median(heights)
        x_lo = min(w.x0 for w in words)
        x_hi = max(w.x1 for w in words)
        y_lo = min(w.y0 for w in words)

        # Horizontal gaps, measured on the "core" of each word (centre +- 0.3 h)
        # so slightly overlapping line boxes do not hide the gap between lines.
        core = []
        for w in words:
            yc, h = (w.y0 + w.y1) / 2, w.y1 - w.y0
            core.append((yc - 0.3 * h, yc + 0.3 * h))
        hgaps = [(a, b, (b - a) + 0.6 * med_h) for a, b in free_gaps(core)]
        best_h = None
        if hgaps:
            mid = (min(c[0] for c in core) + max(c[1] for c in core)) / 2
            best_h = max(hgaps, key=lambda g: (round(g[2], 1), -abs((g[0] + g[1]) / 2 - mid)))

        # Vertical gutters: empty full-height strips. Both sides must be wide
        # enough to be a text column (so bullets / numbering are not "columns").
        min_v = self.min_gutter if self.min_gutter else max(5.0, 0.8 * med_h)
        min_piece = max(20.0, 3 * min_v)
        vgaps = [
            (a, b) for a, b in free_gaps([(w.x0, w.x1) for w in words])
            if (b - a) >= min_v and (a - x_lo) >= min_piece and (x_hi - b) >= min_piece
        ]
        best_v = max(vgaps, key=lambda g: g[1] - g[0]) if vgaps else None

        if best_v and (best_h is None or (best_v[1] - best_v[0]) >= self.v_bias * best_h[2]):
            cut = (best_v[0] + best_v[1]) / 2
            first = [w for w in words if (w.x0 + w.x1) / 2 < cut]
            second = [w for w in words if (w.x0 + w.x1) / 2 >= cut]
            if first and second:
                if self.debug:
                    self.debug(f"{'  ' * depth}V-cut at x={cut:.0f} (gutter {best_v[1]-best_v[0]:.0f}pt) "
                               f"in region y>={y_lo:.0f}, {len(lines)} lines")
                self._cut(first, leaves, depth + 1)
                self._cut(second, leaves, depth + 1)
                return
        if best_h:
            cut = (best_h[0] + best_h[1]) / 2
            first = [w for w in words if (w.y0 + w.y1) / 2 < cut]
            second = [w for w in words if (w.y0 + w.y1) / 2 >= cut]
            if first and second:
                if self.debug and best_h[2] >= 2.5 * med_h:
                    self.debug(f"{'  ' * depth}H-cut at y={cut:.0f} (gap {best_h[2]:.0f}pt), {len(lines)} lines")
                self._cut(first, leaves, depth + 1)
                self._cut(second, leaves, depth + 1)
                return
        leaves.append(lines)  # no cut possible: read top to bottom


# --------------------------------------------------------------------------
# Lines -> text
# --------------------------------------------------------------------------
def to_paragraphs(ordered):
    """Group lines (already in reading order) into paragraphs."""
    if not ordered:
        return []
    runs, cur = [], [ordered[0]]
    for a, b in zip(ordered, ordered[1:]):
        same_column = b.yc > a.yc and (min(a.x1, b.x1) - max(a.x0, b.x0)) > 0
        if same_column:
            cur.append(b)
        else:  # jumped to another column/region
            runs.append(cur)
            cur = [b]
    runs.append(cur)

    paras = []
    for run in runs:
        pitches = [b.yc - a.yc for a, b in zip(run, run[1:])]
        para = [run[0]]
        for i, b in enumerate(run[1:]):
            a = run[i]
            gap = b.y0 - a.y1  # empty space between the two lines
            h = min(a.y1 - a.y0, b.y1 - b.y0)
            # typical line pitch in the neighbourhood (excluding this pair)
            near = pitches[max(0, i - 3):i] + pitches[i + 1:i + 4]
            if near:
                new_para = gap > 0.9 * h and pitches[i] > 1.3 * statistics.median(near)
            else:
                new_para = gap > 1.0 * h
            if new_para:
                paras.append(para)
                para = [b]
            else:
                para.append(b)
        paras.append(para)
    return paras


def join_paragraph(texts):
    out = ""
    for t in texts:
        if not out:
            out = t
        elif len(out) > 1 and out.endswith("-") and out[-2].isalpha() and t[:1].islower():
            out = out[:-1] + t  # re-join hyphenated word
        else:
            out += " " + t
    return out


def layout_page_text(words, gutter=None, join=False, debug=None):
    if not words:
        return ""
    leaves = XYCutter(min_gutter=gutter, debug=debug).run(words)
    ordered = [ln for leaf in leaves for ln in leaf]
    paras = to_paragraphs(ordered)
    if join:
        return "\n\n".join(join_paragraph([ln.text for ln in p]) for p in paras)
    return "\n\n".join("\n".join(ln.text for ln in p) for p in paras)


# --------------------------------------------------------------------------
# Headers / footers
# --------------------------------------------------------------------------
def margin_segments(words):
    """Text pieces of a page: visual lines split at wide gaps (e.g. left footer
    text and a right-aligned page number on the same baseline are separate)."""
    if not words:
        return []
    med_h = statistics.median(w.y1 - w.y0 for w in words)
    gap = max(12.0, 1.5 * med_h)
    segs = []
    for ln in cluster_lines(words):
        segs.extend(split_line(ln, gap))
    return segs


def in_margin(seg, page_height, frac):
    return seg.yc < page_height * frac or seg.yc > page_height * (1 - frac)


def find_repeated(pages_segs, heights, frac, min_ratio=0.4):
    n = len(pages_segs)
    if n < 2:
        return set()
    counts = Counter()
    for segs, h in zip(pages_segs, heights):
        counts.update({normalize(s.text) for s in segs if in_margin(s, h, frac)})
    min_pages = max(2, int(n * min_ratio))
    return {t for t, c in counts.items() if t and c >= min_pages}


def remove_margins(segs, height, frac, repeated, strip_all):
    kept = []
    for s in segs:
        if in_margin(s, height, frac):
            if strip_all or PAGE_NUM_RE.match(s.text.strip()) or normalize(s.text) in repeated:
                continue
        kept.append(s)
    return [w for s in kept for w in s.words]


# --------------------------------------------------------------------------
# PDF reading (text only, never images)
# --------------------------------------------------------------------------
def read_pages(path, backend="auto"):
    """Return [(words, page_height), ...]."""
    if backend in ("auto", "pymupdf"):
        try:
            import pymupdf  # PyMuPDF
        except ImportError:
            if backend == "pymupdf":
                raise
        else:
            doc = pymupdf.open(path)
            try:
                return [([Word(*w[:5]) for w in p.get_text("words") if w[4].strip()],
                         p.rect.height) for p in doc]
            finally:
                doc.close()
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            ws = p.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False)
            pages.append(([Word(w["x0"], w["top"], w["x1"], w["bottom"], w["text"])
                           for w in ws if w["text"].strip()], float(p.height)))
    return pages


def extract_pdf(path, margin_frac=0.08, keep_margins=False, strip_all_margins=False,
                page_breaks=False, gutter=None, join=False, debug=False, backend="auto"):
    pages = read_pages(path, backend)
    heights = [h for _, h in pages]
    pages_segs = [margin_segments(w) for w, _ in pages]
    repeated = set() if keep_margins else find_repeated(pages_segs, heights, margin_frac)

    out = []
    for i, ((words, h), segs) in enumerate(zip(pages, pages_segs), 1):
        if not keep_margins:
            words = remove_margins(segs, h, margin_frac, repeated, strip_all_margins)
        dbg = None
        if debug:
            def dbg(m, i=i):
                print(f"[{Path(str(path)).name} p{i}] {m}", file=sys.stderr)
        out.append(layout_page_text(words, gutter=gutter, join=join, debug=dbg))
    sep = "\n\n----- page break -----\n\n" if page_breaks else "\n\n"
    return sep.join(p for p in out if p)


def main():
    ap = argparse.ArgumentParser(
        description="Extract text from PDFs (column-aware, no images, no headers/footers).")
    ap.add_argument("inputs", nargs="+", help="PDF files or directories")
    ap.add_argument("-o", "--output-dir", help="Write <name>.txt files here (default: stdout)")
    ap.add_argument("--margin", type=float, default=0.08,
                    help="Fraction of page height treated as header/footer zone (default 0.08)")
    ap.add_argument("--keep-margins", action="store_true", help="Do not remove headers/footers")
    ap.add_argument("--strip-all-margins", action="store_true",
                    help="Drop everything in the header/footer zone, not just repeated text/page numbers")
    ap.add_argument("--gutter", type=float, default=None,
                    help="Minimum width (points) of the empty strip between columns "
                         "(default: auto, ~0.8x text height)")
    ap.add_argument("--join-lines", action="store_true",
                    help="Join lines into paragraphs and re-join hyphenated words "
                         "(default: keep the PDF's line breaks)")
    ap.add_argument("--page-breaks", action="store_true", help="Mark page boundaries in the output")
    ap.add_argument("--backend", choices=["auto", "pymupdf", "pdfplumber"], default="auto",
                    help="Library used to read word positions (default: PyMuPDF if installed)")
    ap.add_argument("--debug", action="store_true", help="Print the layout cuts per page to stderr")
    args = ap.parse_args()

    pdfs = []
    for item in args.inputs:
        p = Path(item)
        pdfs.extend(sorted(p.rglob("*.pdf")) if p.is_dir() else [p])

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        try:
            text = extract_pdf(pdf, args.margin, args.keep_margins, args.strip_all_margins,
                               args.page_breaks, args.gutter, args.join_lines, args.debug,
                               args.backend)
        except Exception as exc:  # corrupt/encrypted file etc.
            print(f"Failed on {pdf}: {exc}", file=sys.stderr)
            continue
        if out_dir:
            (out_dir / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")
            print(f"Wrote {out_dir / (pdf.stem + '.txt')}", file=sys.stderr)
        else:
            if len(pdfs) > 1:
                print(f"===== {pdf.name} =====")
            print(text)


if __name__ == "__main__":
    main()