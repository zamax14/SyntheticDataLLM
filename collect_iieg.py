"""Collect varied IIEG publications for the existing QA generator.

Run from this directory with ``python3 collect_iieg.py``. Requires curl and
Poppler's pdfinfo/pdftotext, but no Python packages beyond the standard library.
"""

import csv
import hashlib
import re
import subprocess
import tempfile
import unicodedata
import zlib
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpora" / "iieg_diverso_2023_2025"
MAX_PDF_BYTES = 30_000_000
HOST = "iieg.gob.mx"


@dataclass(frozen=True)
class Source:
    id: str
    area: str
    series: str
    published: str
    page_id: int
    needle: str
    local_pdf: str = ""


# The four municipal books are an intentional small sample of the 124 already
# present in input/. Other areas are expanded from their official listings.
SOURCES = (
    Source("expectativas_2024", "economia", "expectativas", "2024-09", 55, "Expectativas-2S-2024.pdf"),
    Source("sector_tic_2023", "economia", "sector_tic", "2023-09", 55, "Estudio-Sector-TIC-2023-20230925.pdf"),
    Source("proveeduria_automotriz_2024", "economia", "proveeduria_automotriz", "2024-06", 55, "Proveeduria-Cluster-Automotriz.pdf"),
    Source("boletin_economico_2025", "economia", "boletin_economico", "2025-09", 49, "boletin_economico_jalisco_IIEG_agosto_2025.pdf"),
    Source("estructura_demografica_2024", "sociedad", "estructura_demografica", "2024-08", 41899, "EstructuraDemográficaPoblación2024.pdf"),
    Source("endutih_2024", "sociedad", "endutih", "2025-07", 902, "ficha_principales-_resutlados_ENDUTIH_jalisco_2024.pdf"),
    Source("trayectorias_docentes_2024", "sociedad", "trayectorias_docentes", "2024-05", 12219, "Resultados_EncuestaTrayectoriasDocentes.pdf"),
    Source("inclusion_financiera_2024", "sociedad", "inclusion_financiera", "2024-11", 55, "inclusión-financiera-2024.pdf"),
    Source("ordenamientos_2024", "geografia_ambiente", "ordenamientos", "2024-02", 148, "Ordenamientos.pdf"),
    Source("accesibilidad_2024", "geografia_ambiente", "accesibilidad", "2024-11", 176, "Índice-de-Accesibilidad-Geográfica-Potencial.pdf"),
    Source("manual_accesibilidad_2024", "geografia_ambiente", "manual_accesibilidad", "2024-11", 176, "Manual-para-análisis-de-accesibilidad-a-los-centros-de-salud.pdf"),
    Source("analisis_sequia_2024", "geografia_ambiente", "sequia_nddi", "2024-06", 152, "Análisis-de-la-sequía-en-Jalisco-por-medio-del-cálculo-del-Índice-de-Sequía-de-Diferencia-Normalizada-NDDI-2015-–-2020.pdf"),
    Source("encig_2023", "gobierno_seguridad", "encig", "2024-03", 25185, "ENCIG_2023.pdf"),
    Source("envipe_2024", "gobierno_seguridad", "envipe", "2024-09", 25243, "Jalisco_Reporte-de-victimización_2024.pdf"),
    Source("cnge_2024", "gobierno_seguridad", "cnge", "2024-12", 25251, "CNGE-2024.pdf"),
    Source("derechos_humanos_2024", "gobierno_seguridad", "censo_derechos_humanos", "2025-01", 25263, "censo_nacional_derechos_humanos_estatal_jalisco2024.pdf"),
    Source("guadalajara_2025", "municipal", "cuadernillos_2025", "2025-11", 25306, "cuadernillos_municipales_guadalajara_2025.pdf", "cuadernillos_municipales_guadalajara_2025.pdf"),
    Source("puerto_vallarta_2025", "municipal", "cuadernillos_2025", "2025-11", 25306, "cuadernillos_municipales_puerto_vallarta_2025.pdf", "cuadernillos_municipales_puerto_vallarta_2025.pdf"),
    Source("mezquitic_2025", "municipal", "cuadernillos_2025", "2025-11", 25306, "cuadernillos_municipales_mezquitic2025.pdf", "cuadernillos_municipales_mezquitic2025.pdf"),
    Source("lagos_de_moreno_2025", "municipal", "cuadernillos_2025", "2025-11", 25306, "cuadernillos_municipales_lagos_de_moreno_2025.pdf", "cuadernillos_municipales_lagos_de_moreno_2025.pdf"),
)


AREA_PAGES = {
    "sociedad": (881, 4252, 914, 12219, 902, 41899),
    "economia": (49, 52, 281, 284, 316, 287, 290, 11696, 293, 11592, 27002, 55),
    "geografia_ambiente": (126, 129, 132, 9564, 15700, 148, 152, 9391, 176, 188, 23388, 39394),
    "gobierno_seguridad": (25180, 25185, 25190, 25195, 25200, 17448, 25213, 31143,
                            25219, 25224, 25228, 33711, 25233, 25237, 43383, 25247,
                            25256, 21934, 25267, 25271, 25302, 32382, 23571, 39726,
                            25275, 27842, 33200, 36588, 29967, 25284, 31967, 25280,
                            25290, 37216),
}

# Recurrent bulletins/fiches are sampled across years; short catalogues are
# collected in full. All documents from one listing share a split group.
PAGE_LIMITS = {49: 9, 281: 9, 290: 9, 293: 9, 316: 6, 287: 6, 31143: 12, 55: 24}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = ""
        self.label = ""
        self.items = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href", "")
            self.label = ""

    def handle_data(self, data):
        if self.href:
            self.label += data

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.items.append((" ".join(self.label.split()), self.href))
            self.href = ""


def discover_pdf(html: bytes, needle: str, page_url: str) -> tuple[str, str]:
    links = Links()
    links.feed(html.decode("utf-8", errors="replace"))
    matches = {}
    for title, href in links.items:
        url = urljoin(page_url, href)
        parsed = urlsplit(url)
        path = unicodedata.normalize("NFKC", unquote(parsed.path)).casefold()
        if parsed.scheme == "https" and parsed.hostname == HOST and path.endswith(".pdf") and unicodedata.normalize("NFKC", needle).casefold() in path:
            matches[quote(url, safe=":/?&=%")] = title
    if len(matches) != 1:
        raise ValueError(f"Expected one official PDF for {needle!r}, found {len(matches)}")
    url, title = next(iter(matches.items()))
    return title, url


def expanded_sources(pages: dict[int, bytes]) -> tuple[Source, ...]:
    existing = {
        unicodedata.normalize("NFKC", unquote(urlsplit(discover_pdf(
            pages[source.page_id], source.needle,
            f"https://{HOST}/ns/?page_id={source.page_id}")[1]).path)).casefold()
        for source in SOURCES
    }
    selected = []
    seen = set(existing)
    for area, page_ids in AREA_PAGES.items():
        for page_id in page_ids:
            links = Links()
            links.feed(pages[page_id].decode("utf-8", errors="replace"))
            candidates = {}
            for title, href in links.items:
                url = urljoin(f"https://{HOST}/ns/?page_id={page_id}", href)
                parsed = urlsplit(url)
                path = unicodedata.normalize("NFKC", unquote(parsed.path))
                match = re.search(r"/uploads/(2023|2024|2025)/(0[1-9]|1[0-2])/.+\.pdf$", path, re.IGNORECASE)
                if (parsed.scheme != "https" or parsed.hostname != HOST or not match
                        or "privacidad" in path.casefold()):
                    continue
                key = path.casefold()
                if key not in seen:
                    candidates[key] = (title, url, f"{match.group(1)}-{match.group(2)}", path)
            cap = PAGE_LIMITS.get(page_id, len(candidates))
            # Round robin by publication year avoids selecting only the latest
            # edition from long monthly series.
            by_year = {year: [] for year in ("2025", "2024", "2023")}
            for item in candidates.values():
                by_year[item[2][:4]].append(item)
            for items in by_year.values():
                items.sort(key=lambda item: (item[2], item[3]), reverse=True)
            chosen = []
            active_years = [year for year, items in by_year.items() if items]
            for year in active_years:
                items = by_year[year]
                quota = min(len(items), (cap + len(active_years) - 1) // len(active_years))
                chosen.extend(items[(2 * index + 1) * len(items) // (2 * quota)]
                              for index in range(quota))
            if len(chosen) < cap:
                chosen_paths = {item[3] for item in chosen}
                remainder = [item for items in by_year.values() for item in items
                             if item[3] not in chosen_paths]
                chosen.extend(remainder[:cap - len(chosen)])
            chosen = chosen[:cap]
            for title, url, published, path in chosen:
                key = path.casefold()
                if key in seen:
                    continue
                seen.add(key)
                stem = unicodedata.normalize("NFKD", Path(path).stem).encode("ascii", "ignore").decode()
                slug = re.sub(r"[^a-z0-9]+", "_", stem.casefold()).strip("_")[:48]
                identifier = f"{slug}_{hashlib.sha256(url.encode()).hexdigest()[:8]}"
                selected.append(Source(identifier, area, f"listing_{page_id}", published,
                                       page_id, path))
    return tuple(selected)


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part") as tmp:
        result = subprocess.run(
            ["curl", "--fail", "--silent", "--show-error", "--location", "--max-redirs", "2",
             "--proto-redir", "=https", "--max-time", "90", "--max-filesize", str(MAX_PDF_BYTES),
             "--write-out", "%{url_effective}", "--output", tmp.name, url],
            check=True, capture_output=True, text=True,
        )
        if urlsplit(result.stdout).hostname != HOST:
            raise ValueError(f"Redirected outside official site: {result.stdout}")
        if Path(tmp.name).stat().st_size > MAX_PDF_BYTES:
            raise ValueError(f"File too large: {url}")
        if not Path(tmp.name).read_bytes().startswith(b"%PDF-") and target.suffix == ".pdf":
            raise ValueError(f"Not a PDF: {url}")
        target.write_bytes(Path(tmp.name).read_bytes())


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def paragraphs(text: str) -> list[str]:
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\f", "\n\n")):
        passage = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if len(passage) < 200 or len(re.findall(r"[^\W\d_]{2,}", passage)) < 30:
            continue
        if sum(c.isalpha() for c in passage) / len(passage) < 0.45:
            continue
        out.append(passage)
    return out


def signature(text: str) -> set[int]:
    words = normalize(text).split()
    return {zlib.crc32(" ".join(words[i:i + 5]).encode()) for i in range(len(words) - 4)}


def deduplicate(by_source: dict[str, list[str]]) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    kept = {source: [] for source in by_source}
    reports = []
    exact = {}
    buckets = defaultdict(set)
    accepted = []
    for source, passages in by_source.items():
        for passage in passages:
            key = normalize(passage)
            digest = hashlib.sha256(key.encode()).hexdigest()
            if key in exact:
                reports.append({"source": source, "kept_source": exact[key], "kind": "exact", "similarity": "1.000", "sha256": digest})
                continue
            shingles = signature(passage)
            candidates = set().union(*(buckets[h] for h in sorted(shingles)[:8])) if shingles else set()
            near = False
            for index in sorted(candidates):
                previous, owner = accepted[index]
                if owner == source or min(len(key), len(previous)) / max(len(key), len(previous)) < 0.85:
                    continue
                score = SequenceMatcher(None, key, previous, autojunk=False).ratio()
                if score >= 0.94:
                    reports.append({"source": source, "kept_source": owner, "kind": "near", "similarity": f"{score:.3f}", "sha256": digest})
                    near = True
                    break
            if near:
                continue
            index = len(accepted)
            accepted.append((key, source))
            for h in sorted(shingles)[:8]:
                buckets[h].add(index)
            exact[key] = source
            kept[source].append(passage)
    return kept, reports


def extract_pdf(path: Path) -> tuple[int, str]:
    info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True).stdout
    match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot determine pages: {path}")
    content = subprocess.run(["pdftotext", "-layout", "-nopgbrk", str(path), "-"], capture_output=True, check=True).stdout.decode("utf-8", errors="replace")
    if len(content.strip()) < 1000:
        raise ValueError(f"Too little extractable text: {path}")
    return int(match.group(1)), content


def skip_reason(error: Exception) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        if error.cmd[0] == "curl" and error.returncode == 63:
            return "PDF exceeds 30 MB download limit"
        return f"{error.cmd[0]} failed (exit {error.returncode})"
    if isinstance(error, ValueError) and str(error).startswith("Too little extractable text"):
        return "too little extractable text"
    return str(error)


def main() -> None:
    pdf_dir = CORPUS / "pdf"
    md_dir = CORPUS / "md"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    pages = {}
    all_page_ids = {source.page_id for source in SOURCES}
    all_page_ids.update(page_id for ids in AREA_PAGES.values() for page_id in ids)
    for page_id in sorted(all_page_ids):
        page_url = f"https://{HOST}/ns/?page_id={page_id}"
        page_path = CORPUS / f"page_{page_id}.html"
        if not page_path.exists():
            fetch(page_url, page_path)
        pages[page_id] = page_path.read_bytes()

    sources = SOURCES + expanded_sources(pages)

    manifest = []
    raw_passages = {}
    skipped = []
    prior_skips = CORPUS / "skipped.csv"
    oversized = set()
    if prior_skips.exists():
        with prior_skips.open(newline="", encoding="utf-8") as existing:
            oversized = {row["id"] for row in csv.DictReader(existing)
                         if "exit status 63" in row["reason"] or "exceeds 30 MB" in row["reason"]}
    for source in sources:
        if source.id in oversized:
            skipped.append({"id": source.id, "area": source.area,
                            "reason": "PDF exceeds 30 MB download limit"})
            continue
        page_url = f"https://{HOST}/ns/?page_id={source.page_id}"
        try:
            title, url = discover_pdf(pages[source.page_id], source.needle, page_url)
        except ValueError as error:
            if source in SOURCES:
                raise
            skipped.append({"id": source.id, "area": source.area, "reason": str(error)})
            print(f"SKIP {source.id}: {error}", flush=True)
            continue
        path = pdf_dir / f"{source.id}.pdf"
        try:
            if not path.exists():
                if source.local_pdf:
                    path.write_bytes((ROOT / "input" / source.local_pdf).read_bytes())
                else:
                    fetch(url, path)
            pages_count, extracted = extract_pdf(path)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            if source in SOURCES:
                raise
            reason = skip_reason(error)
            skipped.append({"id": source.id, "area": source.area, "reason": reason})
            print(f"SKIP {source.id}: {reason}", flush=True)
            continue
        passages = paragraphs(extracted)
        if len(passages) < 3:
            skipped.append({"id": source.id, "area": source.area, "reason": "fewer than 3 useful passages"})
            print(f"SKIP {source.id}: too few useful passages", flush=True)
            continue
        raw_passages[source.id] = passages
        manifest.append({
            "id": source.id, "area": source.area, "series": source.series,
            "published": source.published, "title": title, "listing_url": page_url,
            "url": url, "pdf": str(path.relative_to(ROOT)),
            "md": str((md_dir / f"{source.id}.md").relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pages": pages_count, "source_group": f"listing_{source.page_id}" if source.area != "municipal" else source.series,
            "raw_passages": len(passages),
        })
        print(f"{source.id}: {pages_count} pages, {len(passages)} passages", flush=True)

    kept, reports = deduplicate(raw_passages)
    previous_manifest = CORPUS / "manifest.csv"
    previous_md = set()
    if previous_manifest.exists():
        with previous_manifest.open(newline="", encoding="utf-8") as existing:
            previous_md = {Path(row["md"]).name for row in csv.DictReader(existing)}
    for row in manifest:
        passages = kept[row["id"]]
        row["kept_passages"] = len(passages)
        if passages:
            (ROOT / row["md"]).write_text("\n\n".join(passages) + "\n", encoding="utf-8")
        else:
            skipped.append({"id": row["id"], "area": row["area"],
                            "reason": "all passages duplicate other documents"})
    manifest = [row for row in manifest if row["kept_passages"]]
    current_md = {Path(row["md"]).name for row in manifest}
    for name in previous_md - current_md:
        (md_dir / name).unlink(missing_ok=True)
    with (CORPUS / "manifest.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (CORPUS / "similarity.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["source", "kept_source", "kind", "similarity", "sha256"])
        writer.writeheader()
        writer.writerows(reports)
    with (CORPUS / "skipped.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["id", "area", "reason"])
        writer.writeheader()
        writer.writerows(skipped)
    print(f"Corpus: {len(manifest)} PDFs, {sum(map(len, kept.values()))} kept passages; {len(reports)} similarities; {len(skipped)} skipped")


if __name__ == "__main__":
    main()
