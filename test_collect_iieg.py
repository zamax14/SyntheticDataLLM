import unittest

from collect_iieg import discover_pdf, deduplicate


class CollectIIEGTest(unittest.TestCase):
    def test_discovers_only_the_named_pdf(self):
        html = b'<a href="/ns/file.pdf">Estudio</a><a href="/ns/other.pdf">Otro</a>'
        title, url = discover_pdf(html, "file.pdf", "https://iieg.gob.mx/ns/")
        self.assertEqual(title, "Estudio")
        self.assertEqual(url, "https://iieg.gob.mx/ns/file.pdf")

    def test_removes_exact_and_near_duplicate_passages(self):
        base = "El instituto registró datos de población para Jalisco en 2024 y comparó los resultados por municipio. " * 4
        near = base.replace("2024", "2025")
        kept, removed = deduplicate({"uno": [base], "dos": [base, near]})
        self.assertEqual(sum(map(len, kept.values())), 1)
        self.assertEqual([r["kind"] for r in removed], ["exact", "near"])


if __name__ == "__main__":
    unittest.main()
