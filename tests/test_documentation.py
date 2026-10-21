from pathlib import Path
import re
import unittest


class DocumentationTests(unittest.TestCase):
    def test_local_links_and_images_resolve(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / "README.md", root / "README_CN.md", *(root / "docs").glob("*.md")):
            text = path.read_text(encoding="utf-8")
            links = re.findall(r"\]\(([^\s)]+)\)", text)
            links.extend(re.findall(r'<img[^>]+src="([^"]+)"', text))
            for link in links:
                if link.startswith(("https://", "http://", "#", "mailto:")):
                    continue
                target = path.parent / link.split("#", 1)[0]
                with self.subTest(document=path.name, target=link):
                    self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
