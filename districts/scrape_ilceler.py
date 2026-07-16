"""Fetch Turkish district (ilçe) names from Turkish Wikipedia.

Separate side task: generate Turkish district names with the ORIGINAL repo's
character tokeniser. This script only produces the raw name list; cleaning is
delegated to the repo's own single_letter_transformers/data/temizle_isimler.py,
exactly as it cleans the Turkish first names.

Source page: "Türkiye'nin ilçeleri" on tr.wikipedia. It holds one big sortable
table; column 1 is the province (il) and column 2 is the district (ilçe),
written as '''[[Target|Display]]'''. We keep the display text of column 2.

Run:  python districts/scrape_ilceler.py
Out:  districts/ilceler_ham.txt   (one raw district name per line)
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "districts" / "ilceler_ham.txt"

API_URL = "https://tr.wikipedia.org/w/api.php"
PAGE = "Türkiye'nin ilçeleri"
USER_AGENT = "eldamar-districts/0.1 (educational; contact erenyanic@protonmail.com)"

LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def fetch_wikitext() -> str:
    """Return the raw wikitext of the districts page via the MediaWiki API.

    The page title carries Turkish letters; urlencode handles the escaping so the
    request never depends on a hand-typed (and easily corrupted) URL.
    """
    params = urllib.parse.urlencode(
        {
            "action": "parse",
            "page": PAGE,
            "prop": "wikitext",
            "format": "json",
            "formatversion": "2",
        }
    )
    request = urllib.request.Request(
        f"{API_URL}?{params}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["parse"]["wikitext"]


def extract_districts(wikitext: str) -> list[str]:
    """Pull the district name (column 2) out of every row of the big table."""
    start = wikitext.find("{|")
    end = wikitext.find("\n|}", start)
    table = wikitext[start:end]

    districts: list[str] = []
    for row in table.split("\n|-"):
        cells: list[str] = []
        is_header = False
        for line in row.splitlines():
            stripped = line.strip()
            if stripped.startswith("!"):        # a header row -- skip the whole row
                is_header = True
                break
            if stripped.startswith("|") and not stripped.startswith("|}"):
                cells.append(stripped[1:].strip())
        if is_header or len(cells) < 2:
            continue

        district_cell = cells[1]
        link = LINK_RE.search(district_cell)
        if link:                                 # '''[[Çukurova, Adana|Çukurova]]''' -> Çukurova
            name = link.group(1).split("|")[-1].strip()
        else:                                    # bare text: drop bold marks / html
            name = re.sub(r"'''|<[^>]+>", "", district_cell).strip()
        name = name.strip("'").strip()
        if name:
            districts.append(name)
    return districts


def main() -> None:
    print(f"Fetching '{PAGE}' from Turkish Wikipedia ...")
    try:
        wikitext = fetch_wikitext()
    except Exception as error:  # noqa: BLE001 -- surface any network/API failure plainly
        sys.exit(f"Failed to fetch the districts page: {error}")

    names = extract_districts(wikitext)
    if not names:
        sys.exit("No district names were extracted -- the table format may have changed.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Wrote {len(names)} raw district names to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
