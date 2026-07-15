"""Fetch Middle-earth place names from Tolkien Gateway.

The site blocks plain page fetches, but it runs MediaWiki, whose API happily
serves the raw wikitext of the A-Z locations index when we send a polite
User-Agent. We ask for one page -- ``Index:Locations`` -- and pull the display
text of every ``*[[link]]`` bullet out of it.

Each index bullet looks like one of:

    *[[Abyss]]
    *[[Tarlang's Neck|Achad Tarlang]] (Tarlang's Neck)
    *[[Ailin (lake in Valinor)]]
    *[[Pass of Aglon|Aglon]], Pass of

We keep the link's *display* text (the part after ``|`` when present) plus any
parenthetical that immediately follows the link, and drop trailing prose such as
", Pass of". No normalising happens here -- that is clean_data.py's job -- so
this script simply writes the raw extracted strings, one per line.

Run:  python src/scrape_locations.py
Out:  data/locations.txt
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# Repo root is the parent of this src/ directory; data/ lives beside it.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "locations.txt"

API_URL = "https://tolkiengateway.net/w/api.php"
PAGE = "Index:Locations"
# A descriptive User-Agent is required; without one the server returns 403.
USER_AGENT = "eldamar-tokenizer/0.1 (Middle-earth name dataset; educational use)"

# A wikilink: [[target]] or [[target|display]]. We want the display text.
LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
# A parenthetical sitting right after the closing ]] , e.g. "]] (Mount Doom)".
TRAILING_PAREN_RE = re.compile(r"\]\]\s*\(([^)]+)\)")


def fetch_wikitext() -> str:
    """Return the raw wikitext of the locations index page via the MediaWiki API."""
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
        payload = json.load(response)
    return payload["parse"]["wikitext"]


def extract_location_strings(wikitext: str) -> list[str]:
    """Pull one raw location string from every bullet-list line in the wikitext."""
    results: list[str] = []
    for raw_line in wikitext.splitlines():
        line = raw_line.strip()
        # Only the bullet lines (``*[[...]]``) name a location; skip headers,
        # table markup (|, !, {|) and blank lines.
        if not line.startswith("*"):
            continue

        link_match = LINK_RE.search(line)
        if not link_match:
            continue

        # Prefer the display half of [[target|display]]; fall back to the target.
        link_body = link_match.group(1)
        name = link_body.split("|")[-1].strip()
        if not name:
            continue

        # Keep an alternative name given in a trailing parenthetical, so the
        # cleaner can later mine it (e.g. "Amon Amarth (Mount Doom)").
        paren_match = TRAILING_PAREN_RE.search(line)
        if paren_match:
            name = f"{name} ({paren_match.group(1).strip()})"

        results.append(name)
    return results


def main() -> None:
    print(f"Fetching '{PAGE}' from Tolkien Gateway ...")
    try:
        wikitext = fetch_wikitext()
    except Exception as error:  # noqa: BLE001 -- surface any network/API failure plainly
        sys.exit(f"Failed to fetch the locations index: {error}")

    names = extract_location_strings(wikitext)
    if not names:
        sys.exit("No location names were extracted -- the page format may have changed.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Wrote {len(names)} raw location strings to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
