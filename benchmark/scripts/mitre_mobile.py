"""Download MITRE ATT&CK Mobile-matrix data, build a document corpus from it, and
ingest it into Qdrant.

AttackQA (benchmark/scripts/attackqa.py) supplies our main corpus but was built
only from ATT&CK's Enterprise matrix, so it has zero coverage of Mobile-matrix
software and techniques -- see the CTI-ATE analysis in report/main.tex
(sec:main-result): 14/60 CTI-ATE questions ask about Mobile-platform malware
whose technique relationships the corpus simply never had. This script closes
that gap directly from MITRE's own STIX 2.1 bundle, in the same two document
shapes AttackQA already uses so retrieval treats both corpora uniformly:

- techniques/: one doc per (non-deprecated) Mobile attack-pattern, its name +
  description -- mirrors AttackQA's `techniques` source.
- relationships_techniques_for_software/: one doc per Mobile malware/tool that
  has at least one non-revoked "uses" relationship to a technique, a single
  sentence listing every technique it uses -- mirrors AttackQA's
  `relationships_techniques_for_software` source.

Pulls the full Mobile matrix, not just the 14 gap IDs CTI-ATE happens to ask
about -- patching only the benchmark's own questions would make the corpus fit
the eval set instead of the domain.

Run with: uv run python -m benchmark.scripts.mitre_mobile
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from rag.ingest import main as ingest


STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/mobile-attack/mobile-attack.json"
RAW_PATH = Path("data/raw/mitre/mobile-attack.json")
DOCUMENTS_DIR = Path("data/documents/mitre_mobile")

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)")
_CITATION = re.compile(r"\s*\(Citation:[^)]*\)")


def _clean_description(text: str) -> str:
    """Strip STIX markdown links and inline citation markers down to plain prose."""
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _CITATION.sub("", text)
    return text.strip()


def _mitre_id(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def download(force: bool = False) -> None:
    """Download the Mobile ATT&CK STIX bundle and cache it locally, unless already cached."""
    if RAW_PATH.exists() and not force:
        print(f"Using cached bundle at {RAW_PATH}")
        return

    response = httpx.get(STIX_URL, timeout=60, follow_redirects=True)
    response.raise_for_status()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_bytes(response.content)
    print(f"Downloaded Mobile ATT&CK STIX bundle to {RAW_PATH}")


def _load_bundle() -> tuple[dict, dict, dict]:
    """Parse the cached bundle into (software_by_id, techniques_by_id, uses[software_id] -> [technique_id])."""
    objects = json.loads(RAW_PATH.read_text(encoding="utf-8"))["objects"]
    by_stix_id = {obj["id"]: obj for obj in objects}

    software = {
        _mitre_id(obj): obj
        for obj in objects
        if obj["type"] in ("malware", "tool")
        and not obj.get("revoked") and not obj.get("x_mitre_deprecated")
        and _mitre_id(obj)
    }
    techniques = {
        _mitre_id(obj): obj
        for obj in objects
        if obj["type"] == "attack-pattern"
        and not obj.get("revoked") and not obj.get("x_mitre_deprecated")
        and _mitre_id(obj)
    }

    software_stix_ids = {obj["id"] for obj in software.values()}
    technique_stix_ids = {obj["id"] for obj in techniques.values()}

    uses: dict[str, list[str]] = defaultdict(list)
    for obj in objects:
        if (
            obj["type"] == "relationship" and obj.get("relationship_type") == "uses"
            and not obj.get("revoked")
            and obj.get("source_ref") in software_stix_ids
            and obj.get("target_ref") in technique_stix_ids
        ):
            source_id = _mitre_id(by_stix_id[obj["source_ref"]])
            target_id = _mitre_id(by_stix_id[obj["target_ref"]])
            if target_id not in uses[source_id]:
                uses[source_id].append(target_id)

    return software, techniques, uses


def iter_technique_documents(techniques: dict) -> Iterator[tuple[str, Path, dict]]:
    """Yield (document text, relative path, sidecar metadata) per Mobile technique -- mirrors
    AttackQA's `techniques` source shape (see data/documents/attackqa/techniques/*.txt)."""
    for mitre_id, obj in sorted(techniques.items()):
        description = _clean_description(obj["description"])
        document = f"Description of attack technique '{mitre_id}: {obj['name']}':\n{description}"
        metadata = {
            "url": f"https://attack.mitre.org/techniques/{mitre_id}",
            "subject_id": mitre_id,
            "subject_name": obj["name"],
            "subject_type": "techniques",
            "source_category": "techniques",
            "platform": "Mobile",
        }
        yield document, Path("techniques") / f"techniques__{mitre_id}__001.txt", metadata


def iter_relationship_documents(software: dict, techniques: dict, uses: dict) -> Iterator[tuple[str, Path, dict]]:
    """Yield (document text, relative path, sidecar metadata) per Mobile software-uses-technique
    relation -- mirrors AttackQA's `relationships_techniques_for_software` source shape (see
    data/documents/attackqa/relationships_techniques_for_software/*.txt)."""
    for mitre_id, obj in sorted(software.items()):
        technique_ids = uses.get(mitre_id)
        if not technique_ids:
            continue
        labels = [f"'{tid}: {techniques[tid]['name']}'" for tid in technique_ids if tid in techniques]
        if not labels:
            continue
        document = f"The attack techniques used by software '{mitre_id}: {obj['name']}' are: {', '.join(labels)}"
        metadata = {
            "url": f"https://attack.mitre.org/software/{mitre_id}",
            "subject_id": mitre_id,
            "subject_name": obj["name"],
            "subject_type": "software",
            "source_category": "relationships_techniques_for_software",
            "platform": "Mobile",
        }
        yield document, Path("relationships_techniques_for_software") / f"relationships_techniques_for_software__{mitre_id}__001.txt", metadata


def build_documents() -> None:
    """Write one document file plus metadata sidecar per Mobile technique and software-technique relation."""
    software, techniques, uses = _load_bundle()

    written = Counter()
    for document, relative_path, metadata in (
        list(iter_technique_documents(techniques)) + list(iter_relationship_documents(software, techniques, uses))
    ):
        text_path = DOCUMENTS_DIR / relative_path
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(document + "\n", encoding="utf-8")

        sidecar_path = text_path.with_name(text_path.name + ".meta.json")
        sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        written[metadata["source_category"]] += 1

    print(f"Wrote {sum(written.values())} documents to {DOCUMENTS_DIR}: {dict(written)}")
    print(f"({len(software)} Mobile software, {len(techniques)} Mobile techniques, "
          f"{sum(1 for v in uses.values() if v)} software entries with >=1 technique relation)")


def load(*, collection: str = "cyberqa_documents", force_download: bool = False) -> int:
    """Download, build, and ingest Mobile ATT&CK data; return rag.ingest's exit status."""
    download(force=force_download)
    build_documents()
    return ingest([str(DOCUMENTS_DIR), "--collection", collection])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download MITRE ATT&CK Mobile data, build documents, and ingest them into Qdrant.")
    parser.add_argument("--collection", default="cyberqa_documents", help="Qdrant collection name")
    parser.add_argument("--force-download", action="store_true", help="Re-download the bundle even if cached")
    args = parser.parse_args(argv)
    return load(collection=args.collection, force_download=args.force_download)


if __name__ == "__main__":
    raise SystemExit(main())
