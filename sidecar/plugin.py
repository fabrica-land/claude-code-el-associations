"""Association enrichment sidecar plugin for el-sidecar.

Enriches incoming events with associative memory retrieval:
    Stage 1: Keyword extraction + semantic index search (sub-10ms)
    Stage 2: LLM filter for relevance judgment (optional, ~1-4s)

The search module is agent-specific — each agent provides their own
implementation via the ASSOCIATION_SEARCH_MODULE env var. The module
must export a `search_associations(text, top_k=N)` function that returns:
    {"associations": [...], "metrics": {"total_ms": N}}

Registers an enrichment hook — runs on every event during insertion.

Env vars:
    ASSOCIATION_SEARCH_MODULE  — Path to search module (required)
    ASSOCIATION_FILTER_MODULE  — Path to LLM filter module (optional)
    ANTHROPIC_API_KEY          — Enables LLM filter (optional)
"""

import importlib.util
import json
import os
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEARCH_MODULE_PATH = os.environ.get('ASSOCIATION_SEARCH_MODULE', '')
FILTER_MODULE_PATH = os.environ.get('ASSOCIATION_FILTER_MODULE', '')

# Lazy-loaded modules
_search_module = None
_filter_module = None


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------


def _get_search_module():
    global _search_module
    if _search_module is None:
        if not SEARCH_MODULE_PATH:
            return None
        path = os.path.expanduser(SEARCH_MODULE_PATH)
        if not os.path.isfile(path):
            sys.stderr.write(f"[el-associations] Search module not found: {path}\n")
            return None
        spec = importlib.util.spec_from_file_location("association_search", path)
        _search_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_search_module)
    return _search_module


def _get_filter_module():
    global _filter_module
    if _filter_module is None:
        if not FILTER_MODULE_PATH:
            return None
        path = os.path.expanduser(FILTER_MODULE_PATH)
        if not os.path.isfile(path):
            sys.stderr.write(f"[el-associations] Filter module not found: {path}\n")
            return None
        spec = importlib.util.spec_from_file_location("association_filter", path)
        _filter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_filter_module)
    return _filter_module


# ---------------------------------------------------------------------------
# Enrichment hook
# ---------------------------------------------------------------------------


def enrich(text, source='', **kwargs):
    """Enrich event text with associative memory hits.

    Returns a JSON string for the associations column, or None.
    """
    if not text or len(text) <= 10:
        return None

    search = _get_search_module()
    if search is None:
        return None

    try:
        result = search.search_associations(text, top_k=15)
        assocs = result.get("associations", [])
        kw_ms = result.get("metrics", {}).get("total_ms", "?")

        if not assocs:
            return None

        # Stage 2: LLM filter (optional)
        fm = _get_filter_module()
        if fm is not None:
            try:
                # Filter module must have filter_associations(raw_associations, event_text)
                # and optionally _get_api_key() to check for API key presence
                has_key = True
                if hasattr(fm, '_get_api_key'):
                    has_key = bool(fm._get_api_key())

                if has_key:
                    filtered = fm.filter_associations(
                        raw_associations=assocs,
                        event_text=text,
                    )
                    if filtered and filtered.get("filtered"):
                        assocs = filtered["filtered"]
                        filter_ms = filtered.get("metrics", {}).get("total_ms", "?")
                        sys.stderr.write(
                            f"[el-associations] {len(assocs)} associations "
                            f"(filtered from {filtered['raw_count']}) "
                            f"kw:{kw_ms}ms + llm:{filter_ms}ms\n"
                        )
                    else:
                        assocs = assocs[:5]
                        sys.stderr.write(
                            f"[el-associations] {len(assocs)} associations "
                            f"(keyword only) in {kw_ms}ms\n"
                        )
                else:
                    assocs = assocs[:5]
                    sys.stderr.write(
                        f"[el-associations] {len(assocs)} associations "
                        f"(keyword only, no API key) in {kw_ms}ms\n"
                    )
            except Exception as fe:
                assocs = assocs[:5]
                sys.stderr.write(
                    f"[el-associations] filter error ({fe}), "
                    f"keyword fallback: {len(assocs)} in {kw_ms}ms\n"
                )
        else:
            assocs = assocs[:5]
            sys.stderr.write(
                f"[el-associations] {len(assocs)} associations "
                f"(keyword only, no filter module) in {kw_ms}ms\n"
            )

        # Compact for storage
        compact = [
            {"source": a["source"], "type": a["type"],
             "summary": a.get("summary", "")[:150],
             "score": round(a.get("normalized_score", 0), 2),
             **({"reason": a["sonnet_reason"]} if a.get("sonnet_reason") else {})}
            for a in assocs
        ]
        return json.dumps(compact)

    except Exception as e:
        sys.stderr.write(f"[el-associations] enrichment error: {e}\n")
        return None


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(sidecar):
    """Register association enrichment with el-sidecar."""
    if not SEARCH_MODULE_PATH:
        sys.stderr.write(
            "[el-associations] ASSOCIATION_SEARCH_MODULE not set — "
            "enrichment disabled. Set this env var to your agent's "
            "search module path.\n"
        )
        return

    sidecar['register_enrichment']('associations', enrich)
    sys.stderr.write(
        f"[el-associations] Registered enrichment hook "
        f"(search={SEARCH_MODULE_PATH}"
        f"{', filter=' + FILTER_MODULE_PATH if FILTER_MODULE_PATH else ''})\n"
    )
