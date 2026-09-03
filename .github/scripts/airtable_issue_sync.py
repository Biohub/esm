#!/usr/bin/env python3
"""Sync a GitHub issue into an Airtable base.

Env: AIRTABLE_TOKEN, AIRTABLE_BASE, AIRTABLE_TABLE, and (from GitHub Actions)
GITHUB_EVENT_PATH. Optional: LABEL_TYPE_MAP, PRODUCT_KEYWORD_MAP,
DEFAULT_PRODUCTS, DRY_RUN.

Everything written comes out of the webhook payload, apart from the "Automated"
tag on "Issue Source" that marks the row as machine-written; the script makes no
calls to GitHub.

An issue is matched to an existing row by URL and updated in place, otherwise a
row is created. Fields that already hold a value are left as they are.

The Airtable token needs schema.bases:read, data.records:read and
data.records:write.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.airtable.com/v0"

F_TYPE = "Type of Request"
F_DATE = "Submission Date"
F_PRODUCT = "Tool or Product"
F_SOURCE = "Issue Source"
F_TITLE = "Issue Title"
F_DESCRIPTION = "Issue Description"
F_GITHUB_ID = "Github Username"
F_ORIGINAL_Q = "Original Q Location"
F_COMPLETE = "Issue Complete?"
F_DETAILS = "Additional Details"
# The sync key, "<owner>/<repo>#<number>". Held alongside the issue URL so a row
# somebody entered by hand is still matched on the URL alone.
F_REFERENCE = "Issue tracker link"

SOURCE_GITHUB = "Github"
# Stamped on every row this script touches, so rows that arrived by automation
# can be told apart from the ones people enter by hand.
SOURCE_AUTOMATED = "Automated"
STATE_OPEN = "Incomplete"
STATE_CLOSED = "Complete"
COMPLETE_STATES = {"complete", "out of scope"}

# Airtable long text holds 100k characters; stay well under it.
MAX_TEXT_CHARS = 50_000

REQUEST_TIMEOUT_SECONDS = 30

# GitHub label (lowercased) -> "Type of Request" option. Extend with
# LABEL_TYPE_MAP rather than editing this.
DEFAULT_LABEL_TYPE_MAP = {
    "bug": "Bug",
    "defect": "Bug",
    "error": "Error",
    "crash": "Error",
    "enhancement": "Feature Request",
    "feature": "Feature Request",
    "feature request": "Feature Request",
    "question": "Support",
    "support": "Support",
    "help wanted": "Support",
    "documentation": "Support",
    "docs": "Support",
    "feedback": "General Feedback",
    "access": "Access issue",
    "permissions": "Access issue",
}

# Substring (lowercased) -> "Tool or Product" option, scanned over title, body
# and labels.
DEFAULT_PRODUCT_KEYWORD_MAP = {
    "esmfold2": "ESMFold2",
    "esmfold 2": "ESMFold2",
    "esm atlas": "ESM Atlas",
    "metagenomic atlas": "ESM Atlas",
    "biohub platform": "Biohub Platform",
    "esmc": "ESMC",
    "esm3": "ESM3",
    "esm 3": "ESM3",
    "binder": "Binder",
    "sae": "SAE",
}


def require_env(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"Missing required env var: {name}")
    return v


def _lower_keys(raw):
    return {k.lower(): v for k, v in json.loads(raw).items()}


AIRTABLE_TOKEN = require_env("AIRTABLE_TOKEN")
AIRTABLE_BASE = require_env("AIRTABLE_BASE")
AIRTABLE_TABLE = require_env("AIRTABLE_TABLE")
LABEL_TYPE_MAP = {
    **DEFAULT_LABEL_TYPE_MAP,
    **_lower_keys(os.environ.get("LABEL_TYPE_MAP", "{}")),
}
PRODUCT_KEYWORD_MAP = {
    **DEFAULT_PRODUCT_KEYWORD_MAP,
    **_lower_keys(os.environ.get("PRODUCT_KEYWORD_MAP", "{}")),
}
DEFAULT_PRODUCTS = json.loads(os.environ.get("DEFAULT_PRODUCTS", "[]"))
DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

_warned = set()


def warn(msg):
    if msg not in _warned:
        print(f"WARN: {msg}")
        _warned.add(msg)


def _request(req, attempts=4):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 502, 503) and attempt < attempts - 1:
                time.sleep(2**attempt)
                continue
            detail = err.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"{req.get_method()} {req.full_url} -> {err.code}: {detail}"
            )
    raise RuntimeError(f"{req.get_method()} {req.full_url}: retries exhausted")


def airtable(path, method="GET", body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {AIRTABLE_TOKEN}")
    req.add_header("Content-Type", "application/json")
    return _request(req)


def table_path():
    return f"/{AIRTABLE_BASE}/{urllib.parse.quote(AIRTABLE_TABLE, safe='')}"


def get_schema():
    """Field name -> field schema for the target table."""
    tables = airtable(f"/meta/bases/{AIRTABLE_BASE}/tables")["tables"]
    for t in tables:
        if AIRTABLE_TABLE in (t["id"], t["name"]):
            return {f["name"]: f for f in t["fields"]}
    names = ", ".join(f'"{t["name"]}"' for t in tables)
    sys.exit(
        f'Table "{AIRTABLE_TABLE}" not found in base {AIRTABLE_BASE}. Have: {names}'
    )


def truncate(text, url):
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + f"\n\n... (truncated; read the full issue at {url})"


def match_choices(fschema, values):
    """Keep the values that already exist as options, in the schema's casing."""
    options = [c["name"] for c in fschema.get("options", {}).get("choices", [])]
    by_lower = {o.lower(): o for o in options}
    matched = []
    for v in values:
        canonical = by_lower.get(str(v).strip().lower())
        if canonical is None:
            warn(f'"{fschema["name"]}" has no option matching "{v}"; dropping it.')
        elif canonical not in matched:
            matched.append(canonical)
    return matched


def coerce(fschema, value):
    """Value in the shape Airtable wants, or None if it should not be written."""
    t = fschema["type"]
    if t in ("multipleSelects", "singleSelect"):
        values = value if isinstance(value, list) else [value]
        matched = match_choices(fschema, [v for v in values if v])
        if not matched:
            return None
        return matched if t == "multipleSelects" else matched[0]
    if t in ("multilineText", "singleLineText", "richText"):
        return str(value) if value else None
    if t == "date":
        return str(value)[:10] if value else None
    if t == "checkbox":
        return bool(value)
    warn(f'field "{fschema["name"]}" (type {t}) is not written by this sync; skipping.')
    return None


def is_empty(current):
    return current is None or current == "" or current == [] or current is False


class Row:
    """Fields to write, validated against the live schema."""

    def __init__(self, schema, existing):
        self.schema = schema
        self.existing = (existing or {}).get("fields", {})
        self.fields = {}

    def _prepare(self, name, value):
        fschema = self.schema.get(name)
        if not fschema:
            warn(f'field "{name}" not found in the table; skipping.')
            return None
        return coerce(fschema, value)

    def own(self, name, value):
        """Keep this field equal to the GitHub value."""
        prepared = self._prepare(name, value)
        if prepared is not None and self.existing.get(name) != prepared:
            self.fields[name] = prepared

    def fill(self, name, value):
        """Write this field only while it is still empty."""
        if not is_empty(self.existing.get(name)):
            return
        prepared = self._prepare(name, value)
        if prepared is not None:
            self.fields[name] = prepared

    def add(self, name, values):
        """Make sure these values are present without dropping the others.

        ``own`` replaces the cell, which on a multi-select would discard a
        source somebody selected by hand. This sync only needs its own tags to
        be there.
        """
        current = self.existing.get(name) or []
        if not isinstance(current, list):
            current = [current]
        prepared = self._prepare(name, list(current) + list(values))
        if prepared is None:
            return
        wanted = prepared if isinstance(prepared, list) else [prepared]
        if sorted(map(str, wanted)) != sorted(map(str, current)):
            self.fields[name] = prepared


def label_names(issue):
    return [
        str(lb["name"] if isinstance(lb, dict) else lb)
        for lb in issue.get("labels") or []
    ]


def request_types(issue):
    types = []
    for label in label_names(issue):
        mapped = LABEL_TYPE_MAP.get(label.strip().lower())
        if mapped and mapped not in types:
            types.append(mapped)
    return types


def products(issue):
    haystack = "\n".join(
        [
            issue.get("title") or "",
            issue.get("body") or "",
            " ".join(label_names(issue)),
        ]
    ).lower()
    found = []
    # Longest keyword first so "esmfold2" is not shadowed by a shorter match.
    for keyword in sorted(PRODUCT_KEYWORD_MAP, key=len, reverse=True):
        product = PRODUCT_KEYWORD_MAP[keyword]
        if product in found:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", haystack):
            found.append(product)
    return found or list(DEFAULT_PRODUCTS)


def details_block(issue, repo_full_name):
    labels = label_names(issue)
    return "\n".join(
        [
            f"Repository: {repo_full_name}",
            f"Issue: #{issue['number']}",
            f"Title: {issue.get('title') or ''}",
            f"Author: @{(issue.get('user') or {}).get('login', 'unknown')}",
            f"Labels: {', '.join(labels) if labels else '(none)'}",
        ]
    )


def escape_formula(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def find_record(sync_key, url):
    formula = (
        f'OR({{{F_REFERENCE}}}="{escape_formula(sync_key)}",'
        f' {{{F_ORIGINAL_Q}}}="{escape_formula(url)}")'
    )
    query = urllib.parse.urlencode({"filterByFormula": formula, "maxRecords": "10"})
    records = airtable(f"{table_path()}?{query}").get("records", [])
    if len(records) > 1:
        ids = ", ".join(r["id"] for r in records)
        warn(f"{len(records)} rows match {sync_key} ({ids}); updating the first.")
    return records[0] if records else None


def desired_state(action, current):
    """Follow the issue's open/closed state without discarding another value."""
    values = current if isinstance(current, list) else [current] if current else []
    is_complete = any(str(v).strip().lower() in COMPLETE_STATES for v in values)
    if action == "closed":
        return None if is_complete else STATE_CLOSED
    if action == "reopened":
        return STATE_OPEN if (not values or is_complete) else None
    return None


def build_row(schema, existing, issue, repo_full_name, action):
    url = issue["html_url"]
    title = issue.get("title") or f"Issue #{issue['number']}"
    body = (issue.get("body") or "").strip() or "(No description was provided.)"
    login = (issue.get("user") or {}).get("login")

    row = Row(schema, existing)
    row.own(F_ORIGINAL_Q, url)
    row.own(F_REFERENCE, f"{repo_full_name}#{issue['number']}")
    row.add(F_SOURCE, [SOURCE_GITHUB, SOURCE_AUTOMATED])
    row.own(F_GITHUB_ID, login)

    row.fill(F_DATE, issue.get("created_at"))
    row.fill(F_TITLE, title)
    row.fill(F_DESCRIPTION, truncate(body, url))
    row.fill(F_DETAILS, details_block(issue, repo_full_name))
    row.fill(F_TYPE, request_types(issue))
    row.fill(F_PRODUCT, products(issue))

    state = (
        desired_state(action, row.existing.get(F_COMPLETE)) if existing else STATE_OPEN
    )
    if state:
        row.own(F_COMPLETE, [state])

    return row.fields


def main():
    with open(require_env("GITHUB_EVENT_PATH"), encoding="utf-8") as fh:
        event = json.load(fh)
    action = event.get("action")
    issue = event.get("issue")
    if not issue:
        print(f'No issue in payload (action="{action}"); nothing to do.')
        return

    if (issue.get("user") or {}).get("type") == "Bot":
        print(f"Issue #{issue['number']} was opened by a bot; skipping.")
        return

    repo = event.get("repository") or {}
    repo_full_name = repo.get("full_name") or os.environ.get("GITHUB_REPOSITORY", "")
    sync_key = f"{repo_full_name}#{issue['number']}"

    schema = get_schema()
    existing = find_record(sync_key, issue["html_url"])
    fields = build_row(schema, existing, issue, repo_full_name, action)

    if not fields:
        print(f'{sync_key}: nothing to change (action="{action}").')
        return
    if DRY_RUN:
        target = existing["id"] if existing else "(new record)"
        print(f"DRY_RUN {sync_key} -> {target}\n{json.dumps(fields, indent=2)}")
        return

    if existing:
        airtable(f"{table_path()}/{existing['id']}", "PATCH", {"fields": fields})
        print(
            f'Updated {sync_key} as {existing["id"]} (action="{action}"): '
            f"{', '.join(sorted(fields))}."
        )
    else:
        resp = airtable(table_path(), "POST", {"fields": fields})
        print(f'Created {sync_key} as {resp["id"]} (action="{action}").')


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # noqa: BLE001
        sys.exit(f"Sync failed: {err}")
