#!/usr/bin/env python3
"""
LinkedIn Connections Network Visualizer
=========================================

Takes the CSV you export from LinkedIn yourself (Settings & Privacy ->
Data Privacy -> Get a copy of your data -> "Connections") and builds a
similarity graph of your network.

LIMITATIONS
------------------------------
LinkedIn's personal data export only tells you who is connected to *you*.
It does NOT tell you which of your connections are connected to each other
(that data isn't exposed to individual users via export or API). So this
script does not draw your "real" connection graph — no such thing is
available to you without violating LinkedIn's terms.

Instead, it builds a *similarity graph*:
  - Nodes = your connections
  - Edges = drawn between two people who share a company, or who are
    inferred to be in the same industry/role bucket
  - Node color = inferred industry
  - Node clustering = pulled together if they share a company (via
    edge weighting in the layout)

USAGE
-----
    python linkedin_network_graph.py Connections.csv
    python linkedin_network_graph.py Connections.csv --industry-map my_industries.csv
    python linkedin_network_graph.py Connections.csv --output-prefix my_network

Expected LinkedIn export columns (as of 2025/2026 export format):
    First Name, Last Name, URL, Email Address, Company, Position, Connected On

The script is tolerant of:
    - The extra "Notes:" preamble LinkedIn puts at the top of the CSV
    - Missing/blank columns (Email Address is often blank)
    - Encoding issues (falls back from utf-8 to latin-1)
    - Duplicate or malformed rows

OUTPUT
------
A single interactive HTML "constellation" graph (open it in any browser):
    - Connections cluster together ("constellations") when they share a
      company or an inferred industry; contacts with no shared link drift
      as lone background stars
    - Star size = how connected that person is in your network
    - Star color = inferred industry (see the on-page legend)
    - Live search box to find and highlight a specific person
    - Hover any star for their name, company, role, and industry
    - Drag, scroll to zoom, and pan around freely
"""

import argparse
import base64
import json
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import networkx as nx


# --------------------------------------------------------------------------
# Industry inference
# --------------------------------------------------------------------------
# LinkedIn's export does not include an "industry" field, so we infer one
# from keywords in the Position (job title) and Company fields. This is a
# heuristic, not ground truth. You can override/improve it by passing
# --industry-map a CSV with columns: Company,Industry (exact company name
# matches take priority over keyword inference).

INDUSTRY_KEYWORDS = {
    "Technology": [
        "software", "engineer", "developer", "programmer", "data scientist",
        "it ", "tech", "cloud", "devops", "cyber", "ai", "machine learning",
        "product manager", "cto", "sre", "qa engineer", "full stack",
        "front end", "backend", "web developer",
    ],
    "Finance": [
        "finance", "financial", "accountant", "accounting", "investment",
        "banker", "banking", "audit", "tax", "actuary", "treasury",
        "equity", "portfolio", "credit analyst", "cfo",
    ],
    "Sales & Marketing": [
        "sales", "marketing", "account executive", "business development",
        "growth", "brand", "seo", "advertising", "social media",
        "account manager", "customer success",
    ],
    "Healthcare": [
        "nurse", "doctor", "physician", "clinical", "medical", "health",
        "pharma", "dentist", "therapist", "surgeon", "healthcare",
    ],
    "Education": [
        "teacher", "professor", "lecturer", "tutor", "education",
        "principal", "academic", "researcher", "phd student",
    ],
    "Legal": [
        "lawyer", "legal", "attorney", "solicitor", "paralegal",
        "counsel", "barrister",
    ],
    "Engineering & Manufacturing": [
        "mechanical engineer", "civil engineer", "manufacturing",
        "electrical engineer", "industrial", "supply chain", "logistics",
        "operations manager", "plant manager",
    ],
    "Consulting": [
        "consultant", "consulting", "advisory", "strategy",
    ],
    "HR & Recruiting": [
        "human resources", "hr ", "recruiter", "recruiting", "talent",
        "people operations",
    ],
    "Creative & Media": [
        "designer", "creative", "writer", "editor", "journalist",
        "photographer", "video", "content", "artist", "producer",
    ],
    "Executive / Leadership": [
        "ceo", "founder", "co-founder", "president", "managing director",
        "vp ", "vice president", "chief ",
    ],
}

DEFAULT_INDUSTRY = "Other / Unclassified"


def infer_industry(position: str, company: str, override_map: dict) -> str:
    """Infer an industry bucket from position/company text."""
    company_clean = (company or "").strip()

    # 1. Exact override from user-supplied map takes priority
    if company_clean and company_clean.lower() in override_map:
        return override_map[company_clean.lower()]

    text = f"{position or ''} {company or ''}".lower()
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return industry
    return DEFAULT_INDUSTRY


# --------------------------------------------------------------------------
# CSV loading (tolerant of LinkedIn's export quirks)
# --------------------------------------------------------------------------

def load_linkedin_csv(path: Path) -> pd.DataFrame:
    """
    Load a LinkedIn Connections export CSV.

    LinkedIn prepends a few "Notes:" lines before the real header row,
    and sometimes uses latin-1 encoding instead of utf-8. This function
    handles both.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find file: {path}\n"
            f"Make sure you've exported your LinkedIn data and the path is correct."
        )

    # Find the real header row (the one containing "First Name")
    header_row_idx = 0
    encoding_used = "utf-8"
    lines = None

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, errors="strict") as f:
                lines = f.readlines()
            encoding_used = encoding
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if lines is None:
        raise ValueError(
            "Could not decode the CSV file with utf-8 or latin-1 encoding. "
            "The file may be corrupted."
        )

    for i, line in enumerate(lines):
        if "First Name" in line and "Last Name" in line:
            header_row_idx = i
            break
    else:
        raise ValueError(
            "Could not find a header row containing 'First Name' / 'Last Name'. "
            "This doesn't look like a standard LinkedIn Connections export. "
            "Expected columns: First Name, Last Name, URL, Email Address, "
            "Company, Position, Connected On."
        )

    try:
        df = pd.read_csv(
            path,
            skiprows=header_row_idx,
            encoding=encoding_used,
            dtype=str,
            on_bad_lines="skip",
            engine="python",
        )
    except Exception as e:
        raise ValueError(f"Failed to parse CSV after finding header row: {e}")

    if df.empty:
        raise ValueError("The CSV was parsed but contains no connection rows.")

    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    required_cols = {"First Name", "Last Name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Ensure optional columns exist even if absent from this export
    for col in ("Company", "Position", "Connected On", "URL", "Email Address"):
        if col not in df.columns:
            df[col] = ""

    # Clean whitespace / NaNs
    for col in ("First Name", "Last Name", "Company", "Position"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Drop rows with no name at all
    df = df[(df["First Name"] != "") | (df["Last Name"] != "")]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid connection rows found after cleaning.")

    return df


def load_industry_override(path: str | None) -> dict:
    """Load an optional Company,Industry override CSV into a lowercase-keyed dict."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"Warning: industry map file '{path}' not found, ignoring.", file=sys.stderr)
        return {}
    try:
        df = pd.read_csv(p, dtype=str).fillna("")
        cols = {c.strip().lower(): c for c in df.columns}
        if "company" not in cols or "industry" not in cols:
            print(
                "Warning: industry map CSV must have 'Company' and 'Industry' "
                "columns. Ignoring.",
                file=sys.stderr,
            )
            return {}
        return {
            row[cols["company"]].strip().lower(): row[cols["industry"]].strip()
            for _, row in df.iterrows()
            if row[cols["company"]].strip()
        }
    except Exception as e:
        print(f"Warning: could not read industry map ({e}), ignoring.", file=sys.stderr)
        return {}


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def build_similarity_graph(df: pd.DataFrame, override_map: dict) -> nx.Graph:
    """
    Build a graph where:
      - each connection is a node, colored by inferred industry
      - each company with 2+ shared connections gets its own node too —
        employees connect to their company, not to an arbitrarily-chosen
        coworker, so cluster centers are meaningful (the company) rather
        than an accident of row order in the CSV
      - a lighter-weight chain of edges links people in the same inferred
        industry, but only for small-to-medium buckets; very large
        buckets (e.g. hundreds of "Technology" contacts) skip industry
        edges entirely since color already encodes industry and the
        edges would add clutter, not information
    """
    graph = nx.Graph()
    company_groups = defaultdict(list)
    industry_groups = defaultdict(list)

    for idx, row in df.iterrows():
        name = f"{row['First Name']} {row['Last Name']}".strip()
        if not name:
            name = f"Unknown_{idx}"
        # Disambiguate duplicate names so nodes don't collide
        node_id = name
        suffix = 1
        while graph.has_node(node_id):
            suffix += 1
            node_id = f"{name} ({suffix})"

        company = row["Company"].strip() or "Unknown Company"
        position = row["Position"].strip() or "Unknown Role"
        industry = infer_industry(position, company, override_map)

        graph.add_node(
            node_id,
            node_type="person",
            company=company,
            position=position,
            industry=industry,
        )
        company_groups[company].append(node_id)
        industry_groups[industry].append(node_id)

    # Company clustering: instead of picking one employee to act as a hub
    # (arbitrary — just whoever appeared first in the CSV) or a full clique
    # (too dense), each company becomes its own node, and every employee
    # connects to that company node. The center of each cluster is then
    # genuinely meaningful — it's the company itself, not a random person —
    # and nobody's size is artificially inflated by sitting at a hub.
    for company, members in company_groups.items():
        if company == "Unknown Company" or len(members) < 2:
            continue
        company_node_id = f"company::{company}"
        graph.add_node(
            company_node_id,
            node_type="company",
            company=company,
            employee_count=len(members),
        )
        for member in members:
            graph.add_edge(company_node_id, member, weight=3, kind="company")

    # Weaker edges: shared industry, as a simple chain (A-B-C-D, not a
    # clique and not a hub) so no single person becomes structurally
    # central just for sharing an inferred industry with many others.
    # Only applied to buckets small enough that it adds readable signal.
    INDUSTRY_CHAIN_MAX = 40
    for industry, members in industry_groups.items():
        if industry == DEFAULT_INDUSTRY or len(members) < 2:
            continue
        if len(members) > INDUSTRY_CHAIN_MAX:
            continue
        for i in range(len(members) - 1):
            a, b = members[i], members[i + 1]
            if not graph.has_edge(a, b):
                graph.add_edge(a, b, weight=1, kind="industry")

    return graph


# --------------------------------------------------------------------------
# Interactive "constellation" visualization
# --------------------------------------------------------------------------

# Bright, saturated colors chosen to glow against a near-black background
INDUSTRY_COLORS = {
    "Technology": "#5B9BFF",
    "Finance": "#4ED17E",
    "Sales & Marketing": "#FF6B6B",
    "Healthcare": "#B98CFF",
    "Education": "#F4D35E",
    "Legal": "#4FD6E8",
    "Engineering & Manufacturing": "#FF9F40",
    "Consulting": "#E37FE0",
    "HR & Recruiting": "#FF8FC7",
    "Creative & Media": "#D4D45A",
    "Executive / Leadership": "#C9A67E",
    "Other / Unclassified": "#5A6270",
}


def _star_size(degree: int) -> float:
    """Sqrt scaling so hub nodes stand out without dwarfing everything else."""
    return round(4 + math.sqrt(max(degree, 0)) * 4.5, 1)


def load_company_icon(path: str | None) -> str | None:
    """
    Load a company icon image and embed it as a base64 data URI so the
    final HTML stays a single, fully self-contained file (no separate
    image file to keep track of or that could go missing later).
    Returns None if no path was given or the file can't be read, in which
    case company nodes fall back to a plain gold star.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = p.read_bytes()
    except Exception as e:
        print(f"Warning: could not read company icon '{path}' ({e}), using default star instead.", file=sys.stderr)
        return None
    ext = p.suffix.lower().lstrip(".") or "png"
    mime = "image/png" if ext == "png" else f"image/{ext}"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _building_icon_data_uri(fill: str = "#FFD770", stroke: str = "#3a2f0a") -> str:
    """
    A small flat-style office building icon (rectangle body, grid of
    windows, a door), returned as a self-contained base64 SVG data URI.
    Used for company nodes instead of an external icon font/library, so
    the graph stays fully offline-safe and isn't at the mercy of a CDN
    (see the earlier vis-network CDN loading issue).
    """
    windows = "".join(
        f'<rect x="{x}" y="{y}" width="7" height="7" fill="#05070d"/>'
        for y in (16, 27, 38, 49)
        for x in (21, 32, 43)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
        f'<rect x="13" y="8" width="38" height="54" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        f'{windows}'
        f'<rect x="26" y="55" width="12" height="7" fill="#05070d"/>'
        '</svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def build_constellation_html(graph: nx.Graph, output_path: str, df: pd.DataFrame, company_icon: str | None = None):
    """
    Render the graph as a self-contained interactive HTML "constellation":
    connected clusters of people pull together, unconnected contacts drift
    as lone background stars, colored by inferred industry and sized by
    how connected they are. Includes a legend and a live search box.

    company_icon: optional base64 data URI (see load_company_icon) used
    for every company node, shown as a small icon on a gold badge. Falls
    back to the built-in generated building icon if not provided.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print(
            "pyvis is not installed. Install it with: pip install pyvis",
            file=sys.stderr,
        )
        return

    if graph.number_of_nodes() == 0:
        print("Graph is empty, nothing to visualize.", file=sys.stderr)
        return

    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#05070d",
        font_color="#e8ecf5",
        directed=False,
        cdn_resources="in_line",
    )

    # Company nodes are always labeled (there are relatively few of them and
    # each one is a meaningful cluster anchor). Person nodes are only
    # labeled if they're relatively well-connected, to keep the canvas
    # readable — everyone else still shows full detail on hover.
    degrees = dict(graph.degree())
    person_degrees = {n: d for n, d in degrees.items() if graph.nodes[n].get("node_type") == "person"}
    labeled_count = max(15, min(50, int(len(person_degrees) * 0.03)))
    hub_people = {
        n for n, _ in sorted(person_degrees.items(), key=lambda kv: kv[1], reverse=True)[:labeled_count]
        if person_degrees[n] > 0
    }

    for node, attrs in graph.nodes(data=True):
        degree = degrees.get(node, 0)
        node_type = attrs.get("node_type", "person")

        if node_type == "company":
            company_name = attrs.get("company", "Unknown Company")
            employee_count = attrs.get("employee_count", degree)
            gold = "#FFD770"
            tooltip = (
                f"<b>{company_name}</b><br>"
                f"{employee_count} connection{'s' if employee_count != 1 else ''} here"
            )
            if company_icon:
                # Real user-supplied icon: sit it on a gold circular badge
                # so its own artwork (often dark line-art) still reads
                # clearly against the near-black background.
                #
                # The full icon is NOT embedded per-node here — with ~75
                # company nodes that would duplicate the same base64 image
                # 75 times and could bloat the HTML into tens of megabytes.
                # Instead every company node starts with the tiny generated
                # placeholder glyph, and a small script (see
                # _apply_constellation_theme) swaps in the one real icon
                # — embedded exactly once — right after the page loads.
                net.add_node(
                    node,
                    label=company_name,
                    title=tooltip,
                    name=company_name,
                    company=company_name,
                    industry="",
                    icon_pending=True,
                    shape="circularImage",
                    image=_building_icon_data_uri(gold),
                    color={"background": gold, "border": gold, "highlight": {"background": "#ffffff", "border": gold}},
                    borderWidth=2,
                    size=round(14 + math.sqrt(employee_count) * 4, 1),
                    font={"color": "#ffe9b3", "size": 15, "strokeWidth": 3, "strokeColor": "#05070d"},
                    shadow={"enabled": True, "color": gold, "size": 18, "x": 0, "y": 0},
                    physics=False,
                )
            else:
                # No custom icon supplied: fall back to a generated gold
                # building glyph so the graph still looks intentional.
                net.add_node(
                    node,
                    label=company_name,
                    title=tooltip,
                    name=company_name,
                    company=company_name,
                    industry="",
                    shape="image",
                    image=_building_icon_data_uri(gold),
                    size=round(14 + math.sqrt(employee_count) * 4, 1),
                    font={"color": "#ffe9b3", "size": 15, "strokeWidth": 3, "strokeColor": "#05070d"},
                    shadow={"enabled": True, "color": gold, "size": 18, "x": 0, "y": 0},
                )
            continue

        industry = attrs.get("industry", "Other / Unclassified")
        color = INDUSTRY_COLORS.get(industry, INDUSTRY_COLORS["Other / Unclassified"])
        is_hub = node in hub_people

        tooltip = (
            f"<b>{node}</b><br>"
            f"{attrs.get('position', 'Unknown Role')}<br>"
            f"{attrs.get('company', 'Unknown Company')}<br>"
            f"<i>{industry}</i><br>"
            f"{degree} shared-company/industry link{'s' if degree != 1 else ''}"
        )

        net.add_node(
            node,
            label=node if is_hub else " ",
            title=tooltip,
            name=node,
            company=attrs.get("company", ""),
            industry=industry,
            color={"background": color, "border": color, "highlight": {"background": "#ffffff", "border": color}},
            size=_star_size(degree),
            shape="dot",
            borderWidth=0,
            font={"color": "#e8ecf5" if is_hub else "rgba(0,0,0,0)", "size": 13 if is_hub else 0, "strokeWidth": 0},
            shadow={"enabled": True, "color": color, "size": 12 if degree > 0 else 4, "x": 0, "y": 0},
        )

    for u, v, attrs in graph.edges(data=True):
        is_company_edge = attrs.get("kind") == "company"
        net.add_edge(
            u, v,
            color={"color": "rgba(120,170,255,0.28)" if is_company_edge else "rgba(255,255,255,0.07)"},
            width=1.4 if is_company_edge else 0.6,
            smooth=False,
        )

    net.set_options(json.dumps({
        "interaction": {
            "hover": True,
            "tooltipDelay": 80,
            "zoomView": True,
            "dragView": True,
            "navigationButtons": False,
        },
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -60,
                "centralGravity": 0.008,
                "springLength": 90,
                "springConstant": 0.02,
                "damping": 0.4,
                "avoidOverlap": 0.6,
            },
            "stabilization": {"enabled": True, "iterations": 250, "fit": True},
            "minVelocity": 0.5,
        },
    }))

    try:
        # Don't use net.write_html() directly: pyvis opens the output file
        # with Python's platform default encoding, which is cp1252 on
        # Windows and crashes ("'charmap' codec can't encode character...")
        # on any name with non-Latin-1 characters (accents, CJK, emoji,
        # etc.). Generating the HTML string ourselves and writing it with
        # an explicit utf-8 encoding avoids that entirely.
        html = net.generate_html(notebook=False)
        Path(output_path).write_text(html, encoding="utf-8")
    except Exception as e:
        print(f"Failed to write interactive HTML graph: {e}", file=sys.stderr)
        return

    _apply_constellation_theme(output_path, graph, company_icon=company_icon)
    print(f"Constellation graph saved to: {output_path} (open it in a browser)")


def _apply_constellation_theme(output_path: str, graph: nx.Graph, company_icon: str | None = None):
    """
    Post-process the pyvis-generated HTML: strip pyvis's default white card
    chrome, inject a full-page dark layout, an industry legend, a live
    search box, and a "stop physics once settled" behavior so the graph
    is calm to interact with once it's laid out. Also, if a custom
    company_icon was supplied, injects it exactly once and swaps it onto
    every company node at load time (see the note in build_constellation_html
    on why this happens here instead of being embedded per-node).
    """
    path = Path(output_path)
    html = path.read_text(encoding="utf-8")

    person_nodes = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "person"]
    company_nodes = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "company"]

    industry_counts = Counter(graph.nodes[n]["industry"] for n in person_nodes)
    isolate_count = sum(1 for n in person_nodes if graph.degree(n) == 0)
    connected_count = len(person_nodes) - isolate_count

    legend_rows = "".join(
        f'<div class="legend-row">'
        f'<span class="dot" style="background:{INDUSTRY_COLORS.get(ind, "#5A6270")}"></span>'
        f'<span class="legend-label">{ind}</span>'
        f'<span class="legend-count">{count}</span>'
        f'</div>'
        for ind, count in industry_counts.most_common()
    )

    icon_swap_js = ""
    if company_icon:
        # Embedded exactly once here (not per-node — see the comment in
        # build_constellation_html) and applied to every company node
        # right after load.
        icon_swap_js = f"""
    var __companyIcon = {json.dumps(company_icon)};
    var __iconUpdates = [];
    nodes.forEach(function(n) {{
      if (n.icon_pending) {{ __iconUpdates.push({{ id: n.id, image: __companyIcon }}); }}
    }});
    if (__iconUpdates.length) {{ nodes.update(__iconUpdates); }}
"""

    overlay = f"""
<style>
  html, body {{ margin:0; padding:0; background:#05070d; overflow:hidden; }}
  .card {{ border:none !important; background:transparent !important; }}
  #mynetwork {{ background:#05070d !important; }}

  #constellation-ui * {{ box-sizing:border-box; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }}
  .panel {{
    position:fixed; z-index:1000; background:rgba(12,16,28,0.82);
    backdrop-filter: blur(10px); border:1px solid rgba(255,255,255,0.08);
    border-radius:12px; color:#e8ecf5; box-shadow:0 8px 24px rgba(0,0,0,0.4);
  }}
  #search-panel {{ top:18px; left:18px; padding:12px 14px; width:260px; }}
  #search-panel input {{
    width:100%; padding:9px 10px; border-radius:8px; border:1px solid rgba(255,255,255,0.12);
    background:rgba(255,255,255,0.06); color:#fff; font-size:14px; outline:none;
  }}
  #search-panel input::placeholder {{ color:rgba(232,236,245,0.45); }}
  #search-meta {{ margin-top:8px; font-size:12px; color:rgba(232,236,245,0.6); min-height:16px; }}
  #reset-btn {{
    margin-top:8px; width:100%; padding:7px; border-radius:8px; border:1px solid rgba(255,255,255,0.12);
    background:rgba(255,255,255,0.06); color:#e8ecf5; font-size:12px; cursor:pointer;
  }}
  #reset-btn:hover {{ background:rgba(255,255,255,0.14); }}

  #legend-panel {{ top:18px; right:18px; padding:14px 16px; max-height:80vh; overflow-y:auto; width:230px; }}
  #legend-panel h3 {{ margin:0 0 10px 0; font-size:13px; letter-spacing:0.04em; text-transform:uppercase; color:rgba(232,236,245,0.75); }}
  .legend-row {{ display:flex; align-items:center; gap:8px; padding:4px 0; font-size:13px; }}
  .dot {{ width:10px; height:10px; border-radius:50%; flex:0 0 auto; box-shadow:0 0 6px currentColor; }}
  .legend-label {{ flex:1; }}
  .legend-count {{ color:rgba(232,236,245,0.5); font-size:12px; }}
  #stats-line {{ margin-top:10px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.08); font-size:12px; color:rgba(232,236,245,0.6); line-height:1.5; }}
</style>

<div id="constellation-ui">
  <div class="panel" id="search-panel">
    <input type="text" id="star-search" placeholder="Search a connection by name..." autocomplete="off">
    <div id="search-meta"></div>
    <button id="reset-btn">Reset view</button>
  </div>
  <div class="panel" id="legend-panel">
    <h3>Industry</h3>
    {legend_rows}
    <div id="stats-line">{connected_count} connected &middot; {isolate_count} unconnected<br>{len(person_nodes)} total connections &middot; {len(company_nodes)} company hubs</div>
    <div class="legend-row" style="margin-top:8px;">
      <span class="dot" style="background:#FFD770;box-shadow:0 0 6px #FFD770"></span>
      <span class="legend-label" style="font-size:11px; color:rgba(232,236,245,0.65);">Gold stars = companies</span>
    </div>
  </div>
</div>

<script>
(function() {{
  function whenReady() {{
    if (typeof network === "undefined" || typeof nodes === "undefined") {{
      setTimeout(whenReady, 100);
      return;
    }}

    var originalColors = {{}};
    nodes.forEach(function(n) {{ originalColors[n.id] = n.color; }});
{icon_swap_js}
    // Freeze layout once it settles so interaction stays smooth
    network.on("stabilizationIterationsDone", function() {{
      network.setOptions({{ physics: false }});
    }});

    var searchInput = document.getElementById("star-search");
    var meta = document.getElementById("search-meta");
    var resetBtn = document.getElementById("reset-btn");

    function dim(color) {{
      return {{ background: "rgba(60,64,74,0.25)", border: "rgba(60,64,74,0.25)" }};
    }}

    function runSearch() {{
      var term = searchInput.value.trim().toLowerCase();
      if (!term) {{
        var restore = nodes.map(function(n) {{ return {{ id: n.id, color: originalColors[n.id] }}; }});
        nodes.update(restore);
        meta.textContent = "";
        return;
      }}
      var matches = [];
      var updates = nodes.map(function(n) {{
        var isMatch = (n.name || "").toLowerCase().indexOf(term) !== -1;
        if (isMatch) matches.push(n.id);
        return {{ id: n.id, color: isMatch ? originalColors[n.id] : dim(originalColors[n.id]) }};
      }});
      nodes.update(updates);
      meta.textContent = matches.length + " match" + (matches.length === 1 ? "" : "es");
      if (matches.length > 0 && matches.length <= 25) {{
        network.fit({{ nodes: matches, animation: {{ duration: 500, easingFunction: "easeInOutQuad" }} }});
      }}
    }}

    searchInput.addEventListener("input", runSearch);
    resetBtn.addEventListener("click", function() {{
      searchInput.value = "";
      runSearch();
      network.fit({{ animation: {{ duration: 500, easingFunction: "easeInOutQuad" }} }});
    }});
  }}
  whenReady();
}})();
</script>
"""

    html = html.replace("</body>", overlay + "\n</body>")
    path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------
# Summary stats
# --------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, graph: nx.Graph):
    person_nodes = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "person"]
    company_nodes = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "company"]

    print("\n--- Network Summary ---")
    print(f"Total connections loaded: {len(df)}")
    print(
        f"Graph nodes: {len(person_nodes)} people + {len(company_nodes)} company hubs "
        f"= {graph.number_of_nodes()}  |  Graph edges: {graph.number_of_edges()}"
    )

    industries = Counter(graph.nodes[n]["industry"] for n in person_nodes)
    print("\nTop inferred industries:")
    for industry, count in industries.most_common(10):
        print(f"  {industry:<30} {count}")

    companies = Counter(graph.nodes[n]["company"] for n in person_nodes)
    companies.pop("Unknown Company", None)
    print("\nTop companies in your network:")
    for company, count in companies.most_common(10):
        if count > 1:
            print(f"  {company:<30} {count}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build an interactive HTML constellation graph from your LinkedIn Connections CSV export."
    )
    parser.add_argument("csv_path", help="Path to your LinkedIn Connections.csv export")
    parser.add_argument(
        "--industry-map", default=None,
        help="Optional CSV (Company,Industry) to override automatic industry inference",
    )
    parser.add_argument(
        "--output-prefix", default="linkedin_network",
        help="Filename (without extension) for the output HTML file (default: linkedin_network)",
    )
    parser.add_argument(
        "--company-icon", default="company-icon.png",
        help="Path to an icon image shown on every company node (default: looks for "
             "'company-icon.png' next to where you run the script; falls back to a "
             "generated building icon if not found)",
    )
    args = parser.parse_args()

    try:
        override_map = load_industry_override(args.industry_map)
        df = load_linkedin_csv(Path(args.csv_path))
        graph = build_similarity_graph(df, override_map)
        print_summary(df, graph)
        company_icon = load_company_icon(args.company_icon)
        build_constellation_html(graph, f"{args.output_prefix}.html", df, company_icon=company_icon)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
