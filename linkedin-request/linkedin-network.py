#!/usr/bin/env python3
"""
LinkedIn Connections Network Visualizer
=========================================

  - Nodes = your connections
  - Edges = drawn between two people who share a company, or who are
    inferred to be in the same industry/role bucket
  - Node color = inferred industry
  - Node clustering = pulled together if they share a company (via
    edge weighting in the layout)


USAGE
-----
    python linkedin_network_graph.py Connections.csv
    python linkedin_network_graph.py Connections.csv --interactive
    python linkedin_network_graph.py Connections.csv --industry-map my_industries.csv

Expected LinkedIn export columns (as of 2025/2026 export format):
    First Name, Last Name, URL, Email Address, Company, Position, Connected On

The script is tolerant of:
    - The extra "Notes:" preamble LinkedIn puts at the top of the CSV
    - Missing/blank columns (Email Address is often blank)
    - Encoding issues (falls back from utf-8 to latin-1)
    - Duplicate or malformed rows
"""

import argparse
import csv
import sys
import re
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

# Distinct colors for up to ~12 industry buckets (matplotlib-friendly names)
INDUSTRY_COLORS = {
    "Technology": "#4C72B0",
    "Finance": "#55A868",
    "Sales & Marketing": "#C44E52",
    "Healthcare": "#8172B2",
    "Education": "#CCB974",
    "Legal": "#64B5CD",
    "Engineering & Manufacturing": "#E17C05",
    "Consulting": "#B276B2",
    "HR & Recruiting": "#DA8BC3",
    "Creative & Media": "#8C8C00",
    "Executive / Leadership": "#937860",
    DEFAULT_INDUSTRY: "#A9A9A9",
}


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
      - each connection is a node, labeled by name, colored by inferred industry
      - an edge is added between two people who share the same company
      - a lighter-weight edge is added between two people in the same
        inferred industry bucket (only if they don't already share an edge),
        capped so we don't create an unreadable hairball
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
            company=company,
            position=position,
            industry=industry,
            color=INDUSTRY_COLORS.get(industry, INDUSTRY_COLORS[DEFAULT_INDUSTRY]),
        )
        company_groups[company].append(node_id)
        industry_groups[industry].append(node_id)

    # Strong edges: shared company
    for company, members in company_groups.items():
        if company == "Unknown Company" or len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                graph.add_edge(members[i], members[j], weight=3, kind="company")

    # Weaker edges: shared industry (only connect a chain within each
    # industry bucket rather than a full clique, to keep it readable)
    for industry, members in industry_groups.items():
        if industry == DEFAULT_INDUSTRY or len(members) < 2:
            continue
        for i in range(len(members) - 1):
            a, b = members[i], members[i + 1]
            if not graph.has_edge(a, b):
                graph.add_edge(a, b, weight=1, kind="industry")

    return graph


# --------------------------------------------------------------------------
# Static visualization (matplotlib)
# --------------------------------------------------------------------------

def visualize_static(graph: nx.Graph, output_path: str):
    import matplotlib.pyplot as plt

    if graph.number_of_nodes() == 0:
        print("Graph is empty, nothing to visualize.", file=sys.stderr)
        return

    plt.figure(figsize=(16, 12))

    weights = [graph[u][v].get("weight", 1) for u, v in graph.edges()]
    pos = nx.spring_layout(graph, k=0.6, weight="weight", seed=42, iterations=100)

    node_colors = [graph.nodes[n].get("color", "#A9A9A9") for n in graph.nodes()]
    node_sizes = [200 + 60 * graph.degree(n) for n in graph.nodes()]

    nx.draw_networkx_edges(graph, pos, alpha=0.25, width=[0.3 + 0.4 * w for w in weights])
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)

    # Only label nodes with at least one connection to reduce clutter
    labels = {n: n for n in graph.nodes() if graph.degree(n) > 0}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7)

    # Legend for industries actually present
    present_industries = sorted({graph.nodes[n]["industry"] for n in graph.nodes()})
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=ind,
            markerfacecolor=INDUSTRY_COLORS.get(ind, "#A9A9A9"), markersize=9,
        )
        for ind in present_industries
    ]
    plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

    plt.title("LinkedIn Network — Similarity Graph (by shared company / industry)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Static graph saved to: {output_path}")
    plt.close()


# --------------------------------------------------------------------------
# Interactive visualization (pyvis)
# --------------------------------------------------------------------------

def visualize_interactive(graph: nx.Graph, output_path: str):
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

    net = Network(height="900px", width="100%", bgcolor="#111111", font_color="white")
    net.barnes_hut(gravity=-8000, spring_length=120)

    for node, attrs in graph.nodes(data=True):
        title = (
            f"{node}<br>Company: {attrs.get('company', 'N/A')}"
            f"<br>Role: {attrs.get('position', 'N/A')}"
            f"<br>Industry: {attrs.get('industry', 'N/A')}"
        )
        net.add_node(
            node,
            label=node,
            title=title,
            color=attrs.get("color", "#A9A9A9"),
            size=12 + 3 * graph.degree(node),
        )

    for u, v, attrs in graph.edges(data=True):
        net.add_edge(u, v, value=attrs.get("weight", 1), title=attrs.get("kind", ""))

    try:
        net.write_html(output_path, open_browser=False, notebook=False)
        print(f"Interactive graph saved to: {output_path} (open it in a browser)")
    except Exception as e:
        print(f"Failed to write interactive HTML graph: {e}", file=sys.stderr)


# --------------------------------------------------------------------------
# Summary stats
# --------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, graph: nx.Graph):
    print("\n--- Network Summary ---")
    print(f"Total connections loaded: {len(df)}")
    print(f"Graph nodes: {graph.number_of_nodes()}  |  Graph edges: {graph.number_of_edges()}")

    industries = Counter(nx.get_node_attributes(graph, "industry").values())
    print("\nTop inferred industries:")
    for industry, count in industries.most_common(10):
        print(f"  {industry:<30} {count}")

    companies = Counter(nx.get_node_attributes(graph, "company").values())
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
        description="Build a similarity graph from your LinkedIn Connections CSV export."
    )
    parser.add_argument("csv_path", help="Path to your LinkedIn Connections.csv export")
    parser.add_argument(
        "--interactive", action="store_true",
        help="Also generate an interactive HTML graph (pyvis) in addition to the static PNG",
    )
    parser.add_argument(
        "--industry-map", default=None,
        help="Optional CSV (Company,Industry) to override automatic industry inference",
    )
    parser.add_argument(
        "--output-prefix", default="linkedin_network",
        help="Prefix for output files (default: linkedin_network)",
    )
    args = parser.parse_args()

    try:
        override_map = load_industry_override(args.industry_map)
        df = load_linkedin_csv(Path(args.csv_path))
        graph = build_similarity_graph(df, override_map)
        print_summary(df, graph)
        visualize_static(graph, f"{args.output_prefix}.png")
        if args.interactive:
            visualize_interactive(graph, f"{args.output_prefix}.html")
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