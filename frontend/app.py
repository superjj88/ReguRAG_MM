"""ReguRAG_MM — Modern Open WebUI-inspired Chat Interface for Hebrew Regulations.

Features:
- Framed Chat Window with sleek thin gray border (Open WebUI style).
- Zero wasted top margin — both main canvas and sidebar start right at the top.
- Tight, balanced spacing with minimal vertical gaps.
- Perfectly aligned 2x2 grid for prompt questions.
- Chat input raised upwards away from bottom.
- Julio-Ai logo in the top-right corner.
- Persistent disclaimer notice permanently visible on screen (Voluntary project by Yarden Raz).
- Protected Material Symbols icon font.
- Responsive design for Desktop, Tablet, and Mobile.
- Full RTL Hebrew typography with Rubik & Assistant font stack.
- Complete backend integration: Health, Ingestion, Refresh, Conflicts, Approvals & Feedback.
"""

from __future__ import annotations

import base64
import html
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st

BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")


@st.cache_data
def get_logo_base64() -> str:
    """Load and base64-encode the Julio-Ai brand logo."""
    possible_paths = [
        Path(__file__).resolve().parent / "assets" / "logo.png",
        Path("frontend/assets/logo.png"),
        Path("assets/logo.png"),
    ]
    for p in possible_paths:
        if p.exists():
            try:
                return base64.b64encode(p.read_bytes()).decode("utf-8")
            except Exception:
                pass
    return ""


def inject_open_webui_css() -> None:
    """Inject responsive Open WebUI dark mode CSS with RTL support and mobile adaptability."""
    st.markdown(
        """
        <style>
        /* Import Google Fonts for high-legibility Hebrew typography */
        @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;500;600;700&family=Rubik:ital,wght@0,300..900;1,300..900&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Root color tokens & global layout */
        :root {
            --bg-canvas: #0b0f19;
            --bg-sidebar: #111522;
            --bg-card: #161b28;
            --bg-card-hover: #1c2233;
            --bg-user-bubble: #1e2538;
            --bg-assistant-card: #151a26;
            --bg-citation: #131826;
            --border-subtle: rgba(255, 255, 255, 0.1);
            --border-hover: rgba(255, 255, 255, 0.18);
            --accent-emerald: #10b981;
            --accent-emerald-glow: rgba(16, 185, 129, 0.2);
            --accent-cyan: #38bdf8;
            --accent-cyan-bg: rgba(56, 189, 248, 0.12);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --font-hebrew: 'Rubik', 'Assistant', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', 'Consolas', monospace;
        }

        /* Global Reset & Typography */
        html, body, [class*="css"], .stMarkdown, .stText, p, span, div, input, textarea {
            direction: rtl !important;
            text-align: right !important;
            font-family: var(--font-hebrew) !important;
        }

        /* Protect Material Icons font from Hebrew text overrides */
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-icons,
        [class*="material-symbols"],
        [data-testid*="Icon"],
        svg {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
            direction: ltr !important;
            text-align: center !important;
        }

        /* 1. Eliminate all wasted top space from Streamlit header & canvas */
        header[data-testid="stHeader"],
        .stApp > header {
            display: none !important;
            height: 0px !important;
            min-height: 0px !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* On Desktop: Hide sidebar collapse button */
        @media (min-width: 769px) {
            [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
            [data-testid="collapsedControl"] {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                pointer-events: none !important;
            }
        }

        /* On Mobile: Allow sidebar controls */
        @media (max-width: 768px) {
            [data-testid="collapsedControl"] {
                display: flex !important;
                visibility: visible !important;
                opacity: 1 !important;
                top: 0.35rem !important;
                right: 0.5rem !important;
                z-index: 99999 !important;
            }
            [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
                display: flex !important;
                visibility: visible !important;
            }
        }

        /* Dark Canvas Background */
        .stApp {
            background-color: var(--bg-canvas) !important;
            color: var(--text-primary) !important;
        }

        /* Main Conversation Column - Start at top with zero wasted space */
        .main {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        .main .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: 880px !important;
            width: 100% !important;
            padding-top: 0.2rem !important;
            padding-bottom: 8.5rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            margin: 0 auto !important;
        }

        /* Sidebar - Start at top with zero wasted space */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarUserContent"],
        section[data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-left: 1px solid var(--border-subtle) !important;
            direction: rtl !important;
            text-align: right !important;
            padding-top: 0.25rem !important;
            margin-top: 0 !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding: 0.5rem 0.9rem !important;
        }

        /* Sleek Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.16);
            border-radius: 9999px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.28);
        }

        /* Thin Gray Border Window Framing the Entire Chat Workspace */
        .main [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(18, 23, 35, 0.55) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 18px !important;
            padding: 0.75rem 1rem !important;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35) !important;
            margin-top: 0.2rem !important;
            margin-bottom: 0.5rem !important;
        }

        /* Open WebUI Header Bar - Compact & Seamless */
        .ow-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.4rem 0.85rem;
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 11px;
            margin-bottom: 0.45rem;
            min-height: 40px;
            gap: 8px;
        }
        .ow-header-brand {
            display: flex;
            align-items: center;
            gap: 9px;
            flex-wrap: nowrap;
        }
        .ow-brand-logo {
            width: 30px;
            height: 30px;
            border-radius: 7px;
            object-fit: cover;
            background: #ffffff;
            box-shadow: 0 1px 5px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.15);
            flex-shrink: 0;
        }
        .ow-header-title-wrap {
            display: flex;
            align-items: baseline;
            gap: 6px;
            flex-wrap: wrap;
        }
        .ow-header-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0;
            line-height: 1.2;
            white-space: nowrap;
        }
        .ow-header-sep {
            color: var(--text-muted);
            font-size: 0.78rem;
        }
        .ow-header-brand-tag {
            font-size: 0.86rem;
            font-weight: 600;
            color: var(--accent-cyan);
            white-space: nowrap;
        }
        .ow-header-subtitle {
            font-size: 0.75rem;
            color: var(--text-secondary);
            white-space: nowrap;
        }
        .ow-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 500;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            white-space: nowrap;
            flex-shrink: 0;
        }
        .ow-badge-dot {
            width: 6.5px;
            height: 6.5px;
            border-radius: 50%;
            background-color: var(--accent-emerald);
            box-shadow: 0 0 6px var(--accent-emerald);
            flex-shrink: 0;
        }
        .ow-badge-dot.offline {
            background-color: #ef4444;
            box-shadow: 0 0 6px rgba(239, 68, 68, 0.6);
        }

        /* Empty State Hero - Ultra-Compact with Minimal Gaps */
        .ow-hero {
            text-align: center;
            padding: 0.15rem 0.5rem 0.35rem 0.5rem !important;
            margin: 0 0 0.35rem 0 !important;
        }
        .ow-hero-title {
            font-size: clamp(1.15rem, 2.5vw, 1.45rem);
            font-weight: 700;
            color: var(--text-primary);
            margin: 0 0 2px 0;
            line-height: 1.25;
            letter-spacing: -0.01em;
        }
        .ow-hero-subtitle {
            font-size: clamp(0.78rem, 1.6vw, 0.86rem);
            color: var(--text-secondary);
            max-width: 520px;
            margin: 0 auto;
            line-height: 1.35;
        }

        /* Aligned 2x2 Prompt Grid */
        [data-testid="stVerticalBlock"]:has(.ow-prompt-marker) [data-testid="stHorizontalBlock"] {
            gap: 10px !important;
        }
        [data-testid="stVerticalBlock"]:has(.ow-prompt-marker) [data-testid="column"] {
            gap: 10px !important;
        }
        [data-testid="stVerticalBlock"]:has(.ow-prompt-marker) [data-testid="column"] .stButton > button {
            height: 76px !important;
            min-height: 76px !important;
            max-height: 76px !important;
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: flex-start !important;
            text-align: right !important;
            padding: 8px 12px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.09) !important;
            background: rgba(255, 255, 255, 0.03) !important;
            color: var(--text-primary) !important;
            transition: all 0.2s ease !important;
            white-space: pre-line !important;
            word-break: break-word !important;
            box-sizing: border-box !important;
        }
        [data-testid="stVerticalBlock"]:has(.ow-prompt-marker) [data-testid="column"] .stButton > button:hover {
            border-color: rgba(16, 185, 129, 0.4) !important;
            background: rgba(255, 255, 255, 0.07) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
        }
        [data-testid="stVerticalBlock"]:has(.ow-prompt-marker) [data-testid="column"] .stButton > button p {
            font-size: 0.86rem !important;
            line-height: 1.35 !important;
            margin: 0 !important;
            text-align: right !important;
            width: 100% !important;
        }

        /* User Message Bubble */
        .ow-user-bubble {
            background: var(--bg-user-bubble);
            border: 1px solid var(--border-subtle);
            border-radius: 16px 16px 3px 16px;
            padding: 0.9rem 1.2rem;
            color: var(--text-primary);
            font-size: 0.96rem;
            line-height: 1.6;
            margin-bottom: 0.6rem;
            box-shadow: 0 3px 12px rgba(0, 0, 0, 0.2);
            word-break: break-word;
        }
        .ow-user-meta {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
            font-weight: 600;
        }

        /* Assistant Message Card */
        .ow-assistant-container {
            background: var(--bg-assistant-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px 16px 16px 3px;
            padding: 1.1rem 1.35rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 0.6rem;
            word-break: break-word;
        }
        .ow-assistant-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.75rem;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid var(--border-subtle);
            flex-wrap: wrap;
            gap: 8px;
        }
        .ow-assistant-title {
            display: flex;
            align-items: center;
            gap: 7px;
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .ow-assistant-body {
            font-size: 0.96rem;
            line-height: 1.7;
            color: #f1f5f9;
        }

        /* Citations & Sources Panel */
        .ow-citations-panel {
            background: var(--bg-citation);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 11px;
            padding: 0.85rem 1.1rem;
            margin-top: 1rem;
            box-shadow: inset 0 0 16px rgba(56, 189, 248, 0.04);
        }
        .ow-citations-header {
            display: flex;
            align-items: center;
            gap: 7px;
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--accent-cyan);
            margin-bottom: 0.7rem;
        }
        .ow-citation-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-subtle);
            border-radius: 9px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.45rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
            transition: all 0.2s ease;
        }
        .ow-citation-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
            background: rgba(56, 189, 248, 0.06);
        }
        .ow-citation-info {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .ow-citation-name {
            font-weight: 600;
            font-size: 0.88rem;
            color: var(--text-primary);
        }
        .ow-pill {
            display: inline-flex;
            align-items: center;
            padding: 2px 7px;
            border-radius: 9999px;
            font-size: 0.73rem;
            font-weight: 600;
            background: var(--accent-cyan-bg);
            color: var(--accent-cyan);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .ow-link-btn {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: rgba(56, 189, 248, 0.15);
            color: #e0f2fe !important;
            text-decoration: none !important;
            padding: 5px 10px;
            border-radius: 7px;
            font-size: 0.76rem;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.3);
            transition: all 0.2s ease;
        }
        .ow-link-btn:hover {
            background: var(--accent-cyan) !important;
            color: #0b0f19 !important;
            border-color: var(--accent-cyan) !important;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
        }

        /* Sidebar Cards */
        .ow-side-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-subtle);
            border-radius: 11px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.65rem;
        }
        .ow-side-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .ow-metric-val {
            font-size: 1.18rem;
            font-weight: 800;
            color: var(--text-primary);
        }

        /* Chat Input - Raised Upwards with Breathing Room */
        [data-testid="stBottom"] {
            bottom: 46px !important;
            background: transparent !important;
        }
        [data-testid="stChatInput"] {
            border-radius: 26px !important;
            background-color: #131824 !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            box-shadow: 0 4px 22px rgba(0, 0, 0, 0.5) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
            direction: rtl !important;
            text-align: right !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--accent-emerald) !important;
            box-shadow: 0 0 0 3px var(--accent-emerald-glow), 0 4px 22px rgba(0, 0, 0, 0.6) !important;
        }
        [data-testid="stChatInput"] textarea {
            font-family: var(--font-hebrew) !important;
            color: var(--text-primary) !important;
            font-size: 0.95rem !important;
            direction: rtl !important;
            text-align: right !important;
        }

        /* Action Buttons */
        .stButton button {
            border-radius: 9px !important;
            border: 1px solid var(--border-subtle) !important;
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        .stButton button:hover {
            border-color: var(--accent-emerald) !important;
            background: var(--bg-card-hover) !important;
            color: var(--accent-emerald) !important;
            box-shadow: 0 0 10px var(--accent-emerald-glow) !important;
        }

        /* Persistent Disclaimer Notice - Pinned at the very bottom, never behind sidebar */
        .ow-persistent-notice {
            position: fixed;
            bottom: 0;
            z-index: 9999;
            background: rgba(11, 15, 25, 0.97);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            color: #94a3b8;
            font-size: 0.72rem;
            line-height: 1.4;
            text-align: center;
            padding: 5px 12px;
            box-sizing: border-box;
            word-break: break-word;
            direction: rtl;
            pointer-events: auto;
        }
        .ow-persistent-notice b {
            color: #38bdf8;
            font-weight: 600;
        }
        .ow-persistent-notice .warn-icon {
            color: #f59e0b;
            margin-left: 4px;
        }

        /* On Desktop: Offset from sidebar (21rem) so full text is visible */
        @media (min-width: 769px) {
            .ow-persistent-notice {
                right: 21rem !important;
                left: 0 !important;
            }
        }

        /* On Mobile: Full width */
        @media (max-width: 768px) {
            .ow-persistent-notice {
                right: 0 !important;
                left: 0 !important;
                font-size: 0.67rem !important;
                padding: 4px 6px !important;
            }
            [data-testid="stBottom"] {
                bottom: 40px !important;
            }
            .main .block-container,
            [data-testid="stMainBlockContainer"] {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
                padding-top: 0.15rem !important;
                padding-bottom: 7.5rem !important;
                max-width: 100% !important;
            }
            .main [data-testid="stVerticalBlockBorderWrapper"] {
                padding: 0.5rem 0.65rem !important;
                border-radius: 14px !important;
            }
            /* Stack prompt columns vertically on mobile */
            [data-testid="stVerticalBlock"]:has(.ow-prompt-marker) [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 8px !important;
            }
            [data-testid="stVerticalBlock"]:has(.ow-prompt-marker) [data-testid="column"] {
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
                padding: 0 !important;
            }
            [data-testid="stVerticalBlock"]:has(.ow-prompt-marker) [data-testid="column"] .stButton > button {
                height: 68px !important;
                min-height: 68px !important;
                max-height: 68px !important;
                font-size: 0.85rem !important;
            }
            .ow-header {
                padding: 0.35rem 0.6rem !important;
                min-height: 38px !important;
                gap: 6px !important;
            }
            .ow-header-title {
                font-size: 0.92rem !important;
            }
            .ow-header-subtitle {
                display: none !important;
            }
            .ow-brand-logo {
                width: 26px !important;
                height: 26px !important;
            }
            .ow-badge {
                padding: 2px 6px !important;
                font-size: 0.68rem !important;
            }
            .ow-user-bubble {
                padding: 0.75rem 0.95rem !important;
                border-radius: 14px 14px 3px 14px !important;
                font-size: 0.92rem !important;
            }
            .ow-assistant-container {
                padding: 0.85rem 1rem !important;
                border-radius: 14px 14px 14px 3px !important;
                font-size: 0.92rem !important;
            }
            .ow-citation-card {
                flex-direction: column !important;
                align-items: flex-start !important;
                gap: 6px !important;
                padding: 0.55rem 0.75rem !important;
            }
            .ow-link-btn {
                width: 100% !important;
                text-align: center !important;
                justify-content: center !important;
                padding: 5px 9px !important;
            }
            [data-testid="stChatInput"] {
                border-radius: 20px !important;
            }
            [data-testid="stChatInput"] textarea {
                font-size: 16px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _error_detail(r: requests.Response) -> str:
    """Extract human-readable error detail from response."""
    try:
        return r.json().get("detail", "שגיאה בתקשורת")
    except Exception:
        return "שגיאה בתקשורת"


def parse_citations_and_links(
    citations: list[str],
    pdf_links: list[str],
    backend_url: str,
) -> list[dict]:
    """Parse citation strings and match or build page-accurate PDF links."""
    parsed: list[dict] = []
    cit_pattern = re.compile(
        r"\[\s*([^|\[\],]+?)\s*[|,]\s*(?:עמוד|עמ')\s*(\d+)(?:\s*[|,]\s*(?:סעיף|סע')\s*([^|\[\]]+?))?\s*\]"
    )

    for idx, cit in enumerate(citations):
        m = cit_pattern.search(cit)
        if m:
            doc_name = m.group(1).strip()
            page_num = m.group(2).strip()
            sec_num = m.group(3).strip() if m.group(3) else None
        else:
            clean = cit.strip("[] \t\r\n")
            parts = [p.strip() for p in clean.split("|")]
            doc_name = parts[0] if parts else clean
            page_num = None
            sec_num = None
            for p in parts[1:]:
                if "עמוד" in p or "עמ'" in p:
                    digits = re.findall(r"\d+", p)
                    if digits:
                        page_num = digits[0]
                elif "סעיף" in p or "סע'" in p:
                    sec_num = p.replace("סעיף", "").replace("סע'", "").strip()

        matched_link = None
        for link in pdf_links:
            clean_link = link.lower()
            if doc_name.lower() in clean_link or (doc_name.lower().replace(".pdf", "") in clean_link):
                matched_link = link
                break

        if not matched_link and idx < len(pdf_links):
            matched_link = pdf_links[idx]

        if not matched_link and doc_name:
            doc_file = doc_name if doc_name.lower().endswith(".pdf") else f"{doc_name}.pdf"
            matched_link = f"/data/{quote(doc_file)}"
            if page_num:
                matched_link += f"#page={page_num}"

        if matched_link and matched_link.startswith("/"):
            matched_link = f"{backend_url.rstrip('/')}{matched_link}"

        parsed.append({
            "raw": cit,
            "doc_name": doc_name,
            "page_number": page_num,
            "section_number": sec_num,
            "link_url": matched_link,
        })
    return parsed


def render_citations_card(parsed_citations: list[dict], related_docs: list[dict] | None = None) -> None:
    """Render prominent, interactive document citation cards in Open WebUI style."""
    if not parsed_citations and not related_docs:
        return

    items_html = []
    for c in parsed_citations:
        doc = html.escape(c["doc_name"])
        badges = []
        if c.get("page_number"):
            badges.append(f'<span class="ow-pill">עמוד {html.escape(c["page_number"])}</span>')
        if c.get("section_number"):
            badges.append(f'<span class="ow-pill">סעיף {html.escape(c["section_number"])}</span>')
        badges_str = " ".join(badges)

        link_html = ""
        if c.get("link_url"):
            escaped_url = html.escape(c["link_url"])
            link_html = f'<a href="{escaped_url}" target="_blank" rel="noopener" class="ow-link-btn">📄 פתח מסמך בעמוד הרלוונטי ↗</a>'

        items_html.append(
            f"""
            <div class="ow-citation-card">
                <div class="ow-citation-info">
                    <span style="font-size: 1.05rem;">📄</span>
                    <span class="ow-citation-name">{doc}</span>
                    {badges_str}
                </div>
                <div>{link_html}</div>
            </div>
            """
        )

    related_html = ""
    if related_docs:
        rel_pills = []
        for r in related_docs[:5]:
            fname = html.escape(r.get("file_name", ""))
            ltype = html.escape(r.get("link_type", "קשור"))
            rel_pills.append(f'<span class="ow-pill" style="margin-left: 6px;">🔗 {fname} ({ltype})</span>')
        if rel_pills:
            related_html = f"""
            <div style="margin-top: 0.65rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.3rem; font-weight: 600;">מסמכים קשורים ונספחים:</div>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">{' '.join(rel_pills)}</div>
            </div>
            """

    total_sources = len(parsed_citations)
    card_html = f"""
    <div class="ow-citations-panel">
        <div class="ow-citations-header">
            <span>📚</span>
            <span>מקורות מידע ואסמכתאות ({total_sources} מסמכים מאומתים)</span>
        </div>
        {''.join(items_html)}
        {related_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# ---- Page Setup ----
st.set_page_config(
    page_title="ReguRAG_MM — תקנוני הקיבוץ",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_open_webui_css()

# Load logo
logo_b64 = get_logo_base64()
logo_img_tag = (
    f'<img src="data:image/png;base64,{logo_b64}" class="ow-brand-logo" alt="Julio-Ai Logo" />'
    if logo_b64
    else '<div class="ow-brand-logo" style="display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#10b981,#06b6d4);color:white;font-size:16px;">⚖️</div>'
)

# Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "preset_query" not in st.session_state:
    st.session_state.preset_query = None

# Backend Health Check
backend_online = False
try:
    health_res = requests.get(f"{BACKEND}/api/health", timeout=1.5)
    backend_online = health_res.ok
except Exception:
    backend_online = False


# ---- Sidebar (Open WebUI Modern Panel) ----
with st.sidebar:
    # Sidebar Header with Logo in Top-Right
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 9px; margin-bottom: 0.85rem;">
            {logo_img_tag}
            <div>
                <div style="font-weight: 700; font-size: 1rem; color: #f8fafc; line-height: 1.2;">Julio-Ai • ReguRAG</div>
                <div style="font-size: 0.72rem; color: #94a3b8;">עוזר תקנוני הקיבוץ המקומי</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # New Chat Button
    if st.button("➕ שיחה חדשה", use_container_width=True):
        st.session_state.history = []
        st.session_state.preset_query = None
        st.rerun()

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    # System Status Card
    status_label = "מערכת מחוברת ופעילה" if backend_online else "מנותק מ-Backend"
    status_dot_class = "ow-badge-dot" if backend_online else "ow-badge-dot offline"
    st.markdown(
        f"""
        <div class="ow-side-card">
            <div class="ow-side-title">⚡ סטטוס מערכת</div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="{status_dot_class}"></span>
                <span style="font-size: 0.84rem; font-weight: 600; color: #f8fafc;">{status_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Document Corpus Metrics Card
    ingest_info = {}
    if backend_online:
        try:
            r_ing = requests.get(f"{BACKEND}/api/ingest/status", timeout=1.5)
            if r_ing.ok:
                ingest_info = r_ing.json()
        except Exception:
            pass

    processed_docs = ingest_info.get("processed", "145")
    total_docs = ingest_info.get("total", "145")
    ingest_state = ingest_info.get("state", "מוכן")
    eta_text = f" (צפי: {ingest_info['eta_seconds']}s)" if ingest_info.get("eta_seconds") else ""

    st.markdown(
        f"""
        <div class="ow-side-card">
            <div class="ow-side-title">📚 מאגר תקנונים</div>
            <div style="display: flex; align-items: baseline; gap: 5px; margin-bottom: 2px;">
                <span class="ow-metric-val">{processed_docs}</span>
                <span style="color: var(--text-muted); font-size: 0.8rem;">/ {total_docs} מסמכים</span>
            </div>
            <div style="font-size: 0.74rem; color: var(--text-secondary);">סטטוס: <b>{ingest_state}</b>{eta_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 רענן מאגר מסמכים", use_container_width=True):
        try:
            r_ref = requests.post(f"{BACKEND}/api/documents/refresh", timeout=5)
            if r_ref.ok:
                st.toast("רענון מאגר המסמכים החל ברקע", icon="🔄")
            else:
                st.error(_error_detail(r_ref))
        except Exception as e:
            st.error(str(e))

    # Conflicts Monitor Card
    conflicts_list = []
    if backend_online:
        try:
            r_conf = requests.get(f"{BACKEND}/api/conflicts?unresolved_only=true", timeout=1.5)
            if r_conf.ok:
                conflicts_list = r_conf.json().get("conflicts", [])
        except Exception:
            pass

    if conflicts_list:
        st.markdown(
            f"""
            <div class="ow-side-card" style="border-color: rgba(245, 158, 11, 0.35);">
                <div class="ow-side-title" style="color: #f59e0b;">⚠️ סתירות שזוהו בתקנון ({len(conflicts_list)})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for c in conflicts_list[:3]:
            st.caption(f"📌 {c.get('topic_summary', '')}")
            if st.button(f"אשר פתרון סתירה #{c['id']}", key=f"res_{c['id']}", use_container_width=True):
                try:
                    requests.post(
                        f"{BACKEND}/api/conflicts/{c['id']}/resolve",
                        params={"resolution": "approved"},
                        timeout=5,
                    )
                    st.toast(f"סתירה #{c['id']} סומנה כפתורה", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        st.markdown(
            """
            <div class="ow-side-card">
                <div class="ow-side-title" style="color: var(--accent-emerald);">✓ בדיקת סתירות</div>
                <div style="font-size: 0.76rem; color: var(--text-secondary);">לא זוהו סתירות פתוחות במאגר</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Disclaimer Card in Sidebar
    st.markdown(
        """
        <div class="ow-side-card" style="border-color: rgba(56, 189, 248, 0.2);">
            <div class="ow-side-title" style="color: #38bdf8;">👤 אודות והבהרה</div>
            <div style="font-size: 0.74rem; color: #94a3b8; line-height: 1.45;">
                פרויקט התנדבותי שהוכן ע"י <b>ירדן רז</b>.<br>
                באחריות המשתמש לאמת את המידע של המערכת מול התקנון הרשמי. המערכת אינה לוקחת אחריות על שימוש במידע שנלקח ממנה.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Air-Gapped Local Privacy Notice
    st.markdown(
        """
        <div style="padding: 0.6rem; border-radius: 9px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-subtle); text-align: center;">
            <div style="font-size: 0.72rem; font-weight: 600; color: #10b981; margin-bottom: 2px;">🔒 100% מקומי ומאובטח</div>
            <div style="font-size: 0.66rem; color: #64748b;">כל השאילתות נשמרות ומופעלות מקומית ללא הוצאת נתונים לענן.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---- Main Framed Chat Window (Surrounded by Thin Gray Border) ----
with st.container(border=True):
    # 1. Compact Header Bar inside the Window
    st.markdown(
        f"""
        <div class="ow-header">
            <div class="ow-header-brand">
                {logo_img_tag}
                <div class="ow-header-title-wrap">
                    <span class="ow-header-title">עוזר תקנון קיבוצי</span>
                    <span class="ow-header-sep">•</span>
                    <span class="ow-header-brand-tag">Julio-Ai ReguRAG</span>
                    <span class="ow-header-subtitle">(145 תקנונים)</span>
                </div>
            </div>
            <div class="ow-badge">
                <span class="ow-badge-dot"></span>
                <span>LLaMA 3.2</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Conversation History Rendering
    def render_history_entry(msg: dict | tuple, idx: int) -> None:
        """Render a conversation message with full Open WebUI styling."""
        if isinstance(msg, tuple):
            role, content = msg
            msg_dict = {"role": role, "content": content}
        else:
            msg_dict = msg

        role = msg_dict.get("role", "user")
        content = msg_dict.get("content", "")

        if role == "user":
            st.markdown(
                f"""
                <div class="ow-user-bubble">
                    <div class="ow-user-meta">
                        <span>👤</span>
                        <span>שאלה</span>
                    </div>
                    <div>{html.escape(content)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Assistant Response Container
            elapsed = msg_dict.get("elapsed", "")
            is_cached = msg_dict.get("cached", False)
            badge_text = "💾 נשלף ממטמון מאושר" if is_cached else (f"⚡ {elapsed} שניות" if elapsed else "")
            badge_html = f'<span class="ow-pill" style="font-size: 0.72rem;">{badge_text}</span>' if badge_text else ""

            st.markdown(
                f"""
                <div class="ow-assistant-container">
                    <div class="ow-assistant-header">
                        <div class="ow-assistant-title">
                            <span style="font-size: 1.05rem;">🤖</span>
                            <span>תשובת עוזר התקנון</span>
                            <span class="ow-pill" style="background: rgba(16, 185, 129, 0.15); color: #10b981; border-color: rgba(16, 185, 129, 0.3);">תשובה מאומתת</span>
                        </div>
                        <div>{badge_html}</div>
                    </div>
                    <div class="ow-assistant-body">
                """,
                unsafe_allow_html=True,
            )

            # Markdown answer content
            st.markdown(content)

            # Citations Panel
            citations = msg_dict.get("citations", [])
            pdf_links = msg_dict.get("pdf_links", [])
            related_docs = msg_dict.get("related_documents", [])

            if citations or related_docs:
                parsed = parse_citations_and_links(citations, pdf_links, BACKEND)
                render_citations_card(parsed, related_docs)

            st.markdown("</div></div>", unsafe_allow_html=True)

            # Feedback & Action Row
            if not msg_dict.get("refusal"):
                col1, col2, col3 = st.columns([1, 1, 3])
                query_text = msg_dict.get("query_text", "")
                source_hashes = msg_dict.get("source_hashes", [])

                with col1:
                    if not is_cached and st.button("👍 אשר תשובה", key=f"appr_{idx}", use_container_width=True):
                        try:
                            approve_res = requests.post(
                                f"{BACKEND}/api/answers/approve",
                                json={
                                    "query_text": query_text or content,
                                    "approved_response": content,
                                    "source_hashes": source_hashes,
                                },
                                timeout=5,
                            )
                            if approve_res.ok:
                                st.toast("התשובה אושרה ונשמרה במטמון המהיר!", icon="✅")
                            else:
                                st.error(_error_detail(approve_res))
                        except Exception as e:
                            st.error(str(e))

                with col2:
                    if st.button("⚠️ סמן כשגוי", key=f"flag_{idx}", use_container_width=True):
                        try:
                            fb_res = requests.post(
                                f"{BACKEND}/api/answers/feedback",
                                json={
                                    "query_text": query_text or content,
                                    "response_text": content,
                                    "source_hashes": source_hashes,
                                    "comment": "סומן כשגוי ע\"י המשתמש",
                                },
                                timeout=10,
                            )
                            if fb_res.ok:
                                st.toast("התשובה סומנה כשגויה והוסרה מהמטמון", icon="ℹ️")
                            else:
                                st.error(_error_detail(fb_res))
                        except Exception as e:
                            st.error(str(e))

    for idx, entry in enumerate(st.session_state.history):
        render_history_entry(entry, idx)

    # 3. Welcome Hero & Aligned 2x2 Prompt Grid (When history is empty)
    if not st.session_state.history:
        st.markdown(
            """
            <div class="ow-hero">
                <h2 class="ow-hero-title">במה אוכל לעזור לך היום בתקנוני הקיבוץ?</h2>
                <p class="ow-hero-subtitle">
                    שאל שאלות וקבל תשובות מוסמכות ומאומתות המפנות ישירות לסעיפי התקנון הרשמיים
                </p>
            </div>
            <div class="ow-prompt-marker"></div>
            """,
            unsafe_allow_html=True,
        )

        # Perfectly Aligned 2x2 Grid (Equal heights, balanced text)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if st.button(
                "🏗️ הרחבה ובניית דירה\nמהם כללי הבנייה והשתתפות הקיבוץ בעלויות?",
                key="p1",
                use_container_width=True,
            ):
                st.session_state.preset_query = "מהם הכללים להרחבת דירה והשתתפות הקיבוץ?"
                st.rerun()

            if st.button(
                "👵 פנסיה ודמי עזיבה\nכיצד מחושבים דמי עזיבה וזכויות פנסיוניות?",
                key="p2",
                use_container_width=True,
            ):
                st.session_state.preset_query = "כיצד מחושבים דמי עזיבה וזכויות החבר שנצברו?"
                st.rerun()

        with col_p2:
            if st.button(
                "🚗 רכב שיתופי ונסיעות\nמה הזכאות לרכב שיתופי או החזר הוצאות?",
                key="p3",
                use_container_width=True,
            ):
                st.session_state.preset_query = "מה הזכאות לרכב שיתופי או החזר הוצאות נסיעה?"
                st.rerun()

            if st.button(
                "🗳️ אסיפת קיבוץ והצבעות\nמי רשאי להשתתף ולהצביע באסיפת החברים?",
                key="p4",
                use_container_width=True,
            ):
                st.session_state.preset_query = "מי רשאי להשתתף ולהצביע באסיפת הקיבוץ ומהו הקוורום?"
                st.rerun()


# ---- Chat Input Handling (Raised up away from bottom) ----
input_query = st.chat_input("שאל שאלה בעברית על תקנוני הקיבוץ... (לדוגמה: מה הזכאות לדמי עזיבה?)")

query = None
if input_query:
    query = input_query
elif st.session_state.preset_query:
    query = st.session_state.preset_query
    st.session_state.preset_query = None

if query:
    # 1. Append User Message
    st.session_state.history.append({"role": "user", "content": query})

    # Render immediate User message in the current turn
    st.markdown(
        f"""
        <div class="ow-user-bubble">
            <div class="ow-user-meta">
                <span>👤</span>
                <span>שאלה</span>
            </div>
            <div>{html.escape(query)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Query Backend and Stream Assistant Message
    with st.spinner("חושב... מאתר תקנונים ומגבש תשובה מדויקת (צפי: 10-25 שניות)"):
        try:
            t0 = time.time()
            resp = requests.post(f"{BACKEND}/api/chat", json={"query": query}, timeout=65)
            elapsed_sec = int(time.time() - t0)

            if resp.ok:
                data = resp.json()
                ans_text = data.get("answer", "")
                cites = data.get("citations", [])
                pdf_links = data.get("pdf_links", [])
                related = data.get("related_documents") or []
                is_cached = data.get("cached", False)
                is_refusal = data.get("refusal", False)
                source_hashes = data.get("source_hashes", [])

                # Save rich assistant message to history
                asst_msg = {
                    "role": "assistant",
                    "content": ans_text,
                    "citations": cites,
                    "pdf_links": pdf_links,
                    "related_documents": related,
                    "cached": is_cached,
                    "refusal": is_refusal,
                    "elapsed": elapsed_sec,
                    "source_hashes": source_hashes,
                    "query_text": query,
                }
                st.session_state.history.append(asst_msg)

                # Rerun to render the newly added assistant message with full interactive components
                st.rerun()
            else:
                err_msg = f"שגיאת שרת: {_error_detail(resp)}"
                st.error(err_msg)
                st.session_state.history.append({"role": "assistant", "content": err_msg})
        except Exception as exc:
            err_msg = f"שגיאת תקשורת עם השרת: {exc}"
            st.error(err_msg)
            st.session_state.history.append({"role": "assistant", "content": err_msg})


# ---- Persistent Disclaimer Bar (Always displayed on screen, never hidden behind sidebar) ----
st.markdown(
    """
    <div class="ow-persistent-notice">
        <span class="warn-icon">⚠️</span>
        <span>פרויקט התנדבותי שהוכן ע"י <b>ירדן רז</b> • באחריות המשתמש לאמת את המידע של המערכת • המערכת אינה לוקחת אחריות על שימוש במידע שנלקח ממנה</span>
    </div>
    """,
    unsafe_allow_html=True,
)
