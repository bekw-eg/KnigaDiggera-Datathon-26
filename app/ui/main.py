"""Local Streamlit host for the Banking Intelligence review workspace.

Run from the repository root: ``streamlit run app/ui/main.py``.
The bundled component is the only frontend; media never goes through a cloud service.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from app.core.config import settings
from app.ui.reports import ReportError, validate_report

st.set_page_config(page_title="Banking Intelligence · Datathon 26", page_icon="◈", layout="wide")
st.markdown(
    """<style>
    [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {display:none}
    .stApp {background:#0b1120}
    .block-container {padding:0!important;max-width:none!important}
    [data-testid="stMainBlockContainer"] {padding:0!important}
    [data-testid="stVerticalBlock"] {gap:0}
    [data-testid="stMain"] {height:100dvh;overflow:hidden}
    iframe {display:block;border:0;height:100dvh!important}
    </style>""",
    unsafe_allow_html=True,
)

workspace = components.declare_component(
    "banking_intelligence", path=Path(__file__).parent / "frontend"
)
request = workspace(
    budget_seconds=settings.TIMEOUT_SECONDS,
    report_response=st.session_state.get("review_report_response"),
    key="banking_workspace",
    default=None,
)
if isinstance(request, dict) and request.get("id") != st.session_state.get("review_request_id"):
    st.session_state.review_request_id = request.get("id")
    if request.get("action") == "validate_report":
        try:
            report = validate_report(request.get("json"))
            response = {"id": request["id"], "report": report, "error": None}
        except ReportError as exc:
            response = {"id": request["id"], "report": None, "error": str(exc)}
        st.session_state.review_report_response = response
        st.rerun()
