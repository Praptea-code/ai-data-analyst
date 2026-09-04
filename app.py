import sys

import streamlit as st

sys.stdout.reconfigure(encoding="utf-8")

st.set_page_config(page_title="AI Data Analyst", layout="wide")

from charts import spec_to_figure
from graph import run_investigation


st.title("AI Data Analyst")
st.caption("Ask a question about the sales database. The agent writes SQL, investigates, builds charts, and answers with evidence.")

DEFAULT_QUESTION = "Why did revenue fall in August 2026?"

with st.form("question_form"):
    question = st.text_input("Question", value=DEFAULT_QUESTION, placeholder="Ask anything about the sales data...")
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted and question.strip():
    with st.spinner("Investigating the database (this may take a minute)..."):
        try:
            state = run_investigation(question.strip())
        except Exception as e:  # noqa: BLE001 - surface rate-limit / API errors in the UI
            st.error(f"The investigation failed. This is often a Groq rate limit (free tier). Try again in a moment.\n\nDetail: {e}")
            st.stop()

    answer = state.get("final_answer", {})

    st.subheader("Executive Summary")
    st.markdown(answer.get("executive_summary") or "_No summary available._")

    st.divider()
    st.subheader("Main Cause")
    st.markdown(answer.get("main_cause") or "_Not determined._")

    st.divider()
    st.subheader("Key Drivers")
    drivers = answer.get("key_drivers") or []
    if drivers:
        for d in drivers:
            st.markdown(f"- {d}")
    else:
        st.markdown("_None identified._")

    st.divider()
    st.subheader("Evidence with Charts")
    charts = state.get("charts") or []
    if charts:
        cols = st.columns(2)
        for i, spec in enumerate(charts):
            try:
                fig = spec_to_figure(spec)
                title = (spec.get("title") or f"chart_{i}").replace(" ", "_").lower()
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}_{title}")
                    st.caption(spec.get("subtitle", ""))
            except Exception as e:  # noqa: BLE001
                st.caption(f"Could not render chart: {e}")
    else:
        st.markdown("_No charts produced._")

    evidence = answer.get("evidence") or []
    if evidence:
        st.markdown("**Evidence statements**")
        for e in evidence:
            st.markdown(f"- {e}")

    st.divider()
    st.subheader("Recommended Investigation")
    recs = answer.get("recommended_investigation") or []
    if recs:
        for r in recs:
            st.markdown(f"- {r}")
    else:
        st.markdown("_None._")

    with st.expander("View raw investigation trace (queries + observations)"):
        st.markdown("**Investigation plan**")
        for i, step in enumerate(state.get("plan", []), 1):
            st.markdown(f"{i}. {step}")
        st.markdown("**Queries executed**")
        for q in state.get("queries", []):
            st.code(q, language="sql")
        st.markdown("**Observations**")
        for obs in state.get("observations", []):
            st.markdown(obs)
            st.markdown("---")
else:
    st.info("Enter a question and click **Analyze** to start. Ensure `data/sales.db` exists (run `python database.py` first).")
