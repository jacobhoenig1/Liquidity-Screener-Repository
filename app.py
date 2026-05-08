import streamlit as st

pg = st.navigation([
    st.Page("liquidity_page.py", title="Liquidity", icon="💧", default=True),
    st.Page("pages/1_Volume_Spikes.py", title="Volume Spikes", icon="🔥"),
])
pg.run()
