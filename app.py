import collections
from pathlib import Path

import pandas as pd
import streamlit as st

root_dir = Path(__file__).resolve().parent

image_dir = root_dir / "images"

members = ["ﾁｬﾝ", "ジョン", "神崎(は)", "神崎(よ)", "山田", "山口"]

df_member = pd.DataFrame(
    {
        "名前": ["ﾁｬﾝ", "ジョン", "神崎(は)", "神崎(よ)", "山田", "山口"],
        "🩷1": members[1:] + members[:1],
        "🩷2": members[2:] + members[:2],
        "🩷3": members[3:] + members[:3],
    }
)

selected_members = (
    df_member["🩷1"].tolist() + df_member["🩷2"].tolist() + df_member["🩷3"].tolist()
)
counter = collections.Counter(selected_members)
df_mote = [
    {"名前": member, "モテ度": num_selected} for member, num_selected in counter.items()
]
print(df_mote)


st.set_page_config(
    page_title="東京グレートベアーズ", page_icon=image_dir / "icon.png", layout="wide"
)

st.header("東京グレートベアーズを応援する会")

st.image(image_dir / "main.png")

st.subheader("この人と隣になりたい(工事中)")

st.dataframe(df_member, use_container_width=True)

st.subheader("モテ度")

st.bar_chart(data=df_mote, x="名前", y="モテ度", use_container_width=True)


st.subheader("席順")
st.text("comming soon...")
