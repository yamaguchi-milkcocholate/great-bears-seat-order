import collections
import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

root_dir = Path(__file__).resolve().parent

image_dir = root_dir / "images"

VOTE_COLUMN_DISPLAY_MAP = {"name": "名前", "love1": "🩷1", "love2": "🩷2", "love3": "🩷3", "datetime": "最終更新"}


def update_votes(conn: GSheetsConnection, name: str, love1: str, love2: str, love3: str) -> pd.DataFrame:
    now_dt = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")

    record = {"name": name, "love1": love1, "love2": love2, "love3": love3, "datetime": now_dt}
    df = conn.read(worksheet="votes", ttl=0)
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    df["sort_key"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    df_last = df.sort_values("sort_key").groupby("name").last().drop(columns=["sort_key"]).reset_index()
    print(df_last)

    df_last = conn.update(worksheet="votes", data=df_last)
    df_last["datetime"] = pd.to_datetime(df_last["datetime"], format="%Y-%m-%d %H:%M:%S")

    df_last.sort_values("name", inplace=True)
    return df_last


def select_recent_votes(conn: GSheetsConnection) -> pd.DataFrame:
    df = conn.read(worksheet="votes", ttl=0)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    df.sort_values("name", inplace=True)
    return df


def check_vote(name: str, love1: str, love2: str, love3: str) -> str:
    # 自分は投票できない
    if name in (love1, love2, love3):
        return "自分には投票できないです..."

    if love1 == love2 or love2 == love3 or love3 == love1:
        return "1~3番目の投票先が重複しています..."

    return ""


def calc_mote_score(df_vote: pd.DataFrame) -> pd.DataFrame:
    df_list = []
    for i in range(3):
        df_tmp = df_vote[[f"love{i + 1}"]].rename(columns={f"love{i + 1}": "name"})
        df_tmp["score"] = 3 - i
        df_list.append(df_tmp)
    df = pd.concat(df_list, ignore_index=True)
    df = df.groupby("name")["score"].sum().reset_index().sort_values("score", ascending=False, ignore_index=True)

    print(df)
    return df


def main() -> None:
    members = ["ﾁｬﾝ", "ジョン", "神崎(は)", "神崎(よ)", "山田", "山口"]

    st.set_page_config(page_title="東京グレートベアーズ", page_icon=image_dir / "icon.png", layout="wide")

    conn = st.connection("gsheets", type=GSheetsConnection)

    st.header("東京グレートベアーズを応援する会")

    st.image(image_dir / "main.png")

    st.subheader("隣になりたい投票箱")

    df_vote = None
    with st.form("vote"):
        st.caption("隣になりたい人を投票してね")

        f_name = st.selectbox("名前", sorted(members))
        f_love1 = st.selectbox("1番目に隣になりたい", sorted(members))
        f_love2 = st.selectbox("2番目に隣になりたい", sorted(members))
        f_love3 = st.selectbox("3番目に隣になりたい", sorted(members))

        submitted = st.form_submit_button("決定")
        if submitted:
            msg = check_vote(name=f_name, love1=f_love1, love2=f_love2, love3=f_love3)
            if msg != "":
                st.warning(msg, icon="⚠️")
            else:
                st.success("投票しました", icon="✅")
                df_vote = update_votes(conn=conn, name=f_name, love1=f_love1, love2=f_love2, love3=f_love3)

    if df_vote is None:
        df_vote = select_recent_votes(conn=conn)

    st.write("投票状況")
    st.dataframe(df_vote.rename(columns=VOTE_COLUMN_DISPLAY_MAP), use_container_width=True)

    st.subheader("モテすこあ")

    df_score = calc_mote_score(df_vote=df_vote)
    st.bar_chart(
        data=df_score,
        x="name",
        y="score",
        x_label="🩷🩷🩷",
        y_label="",
        use_container_width=True,
        horizontal=True,
        color=["#E50A84"],
    )

    st.subheader("席順")
    st.text("comming soon...")


if __name__ == "__main__":
    main()
