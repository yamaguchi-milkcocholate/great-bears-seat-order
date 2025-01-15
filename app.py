import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import pulp
import streamlit as st
from streamlit_gsheets import GSheetsConnection

root_dir = Path(__file__).resolve().parent

image_dir = root_dir / "images"

VOTE_COLUMN_DISPLAY_MAP = {"name": "名前", "love1": "🩷1", "love2": "🩷2", "love3": "🩷3", "datetime": "最終更新"}
MEMBERS = sorted(["ﾁｬﾝ", "ジョン", "神崎はるか", "神崎ようへい", "山田", "山口"])
MEMBER2INDEX = {m: i + 1 for i, m in enumerate(MEMBERS)}


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
    if name is None or love1 is None or love2 is None or love3 is None:
        return "選択してから投票してください..."
    # 自分には投票できない
    if name in (love1, love2, love3):
        print(name, love1, love2, love3)
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
    df = df.groupby("name")["score"].sum().reset_index()
    df = pd.DataFrame({"name": MEMBERS}).merge(df, on="name", how="left").fillna(0)
    df.sort_values("score", ascending=False, ignore_index=True, inplace=True)
    return df


def get_seet_order(df_vote: pd.DataFrame) -> pl.DataFrame:
    records = []
    for _, row in df_vote.iterrows():
        records.append(
            {
                "name": MEMBER2INDEX[row["name"]],
                "love1": MEMBER2INDEX[row["love1"]],
                "love2": MEMBER2INDEX[row["love2"]],
                "love3": MEMBER2INDEX[row["love3"]],
            }
        )
    df_vote = pl.DataFrame(records)
    # df_vote = pl.from_pandas(df_vote.drop(columns=["datetime"])).cast({f"love{i}": pl.Int64 for i in range(1, 4)})
    love_matrix = score_love_matrix(df_vote=df_vote)
    _, travel_indexes = solve_love(love_matrix=love_matrix)

    if travel_indexes[0] != 0:
        raise ValueError("開始時点は端点の必要があります")

    seet_orders = []
    for travel_index in travel_indexes[1:]:
        member_index = travel_index - 1
        seet_orders.append(MEMBERS[member_index])
    return seet_orders


def get_love_matrix(df_vote: pl.DataFrame, rank: int, value: float) -> np.ndarray:
    col = f"love{rank}"
    # members + 端点のマトリックスを作成
    p = pl.DataFrame({"source": [0] + list(MEMBER2INDEX.values())}).join(
        pl.DataFrame({"target": [0] + list(MEMBER2INDEX.values())}), how="cross"
    )
    p = p.join(
        df_vote.with_columns(pl.lit(value).alias("value")).select(["name", col, "value"]),
        left_on=["source", "target"],
        right_on=["name", col],
        how="left",
    )
    p = p.fill_null(0)
    p = p.pivot(on="target", index="source", values="value", aggregate_function="max")

    p = p.select(pl.exclude("source")).to_numpy()
    return p


def score_love_matrix(df_vote: pl.DataFrame) -> np.ndarray:
    lm1 = get_love_matrix(df_vote=df_vote, rank=1, value=3)
    lm2 = get_love_matrix(df_vote=df_vote, rank=2, value=2)
    lm3 = get_love_matrix(df_vote=df_vote, rank=3, value=1)

    # 相思相愛
    love_matrix_list = [lm1, lm2, lm3]
    mutual_love_matrix = np.zeros_like(lm1, dtype=float)
    for i_m in range(len(love_matrix_list)):
        for j_m in range(i_m, len(love_matrix_list)):
            m1 = love_matrix_list[i_m]
            m2 = love_matrix_list[j_m]
            mutual_love_matrix += m1 * m2.T

    # 片想いは相思相愛よりスコアが高くならない(1番高くて3/6=0.5)
    oneside_love_matrix = (lm1 + lm2 + lm3) * (mutual_love_matrix == 0)
    oneside_love_matrix = (oneside_love_matrix + oneside_love_matrix.T) / 6

    love_matrix = mutual_love_matrix + oneside_love_matrix
    return love_matrix


def solve_love(love_matrix: np.ndarray) -> None:
    # 変数を定義
    num_source, num_target = love_matrix.shape
    X = []
    for i in range(num_source):
        i_X = []
        for j in range(num_target):
            x_ij = pulp.LpVariable(f"x_{i}{j}", cat="Binary")
            i_X.append(x_ij)
        X.append(i_X)

    # 訪問順
    U = []
    for i in range(num_source):
        U.append(pulp.LpVariable(f"u_{i}", cat="Integer", lowBound=0, upBound=num_source - 1))

    # 定式化
    problem = pulp.LpProblem("love-seet-order", pulp.LpMaximize)

    # 目的関数を定義
    objectives = []
    for i in range(num_source):
        for j in range(num_target):
            objectives.append(X[i][j] * love_matrix[i][j])
    problem += (pulp.lpSum(objectives), "目的関数")

    # 制約条件を定義
    # 次に隣になる人を決めていくイメージ
    # 右隣の人は一人
    for i in range(num_source):
        constraints = []
        for j in range(num_target):
            constraints.append(X[i][j])
        problem += (pulp.lpSum(constraints) == 1, f"右隣の人は一人_{i}")

    # 左隣の人は一人
    for j in range(num_target):
        constraints = []
        for i in range(num_source):
            constraints.append(X[i][j])
        problem += (pulp.lpSum(constraints) == 1, f"左隣の人は一人_{j}")

    # 自分は隣にならない
    for i in range(num_source):
        problem += (X[i][i] == 0, f"自分は隣にならない_{i}")

    # 部分巡回路にならない
    for i in range(num_source):
        for j in range(num_target):
            if i != j and j != 0:
                problem += (U[i] - U[j] <= num_source * (1 - X[i][j]) - 1, f"部分経路禁止_{i}_{j}")

    problem.solve()

    Y = np.zeros_like(love_matrix)
    U = np.zeros(len(love_matrix), dtype=int)
    for v in problem.variables():
        if v.name.startswith("x_"):
            ij = v.name.replace("x_", "")
            i, j = int(ij[0]), int(ij[1])
            Y[i][j] = int(v.varValue)
        if v.name.startswith("u_"):
            i = int(v.name.replace("u_", ""))
            U[i] = int(v.varValue)

    return Y, U


def main() -> None:
    st.set_page_config(page_title="東京グレートベアーズを応援する会", page_icon=image_dir / "icon.png", layout="wide")

    conn = st.connection("gsheets", type=GSheetsConnection)

    st.header("東京グレートベアーズを応援する会")

    st.image(image_dir / "main.png")

    st.subheader("隣になりたい投票")

    df_vote = None
    with st.form("vote"):
        st.caption("隣になりたい人を投票してね")

        f_name = st.selectbox("自分の名前", MEMBERS, index=None, placeholder="自分の名前を選んでください")
        f_love1 = st.selectbox(
            "1番目に隣になりたい人", MEMBERS, index=None, placeholder="隣になりたい人を選んでください"
        )
        f_love2 = st.selectbox(
            "2番目に隣になりたい人", MEMBERS, index=None, placeholder="隣になりたい人を選んでください"
        )
        f_love3 = st.selectbox(
            "3番目に隣になりたい人", MEMBERS, index=None, placeholder="隣になりたい人を選んでください"
        )

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

    st.subheader("投票状況")
    st.dataframe(df_vote.rename(columns=VOTE_COLUMN_DISPLAY_MAP), use_container_width=True)

    df_score = calc_mote_score(df_vote=df_vote)
    fig = px.bar(df_score, y="score", x="name", text_auto=".2s", title="🩷🩷🩷")
    fig.update_traces(
        textfont_size=12,
        textangle=0,
        textposition="outside",
        cliponaxis=False,
        marker_color="#EB2AA4",
        marker_line_color="#E534AC",
    )
    fig.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("席順")
    st.text("testing...")

    seet_order = get_seet_order(df_vote=df_vote)
    df_seet = pl.DataFrame({"席順": seet_order})
    st.dataframe(df_seet)


if __name__ == "__main__":
    main()
