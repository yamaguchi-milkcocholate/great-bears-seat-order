import random

import numpy as np
import polars as pl
import pulp

members = ["a", "b", "c", "d", "e", "f"]
member2index = {m: i + 1 for i, m in enumerate(members)}


def get_random_vote() -> pl.DataFrame:
    random.seed(43)
    records = []
    for member in members:
        others = [m for m in members if m != member]
        votes = random.sample(others, 3)
        records.append(
            {
                "name": member2index[member],
                "l1": member2index[votes[0]],
                "l2": member2index[votes[1]],
                "l3": member2index[votes[2]],
            }
        )
    return pl.DataFrame(records)


def get_love_matrix(df_vote: pl.DataFrame, rank: int, value: float) -> np.ndarray:
    col = f"l{rank}"
    # members + 端点のマトリックスを作成
    p = pl.DataFrame({"source": [0] + list(member2index.values())}).join(
        pl.DataFrame({"target": [0] + list(member2index.values())}), how="cross"
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


def get_seet_order(travel_indexes: list[int]) -> list[str]:
    seet_orders = []
    for travel_index in travel_indexes[1:]:
        member_index = travel_index - 1
        seet_orders.append(members[member_index])
    return seet_orders


def main() -> None:
    df_vote = get_random_vote()
    love_matrix = score_love_matrix(df_vote=df_vote)
    print(df_vote)
    print(love_matrix.round(2))

    adjacency_matrix, travel_indexes = solve_love(love_matrix=love_matrix)
    print(adjacency_matrix)
    print(travel_indexes)

    seet_orders = get_seet_order(travel_indexes=travel_indexes)

    print(seet_orders)


if __name__ == "__main__":
    main()
