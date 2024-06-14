from collections import defaultdict

import numpy as np
import pandas as pd

N_SAMPLES = 100
N_TMSP = 50
SAMPLE_IDS = [f"sample_{i}" for i in range(N_SAMPLES)]
CHANNELS = [
    "01TRRILE01WSDS",
    "01TRRILE02WSDS",
    "01TRRILE03WSDS",
    "01THSP0400WSAC",
    "01THSP1200WSAC",
    "01TIMPLE00WSAC",
    "01THSP0400WSVE",
    "01THSP1200WSVE",
    "01TIMPLE00WSVE",
    "01THSP0400WSDS",
    "01THSP1200WSDS",
    "01TIMPLE00WSDS",
]
N_CHANNELS = len(CHANNELS)
RNG = np.random.default_rng(42)


def generate_expert_ratings():
    labels = {1: "Good", 2: "Acceptable", 3: "Marginal", 4: "Poor"}
    sub_ratings = ("rating_impactor", "rating_rib_compression", "rating_T04", "rating_T12")
    sub_ratings_num = [f"{sub_rating}_num" for sub_rating in sub_ratings]
    rating = {}

    # sub ratings
    for j, sub_rating in enumerate(sub_ratings):
        rating[sub_ratings_num[j]] = RNG.random(N_SAMPLES) * 3 + 1  # [1:4]
        rating[sub_rating] = [labels[int(i)] for i in rating[sub_ratings_num[j]]]

    # calculate ratings
    total_nums = np.zeros(N_SAMPLES)
    for sub_rating in sub_ratings_num:
        total_nums += rating[sub_rating]
    total_nums /= len(sub_ratings)

    # store total ratings
    rating["rating_calculated"] = total_nums
    rating["rating_total_num"] = total_nums.astype(int)
    rating["rating_total"] = [labels[i] for i in rating["rating_total_num"]]

    # to dataframe
    df = pd.DataFrame(rating, index=SAMPLE_IDS)
    print(f"Expert ratings shape {df.shape} generated")
    print(df.head())

    # store
    f_path = "data/raw/ratings_experts.csv"
    df.to_csv(f_path)
    print(f"Expert ratings stored in {f_path}")


def generate_iso_ratings():
    labels = {1: "Excellent", 2: "Good", 3: "Fair", 4: "Poor"}

    ratings = defaultdict(list)

    for channel in CHANNELS:
        # generate sub ratings
        sub_ratings = ("Corridor Rating", "Phase Rating", "Magnitude Rating", "Slope Rating")
        rating = {sr: RNG.random(N_SAMPLES) for sr in sub_ratings}  # [0:1]

        # generate total rating
        total_rating = np.zeros(N_SAMPLES)
        for sub_rating in sub_ratings:
            total_rating += rating[sub_rating]
        total_rating /= len(sub_ratings)
        rating["Total"] = total_rating  # [0:1]

        # conversion to FAKE ISO 18571
        rating["ISO 18571 Rank"] = (total_rating * 3 + 1).astype(int)  # [1:4]
        rating["ISO 18571 Label"] = [labels[i] for i in rating["ISO 18571 Rank"]]

        # store
        for rt in rating.keys():
            ratings["Channel"].extend([channel] * N_SAMPLES)
            ratings["Rating"].extend([rt] * N_SAMPLES)
            ratings["Value"].extend(rating[rt])
            ratings["SimID"].extend(SAMPLE_IDS)

    # to dataframe
    df = pd.DataFrame(ratings)
    print(f"ISO ratings shape {df.shape} generated")
    print(df.head())

    # store
    f_path = "data/raw/ratings_iso18571.csv"
    df.to_csv(f_path)
    print(f"ISO ratings stored in {f_path}")


def generate_simulation_data():
    data = defaultdict(list)
    tmsps = np.linspace(0, 80, N_TMSP)
    for sid in SAMPLE_IDS:
        for channel in CHANNELS:
            data["rid"].extend([sid] * N_TMSP)
            data["Channel"].extend([channel] * N_TMSP)
            data["Signal"].extend(RNG.random(N_TMSP))
            data["Time"].extend(tmsps)

    # to dataframe
    df = pd.DataFrame(data)
    df.set_index("rid", inplace=True)
    print(f"Simulation Data shape {df.shape} generated")
    print(df.head())

    # store
    f_path = "data/raw/signals_cae.csv"
    df.to_csv(f_path)
    print(f"Simulation Data ratings stored in {f_path}")


if __name__ == "__main__":
    print("Generating random data")

    print("Generating expert ratings")
    generate_expert_ratings()

    print("Generating ISO ratings")
    generate_iso_ratings()

    print("Generating simulation data")
    generate_simulation_data()
