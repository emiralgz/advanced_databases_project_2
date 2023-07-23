import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time


def _preprocess_data(file: str) -> dict:
    if file == "100k":
        # Read in data
        raw_data_path = Path("../advanced_databases/100k.txt")
        raw_data = pd.read_csv(
            raw_data_path,
            sep="\t",
            header=None,
            names=["subject", "property", "object"],
        )
        properties = list(
            raw_data["property"].apply(lambda x: x.split(":")[1]).unique()
        )
        # Create a dictionary of dataframes, one for each property
        data = {}
        for prop in tqdm(properties, desc="Preprocessing data"):
            prop = prop.split(":")[0]
            data[prop] = raw_data[raw_data["property"].str.contains(prop)].drop(
                columns=["property"]
            )
            data[prop] = data[prop].apply(
                lambda row: pd.Series(
                    {
                        "subject": row[0].split(":")[-1],
                        "object": row[1].split(":")[-1].replace(" .", ""),
                    }
                ),
                axis=1,
            )
    elif file == "10m":
        # Read in data
        raw_data_path = Path("../advanced_databases/watdiv.10M.nt")
        raw_data = pd.read_csv(
            raw_data_path,
            sep="\t",
            header=None,
            names=["subject", "property", "object"],
        )
        properties = list(
            raw_data["property"]
            .apply(
                lambda x: x.split("/")[-1].split(">")[0]
                if "#" not in x.split("/")[-1].split(">")[0]
                else x.split("/")[-1].split(">")[0].split("#")[1]
            )
            .unique()
        )
        # Create a dictionary of dataframes, one for each property
        data = {}
        for prop in tqdm(properties, desc="Preprocessing data"):
            data[prop] = raw_data[raw_data["property"].str.contains(prop)].drop(
                columns=["property"]
            )
            data[prop] = data[prop].apply(
                lambda row: pd.Series(
                    {
                        "subject": row[0].split("/")[-1].split(">")[0],
                        "object": row[1].split("/")[-1].split(">")[0],
                    }
                ),
                axis=1,
            )
        print("xde")
    else:
        print("Choose a valid file: 10m or 100k")
    return data


def _hash_join(data: dict) -> pd.DataFrame:
    # Create hash tables
    hash_table_follows = {}
    hash_table_friendOf = {}
    hash_table_likes = {}
    hash_table_hasReview = {}
    # Build hash table for follows
    for row in tqdm(data["follows"].iterrows(), desc="Building hash table for follows"):
        row = row[1]
        subject_val = row["subject"]
        object_val = row["object"]
        if subject_val not in hash_table_follows:
            hash_table_follows[subject_val] = []
        hash_table_follows[subject_val].append(object_val)

    # Build hash table for friendOf
    for row in tqdm(
        data["friendOf"].iterrows(), desc="Building hash table for friendOf"
    ):
        row = row[1]
        subject_val = row["subject"]
        object_val = row["object"]
        if subject_val not in hash_table_friendOf:
            hash_table_friendOf[subject_val] = []
        hash_table_friendOf[subject_val].append(object_val)

    # Build hash table for likes
    for row in tqdm(data["likes"].iterrows(), desc="Building hash table for likes"):
        row = row[1]
        subject_val = row["subject"]
        object_val = row["object"]
        if subject_val not in hash_table_likes:
            hash_table_likes[subject_val] = []
        hash_table_likes[subject_val].append(object_val)

    for row in tqdm(
        data["hasReview"].iterrows(), desc="Building hash table for hasReview"
    ):
        row = row[1]
        subject_val = row["subject"]
        object_val = row["object"]
        if subject_val not in hash_table_hasReview:
            hash_table_hasReview[subject_val] = []
        hash_table_hasReview[subject_val].append(object_val)

    # Perform hash join
    result = []
    for fkey, fvalue in tqdm(hash_table_follows.items(), desc="Performing hash join"):
        follows_subject = fkey
        for follows_object in fvalue:
            for fokey, fovalue in hash_table_friendOf.items():
                if fokey != follows_object:
                    continue
                for friendOf_object in fovalue:
                    for lkey, lvalue in hash_table_likes.items():
                        if lkey != friendOf_object:
                            continue
                        for likes_object in lvalue:
                            for hrkey, hrvalue in hash_table_hasReview.items():
                                if fokey != follows_object:
                                    continue
                                for hasReview_object in hrvalue:
                                    if (
                                        hrkey == likes_object
                                        and lkey == friendOf_object
                                        and fokey == follows_object
                                    ):
                                        result.append(
                                            {
                                                "follows.subject": follows_subject,
                                                "follows.object": follows_object,
                                                "friendOf.object": friendOf_object,
                                                "likes.object": likes_object,
                                                "hasReview.object": hasReview_object,
                                            }
                                        )

    return pd.DataFrame(result)


def _hash_join_improved(data: dict) -> pd.DataFrame:
    # Create hash tables
    hash_table_follows = {}
    hash_table_friendOf = {}
    hash_table_likes = {}
    hash_table_hasReview = {}
    # Build hash table for follows
    # Group the DataFrame by the "subject" column and iterate through the groups
    for subject_val, group in tqdm(
        data["follows"].groupby("subject"), desc="Building hash table for follows"
    ):
        object_values = group["object"].tolist()
        hash_table_follows[subject_val] = object_values

    # Build hash table for friendOf
    for subject_val, group in tqdm(
        data["friendOf"].groupby("subject"), desc="Building hash table for friendOf"
    ):
        object_values = group["object"].tolist()
        hash_table_friendOf[subject_val] = object_values

    # Build hash table for likes
    for subject_val, group in tqdm(
        data["likes"].groupby("subject"), desc="Building hash table for likes"
    ):
        object_values = group["object"].tolist()
        hash_table_likes[subject_val] = object_values

    # Build hash table for hasReview
    for subject_val, group in tqdm(
        data["hasReview"].groupby("subject"), desc="Building hash table for hasReview"
    ):
        object_values = group["object"].tolist()
        hash_table_follows[subject_val] = object_values

    # Perform hash join
    result = []
    for fkey, fvalue in tqdm(hash_table_follows.items(), desc="Performing hash join"):
        follows_subject = fkey
        for follows_object in fvalue:
            if follows_object not in hash_table_friendOf:
                continue
            for friendOf_object in hash_table_friendOf[follows_object]:
                if friendOf_object not in hash_table_likes:
                    continue
                for likes_object in hash_table_likes[friendOf_object]:
                    if likes_object not in hash_table_hasReview:
                        continue
                    for hasReview_object in hash_table_hasReview[likes_object]:
                        result.append(
                            {
                                "follows.subject": follows_subject,
                                "follows.object": follows_object,
                                "friendOf.object": friendOf_object,
                                "likes.object": likes_object,
                                "hasReview.object": hasReview_object,
                            }
                        )

    return pd.DataFrame(result)


def _sort_merge_join(data: dict) -> pd.DataFrame:
    # Get dataframes as list of tuples
    follows_list = [tuple(row) for row in data["follows"][["subject", "object"]].values]
    friendOf_list = [
        tuple(row) for row in data["friendOf"][["subject", "object"]].values
    ]
    likes_list = [tuple(row) for row in data["likes"][["subject", "object"]].values]
    hasReview_list = [
        tuple(row) for row in data["hasReview"][["subject", "object"]].values
    ]

    # Define the sort-merge join function
    def merge(left, right):
        # Sort lists
        left.sort(key=lambda x: x[-1])
        right.sort(key=lambda x: x[0])
        merged = []
        i, j = 0, 0
        pbar = tqdm(total=len(left), desc="Performing sort-merge join")
        while i < len(left) and j < len(right):
            if left[i][-1] == right[j][0]:
                merged.append(left[i] + right[j])
                j += 1
            elif left[i][-1] < right[j][0]:
                i += 1
                j = 0
                pbar.update(1)
            else:
                j += 1
        pbar.close()
        return merged

    # Perform the sort-merge join for the dataframes
    result = merge(follows_list, friendOf_list)
    result = merge(result, likes_list)
    result = merge(result, hasReview_list)
    result = [
        {
            "follows.subject": row[0],
            "follows.object": row[1],
            "friendOf.object": row[3],
            "likes.object": row[5],
            "hasReview.object": row[7],
        }
        for row in result
    ]
    return pd.DataFrame(
        result,
    )


def main(file: str):
    start_time = time.time()
    preprocessed_data = _preprocess_data(file)
    preprocessing_time = time.time() - start_time
    hash_join_result = _hash_join(preprocessed_data)
    hash_join_time = time.time() - preprocessing_time - start_time
    sort_merge_join_result = _sort_merge_join(preprocessed_data)
    sort_merge_join_time = time.time() - hash_join_time - start_time
    hash_join_improved_result = _hash_join_improved(preprocessed_data)
    hash_join_improved_time = time.time() - sort_merge_join_time - start_time
    print(
        f"Preprocessing time: {preprocessing_time}\n Hash join time: {hash_join_time}\n Sort-merge join time: {sort_merge_join_time}\n Improved hash join time: {hash_join_improved_time}"
    )


if __name__ == "__main__":
    main("10m")
    main("100k")
