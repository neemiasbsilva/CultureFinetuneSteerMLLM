"""BERTopic topic modeling on cultural annotation captions and justifications."""

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

BERTOPIC_MIN_CLUSTER_SIZE = 80
BERTOPIC_N_COMPONENTS = 5
BERTOPIC_N_NEIGHBORS = 15
BERTOPIC_MIN_DF = 10


def fit_bertopic(
    texts: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int = BERTOPIC_MIN_CLUSTER_SIZE,
    n_components: int = BERTOPIC_N_COMPONENTS,
    n_neighbors: int = BERTOPIC_N_NEIGHBORS,
    min_df: int = BERTOPIC_MIN_DF,
    seed: int = 42,
) -> tuple[BERTopic, list[int], np.ndarray | None]:
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=seed,
        metric="cosine",
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(min_df=min_df, stop_words="english")
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
    return topic_model, topics, probs


def topic_sentiment_composition(
    df: pd.DataFrame,
    topics: list[int],
    sentiment_col: str = "predicted_sentiment",
) -> pd.DataFrame:
    df = df.copy()
    df["topic"] = topics
    index = [df["condition"], df["topic"]] if "condition" in df.columns else df["topic"]
    cross = pd.crosstab(index, df[sentiment_col], normalize="index")
    return cross


def topic_culture_proportions(
    df: pd.DataFrame,
    topics: list[int],
    culture_col: str = "culture",
) -> pd.DataFrame:
    df = df.copy()
    df["topic"] = topics
    columns = [df["condition"], df[culture_col]] if "condition" in df.columns else df[culture_col]
    cross = pd.crosstab(df["topic"], columns, normalize="columns")
    return cross


def save_topic_info(model: BERTopic, output_path: str, n_words: int = 10) -> pd.DataFrame:
    info: pd.DataFrame = model.get_topic_info()
    info.to_csv(output_path, index=False)
    return info
