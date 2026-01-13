import numpy as np
import pandas as pd
import joblib
from pathlib import Path

class HybridRecommender:
    def __init__(self):
        """
        Loads all required models and data using absolute paths.
        This is FastAPI-safe and production-safe.
        """

        # project root = content_recommendation_system/
        BASE_DIR = Path(__file__).resolve().parents[1]

        MODELS_DIR = BASE_DIR / "models"
        DATA_DIR = BASE_DIR / "data"

        # Models
        self.als_model = joblib.load(MODELS_DIR / "als_tuned_model.pkl")
        self.movie_encoder = joblib.load(MODELS_DIR / "movie_encoder.pkl")
        self.user_encoder = joblib.load(MODELS_DIR / "user_encoder.pkl")

        # Data
        self.user_item_matrix = joblib.load(
            DATA_DIR / "processed" / "item_user_train.pkl"
        )

        # Movies metadata
        self.movies = pd.read_csv(
            DATA_DIR / "raw" / "movies.dat",
            sep="::",
            engine="python",
            encoding="latin-1",
            names=["movie_id", "title", "genres"],
        )

        self.movies["genres"] = self.movies["genres"].str.split("|")

        # Build genre matrix
        self._build_genre_matrix()

    def _build_genre_matrix(self):
        movie_id_to_idx = {
            movie_id: idx
            for idx, movie_id in enumerate(self.movie_encoder.classes_)
        }

        self.movies = self.movies[
            self.movies["movie_id"].isin(movie_id_to_idx)
        ].copy()

        self.movies["movie_idx"] = self.movies["movie_id"].map(movie_id_to_idx)

        all_genres = sorted({g for gs in self.movies["genres"] for g in gs})
        self.genre_to_idx = {g: i for i, g in enumerate(all_genres)}

        n_items = self.user_item_matrix.shape[1]
        self.genre_matrix = np.zeros((n_items, len(all_genres)))

        for _, row in self.movies.iterrows():
            for g in row["genres"]:
                self.genre_matrix[row["movie_idx"], self.genre_to_idx[g]] = 1

    def recommend(
        self,
        raw_user_id: int,
        k: int = 10,
        alpha: float = 0.7,
    ):
        """
        Returns top-k recommended movies for a user
        """

        if raw_user_id not in self.user_encoder.classes_:
            raise ValueError("Unknown user_id")

        user_idx = self.user_encoder.transform([raw_user_id])[0]
        user_items = self.user_item_matrix[user_idx]

        # ALS recommendations
        item_idxs, als_scores = self.als_model.recommend(
            userid=user_idx,
            user_items=user_items,
            N=k,
            filter_already_liked_items=True,
        )

        # Genre-based re-ranking
        liked_items = user_items.indices
        if len(liked_items) > 0:
            user_profile = self.genre_matrix[liked_items].mean(axis=0)
            genre_scores = self.genre_matrix[item_idxs] @ user_profile
        else:
            genre_scores = np.zeros_like(als_scores)

        # Normalize
        als_scores = (als_scores - als_scores.min()) / (als_scores.ptp() + 1e-8)
        genre_scores = (genre_scores - genre_scores.min()) / (
            genre_scores.ptp() + 1e-8
        )

        final_scores = alpha * als_scores + (1 - alpha) * genre_scores
        ranked_items = item_idxs[np.argsort(-final_scores)]

        movie_ids = self.movie_encoder.inverse_transform(ranked_items)

        return (
            self.movies[self.movies["movie_id"].isin(movie_ids)]
            .sort_values("movie_id")
            .head(k)[["movie_id", "title", "genres"]]
        )
