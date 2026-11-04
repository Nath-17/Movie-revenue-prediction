from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FeatureSpec:
    """Describe a source column and its association measure."""

    column: str
    label: str
    kind: Literal["numeric", "categorical"]
    min_group_size: int = 1


FEATURE_REGISTRY = (
    FeatureSpec("budget_known", "Budget", "numeric"),
    FeatureSpec("popularity_score", "Popularity", "numeric"),
    FeatureSpec("length", "Runtime", "numeric"),
    FeatureSpec("release_year", "Release year", "numeric"),
    FeatureSpec("is_franchise", "Franchise", "categorical"),
    FeatureSpec("primary_genre", "Primary genre", "categorical", 30),
    FeatureSpec("language", "Language", "categorical", 30),
    FeatureSpec("main_country", "Main production country", "categorical", 30),
)
