from typing import List
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag.models.assistant_config import AssistantConfig


def build_qdrant_filter(config: AssistantConfig) -> Filter:
    """
    Build Qdrant filter from AssistantConfig.

    Filters applied:
    - org_id must match config.org_id
    - language must match config.primary_language
    - assistant_ids must contain config.id (in should clause)
    - tags must contain any of config.allowed_tags (if provided, in should clause)

    Args:
        config: AssistantConfig with filtering criteria

    Returns:
        Qdrant Filter object ready for search queries
    """
    # Build must conditions (required filters)
    must_conditions = [
        # Filter by organization
        FieldCondition(
            key="org_id",
            match=MatchValue(value=config.org_id)
        ),
        # Filter by primary language
        FieldCondition(
            key="language",
            match=MatchValue(value=config.primary_language)
        )
    ]

    # Build should conditions (at least one must match)
    should_conditions = []

    # Filter by assistant_ids - chunks must be associated with this assistant
    should_conditions.append(
        FieldCondition(
            key="assistant_ids",
            match=MatchAny(any=[config.id])
        )
    )

    # Filter by allowed tags if specified
    if config.allowed_tags:
        should_conditions.append(
            FieldCondition(
                key="tags",
                match=MatchAny(any=config.allowed_tags)
            )
        )

    # Build final filter
    # Note: If both assistant_ids and tags are in should, chunks matching EITHER will be returned
    # This is correct behavior - we want chunks that are either:
    # 1. Explicitly attached to this assistant, OR
    # 2. Tagged with one of the allowed tags
    filter_obj = Filter(
        must=must_conditions,
        should=should_conditions if should_conditions else None
    )

    return filter_obj
