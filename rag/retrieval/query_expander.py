from typing import List, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag.models.assistant_config import AssistantConfig
from rag.ingestion.normalizer import normalize_uzbek


class QueryExpander:
    """
    Uzbek-aware query expander with domain-specific synonym expansion.

    Features:
    - Normalizes queries using UzbekNormalizer
    - Expands queries with domain-specific Uzbek synonyms
    - Supports HR, Legal, and Finance/Banking domains
    - Returns multiple query variants for improved recall
    """

    def __init__(self):
        """Initialize query expander with domain-specific synonym mappings."""
        # Domain-specific synonym dictionaries
        # Structure: {domain: {base_term: [synonyms...]}}
        self.domain_synonyms = {
            "HR": {
                "ta'til": ["otpusk", "mehnat ta'tili"],
                "ish vaqti": ["grafik", "ish jadvali"],
                "maosh": ["ish haqi", "oylik"],
                "xodim": ["ishchi", "hodim"],
                "ishga qabul": ["yollash", "ishga olish"],
            },
            "Human Resources": {  # Alias for HR
                "ta'til": ["otpusk", "mehnat ta'tili"],
                "ish vaqti": ["grafik", "ish jadvali"],
                "maosh": ["ish haqi", "oylik"],
                "xodim": ["ishchi", "hodim"],
                "ishga qabul": ["yollash", "ishga olish"],
            },
            "Legal": {
                "shartnoma": ["kontrakt", "kelishuv"],
                "majburiyat": ["obligatsiya", "burch"],
                "huquq": ["haq", "qonuniy haq"],
                "qonun": ["qoidalar", "nizom"],
                "javobgarlik": ["mas'uliyat", "zimma"],
            },
            "Finance": {
                "kredit": ["qarz", "ssuda"],
                "foiz stavkasi": ["stavka", "protsent"],
                "to'lov": ["haq", "pul"],
                "hisob": ["account", "schet"],
                "byudjet": ["moliya rejasi", "smeta"],
            },
            "Bank": {  # Alias for Finance
                "kredit": ["qarz", "ssuda"],
                "foiz stavkasi": ["stavka", "protsent"],
                "to'lov": ["haq", "pul"],
                "hisob": ["account", "schet"],
                "byudjet": ["moliya rejasi", "smeta"],
            },
            "Banking": {  # Another alias
                "kredit": ["qarz", "ssuda"],
                "foiz stavkasi": ["stavka", "protsent"],
                "to'lov": ["haq", "pul"],
                "hisob": ["account", "schet"],
                "byudjet": ["moliya rejasi", "smeta"],
            },
        }

    def expand(self, query: str, config: AssistantConfig) -> List[str]:
        """
        Expand query with domain-specific synonyms.

        Process:
        1. Normalize the original query
        2. Check if config.domain has relevant synonyms
        3. For each term in the domain dictionary:
           - If term appears in query, generate variant with synonym
        4. Return unique queries: [normalized_query, ...expanded_variants]

        Args:
            query: Raw user query
            config: AssistantConfig with domain information

        Returns:
            List of query variants, with normalized query first
        """
        # Step 1: Normalize the query
        normalized_query = normalize_uzbek(query)

        if not normalized_query:
            return [query]  # Fallback to original if normalization fails

        # Step 2: Check if we have synonyms for this domain
        domain = config.domain
        if domain not in self.domain_synonyms:
            # No domain-specific expansion available
            return [normalized_query]

        # Step 3: Generate expanded variants
        expanded_variants = []
        synonym_dict = self.domain_synonyms[domain]

        # Normalize the base terms for matching
        normalized_terms = {
            normalize_uzbek(base_term): synonyms
            for base_term, synonyms in synonym_dict.items()
        }

        # For each base term, check if it appears in the query
        for base_term_normalized, synonyms in normalized_terms.items():
            if base_term_normalized in normalized_query:
                # Generate a variant for each synonym
                for synonym in synonyms:
                    # Normalize the synonym
                    synonym_normalized = normalize_uzbek(synonym)
                    # Create variant by replacing base term with synonym
                    variant = normalized_query.replace(
                        base_term_normalized,
                        synonym_normalized
                    )
                    if variant != normalized_query and variant not in expanded_variants:
                        expanded_variants.append(variant)

        # Step 4: Return unique queries in order
        # Always start with the normalized original query
        result = [normalized_query]

        # Add expanded variants
        result.extend(expanded_variants)

        return result

    def add_domain_synonyms(self, domain: str, synonyms: Dict[str, List[str]]):
        """
        Add or update synonym mappings for a specific domain.

        Args:
            domain: Domain name (e.g., "Legal", "HR")
            synonyms: Dictionary of {base_term: [synonyms...]}
        """
        if domain in self.domain_synonyms:
            # Update existing domain
            self.domain_synonyms[domain].update(synonyms)
        else:
            # Add new domain
            self.domain_synonyms[domain] = synonyms
