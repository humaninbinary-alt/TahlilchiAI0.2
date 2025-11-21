from typing import List, Optional, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag.models.assistant_config import AssistantConfig
from rag.models.chunk import Chunk


class PromptBuilder:
    """
    Adaptive prompt builder that creates context-aware prompts based on AssistantConfig.

    Features:
    - Configurable system instructions from AssistantConfig
    - Context integration from retrieved chunks
    - Chat history support
    - Language-aware prompts
    - Sensitivity-aware guardrails
    """

    def build_prompt(
        self,
        query: str,
        config: AssistantConfig,
        chunks: List[Chunk],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Build a complete prompt for LLM generation.

        Structure:
        1. System instructions (based on AssistantConfig)
        2. Context snippets (from chunks)
        3. Chat history (optional)
        4. User query
        5. Generation instructions

        Args:
            query: User's current question
            config: AssistantConfig with domain, tone, style, etc.
            chunks: Retrieved context chunks
            history: Optional chat history as list of {"role": "user"/"assistant", "content": "..."}

        Returns:
            Complete prompt string ready for LLM
        """
        sections = []

        # Section 1: System Instructions
        sections.append(self._build_system_instructions(config))

        # Section 2: Context
        sections.append(self._build_context_section(chunks))

        # Section 3: Chat History (optional)
        if history and len(history) > 0:
            sections.append(self._build_history_section(history))

        # Section 4: User Query
        sections.append(self._build_query_section(query))

        # Section 5: Generation Instructions
        sections.append(self._build_generation_instructions(config))

        # Combine all sections
        prompt = "\n\n".join(sections)

        return prompt

    def _build_system_instructions(self, config: AssistantConfig) -> str:
        """Build system instructions from AssistantConfig."""
        # Map sensitivity levels to instructions
        sensitivity_instructions = {
            "LOW": "You can provide informational answers directly.",
            "MEDIUM": "Be careful with advice. For important decisions, suggest consulting relevant experts.",
            "HIGH": "Be conservative. Do NOT make definitive legal, medical, or financial decisions. Always recommend consulting appropriate professionals for critical matters.",
        }

        sensitivity_text = sensitivity_instructions.get(
            config.sensitivity.upper(),
            sensitivity_instructions["MEDIUM"]
        )

        instructions = f"""You are an internal AI assistant for organization {config.org_id}.

Your domain is: {config.domain}
Your main purpose is: {config.use_case}

Answer ALWAYS in {config.primary_language} (Uzbek) unless the user explicitly requests another language.
Tone: {config.tone}
Style: {config.style}

You MUST base your answers ONLY on the provided context snippets from this organization.
If the answer is not clearly supported by the context, say you are not sure and suggest asking a relevant human team or department.

Sensitivity level: {config.sensitivity}
{sensitivity_text}"""

        return instructions

    def _build_context_section(self, chunks: List[Chunk]) -> str:
        """Build context section from retrieved chunks."""
        if not chunks:
            return "[CONTEXT]\nNo relevant context found in the knowledge base."

        context_parts = ["[CONTEXT]"]

        for i, chunk in enumerate(chunks, start=1):
            # Extract metadata
            doc_title = chunk.metadata.get("document_title", "unknown")
            section_title = chunk.metadata.get("section_title", "N/A")
            language = chunk.metadata.get("language", "unknown")

            # Build context snippet
            snippet = f"""[CONTEXT {i}]
Source document: {doc_title}
Section: {section_title}
Language: {language}

{chunk.text}"""

            context_parts.append(snippet)

        return "\n\n".join(context_parts)

    def _build_history_section(self, history: List[Dict[str, str]]) -> str:
        """Build chat history section."""
        history_parts = ["[CHAT HISTORY]"]

        for turn in history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            if role in ["user", "assistant"]:
                history_parts.append(f"{role}: {content}")

        return "\n".join(history_parts)

    def _build_query_section(self, query: str) -> str:
        """Build user query section."""
        return f"""[USER QUESTION]
{query}"""

    def _build_generation_instructions(self, config: AssistantConfig) -> str:
        """Build final generation instructions."""
        instructions = f"""Using ONLY the above CONTEXT and respecting the instructions, answer the user's question in {config.primary_language}.

If the answer cannot be found in the context:
- Clearly state that you don't have enough information
- Suggest which department or team might help (if relevant)
- Ask a clarification question if needed

If the context contains the answer:
- Provide a direct, helpful response
- Cite which context snippet(s) you used
- Use the specified tone ({config.tone}) and style ({config.style})

Remember: Answer in {config.primary_language} (Uzbek) unless explicitly asked otherwise."""

        return instructions
