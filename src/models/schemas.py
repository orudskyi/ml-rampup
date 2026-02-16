from typing import Any

from pydantic import BaseModel, Field


class UpsertRequest(BaseModel):
    """
    Schema for upserting documents into the vector store.
    """

    text: str = Field(..., description="The text content to be stored.")
    metadata: dict[str, Any] | None = Field(
        None, description="Optional metadata associated with the text."
    )


class SearchRequest(BaseModel):
    """
    Schema for searching documents.
    """

    query: str = Field(..., description="The search query text.")
    limit: int = Field(5, gt=0, description="Maximum number of results to return.")


class SearchResponse(BaseModel):
    """
    Schema for search results.
    """

    text: str = Field(..., description="The relevant text retrieved.")
    score: float = Field(..., description="Similarity score of the match.")
