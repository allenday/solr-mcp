"""Vector search results handling."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VectorSearchResult(BaseModel):
    """Individual vector search result."""

    docid: str = Field(description="Internal Solr document ID (_docid_)")
    score: float = Field(description="Search score")
    distance: Optional[float] = Field(None, description="Vector distance if available")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def __getitem__(self, key):
        """Make result subscriptable."""
        if key == "docid":
            return self.docid
        elif key == "score":
            return self.score
        elif key == "distance":
            return self.distance
        elif key == "metadata":
            return self.metadata
        raise KeyError(f"Invalid key: {key}")


class VectorSearchResults(BaseModel):
    """Container for vector search results."""

    results: List[VectorSearchResult] = Field(
        default_factory=list, description="List of search results"
    )
    total_found: int = Field(0, description="Total number of results found")
    top_k: int = Field(..., description="Number of results requested")
    query_time_ms: Optional[int] = Field(
        None, description="Query execution time in milliseconds"
    )

    @property
    def docs(self) -> List[VectorSearchResult]:
        """Get list of search results."""
        return self.results

    @classmethod
    def from_solr_response(
        cls, response: Dict[str, Any], top_k: int = 10
    ) -> "VectorSearchResults":
        """Create VectorSearchResults from Solr response.

        Args:
            response: Raw Solr response dictionary
            top_k: Number of results requested

        Returns:
            VectorSearchResults instance
        """
        # Extract response header
        header = response.get("responseHeader", {})
        query_time = header.get("QTime")

        # Extract main response section
        resp = response.get("response", {})
        docs = resp.get("docs", [])

        # Create results list
        results = []
        for doc in docs:
            # Handle both string and numeric _docid_
            docid = doc.get("_docid_")
            if docid is None:
                # Try alternate field names
                docid = doc.get("[docid]") or doc.get("docid") or "0"
            docid = str(docid)  # Ensure string type

            result = VectorSearchResult(
                docid=docid,
                score=doc.get("score", 0.0),
                distance=doc.get("_vector_distance_"),
                metadata={
                    k: v
                    for k, v in doc.items()
                    if k
                    not in ["_docid_", "[docid]", "docid", "score", "_vector_distance_"]
                },
            )
            results.append(result)

        # Create VectorSearchResults
        return cls(
            results=results,
            total_found=resp.get("numFound", 0),
            top_k=top_k,
            query_time_ms=query_time,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format.

        Returns:
            Dictionary representation of results
        """
        return {
            "results": [result.model_dump() for result in self.results],
            "metadata": {
                "total_found": self.total_found,
                "top_k": self.top_k,
                "query_time_ms": self.query_time_ms,
            },
        }

    def get_doc_ids(self) -> List[str]:
        """Get list of document IDs from results.

        Returns:
            List of document IDs
        """
        return [result.docid for result in self.results]

    def get_scores(self) -> List[float]:
        """Get list of scores from results.

        Returns:
            List of scores
        """
        return [result.score for result in self.results]

    def get_distances(self) -> List[Optional[float]]:
        """Get list of vector distances from results.

        Returns:
            List of distances (None if not available)
        """
        return [result.distance for result in self.results]
