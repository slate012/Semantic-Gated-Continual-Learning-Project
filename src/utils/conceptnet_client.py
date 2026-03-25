"""
ConceptNet Client for SG-CL
===========================
This module provides an interface to query ConceptNet for symbolic knowledge.
It supports querying relations, detecting conflicts, and retrieving guard-rail knowledge.

Key Relations for SG-CL:
- IsA: Taxonomic hierarchy (penguin IsA bird)
- CapableOf: Abilities (bird CapableOf fly)
- NotCapableOf: Inability constraints (penguin NotCapableOf fly)
- HasProperty: Properties (ice HasProperty cold)
- AtLocation: Spatial relations
- UsedFor: Functional relations
"""

import requests
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from functools import lru_cache
import time
import json
import os

# Import local fallback knowledge
try:
    from . import local_knowledge
except ImportError:
    import local_knowledge


@dataclass
class ConceptNetEdge:
    """Represents a single edge (relation) from ConceptNet."""
    start: str          # Subject concept (e.g., "/c/en/penguin")
    end: str            # Object concept (e.g., "/c/en/bird")
    relation: str       # Relation type (e.g., "/r/IsA")
    weight: float       # Confidence weight
    surface_text: str   # Natural language form (e.g., "A penguin is a bird")
    
    @property
    def start_label(self) -> str:
        """Extract human-readable label from start concept."""
        return self.start.split('/')[-1].replace('_', ' ')
    
    @property
    def end_label(self) -> str:
        """Extract human-readable label from end concept."""
        return self.end.split('/')[-1].replace('_', ' ')
    
    @property
    def relation_type(self) -> str:
        """Extract relation name without prefix."""
        return self.relation.split('/')[-1]
    
    def __repr__(self):
        return f"{self.start_label} --[{self.relation_type}]--> {self.end_label} (w={self.weight:.2f})"


@dataclass
class ConflictResult:
    """Result of a conflict check between a claim and existing knowledge."""
    has_conflict: bool
    conflict_type: str  # "none", "direct", "inherited", "exception"
    conflicting_edges: List[ConceptNetEdge]
    explanation: str
    
    
class ConceptNetClient:
    """
    Client for querying ConceptNet knowledge graph.
    
    Provides:
    - Relation querying (IsA, CapableOf, NotCapableOf, etc.)
    - Conflict detection for SG-CL gating
    - Guard-rail knowledge retrieval
    """
    
    BASE_URL = "http://api.conceptnet.io"
    
    # Relation pairs that are logical opposites
    OPPOSITE_RELATIONS = {
        "CapableOf": "NotCapableOf",
        "NotCapableOf": "CapableOf",
        "HasProperty": "NotHasProperty",
        "Desires": "NotDesires",
    }
    
    # Relations that support inheritance
    INHERITABLE_RELATIONS = {"CapableOf", "HasProperty", "AtLocation", "UsedFor"}
    
    def __init__(self, cache_dir: Optional[str] = None, rate_limit_delay: float = 0.1, local_only: bool = False):
        """
        Initialize ConceptNet client.
        
        Args:
            cache_dir: Directory for caching API responses
            rate_limit_delay: Delay between API calls to avoid rate limiting
            local_only: If True, skip remote API calls and use local knowledge only
        """
        self.cache_dir = cache_dir
        self.rate_limit_delay = rate_limit_delay
        self.local_only = local_only
        self._last_request_time = 0
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _get_cache_path(self, key: str) -> str:
        """Generate cache file path for a query."""
        if not self.cache_dir:
            return None
        # Sanitize key for filename
        safe_key = key.replace('/', '_').replace('?', '_').replace('&', '_')
        return os.path.join(self.cache_dir, f"{safe_key}.json")
    
    def _load_from_cache(self, key: str) -> Optional[dict]:
        """Load cached response if available."""
        cache_path = self._get_cache_path(key)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, key: str, data: dict):
        """Save response to cache."""
        cache_path = self._get_cache_path(key)
        if cache_path:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make a request to ConceptNet API with caching and rate limiting."""
        # Skip HTTP when in local-only mode
        if self.local_only:
            return self._fallback_to_local(params)
        
        cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
        
        # Try cache first
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached
        
        # Make API request
        self._rate_limit()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache successful responses
            self._save_to_cache(cache_key, data)
            return data
            
        except requests.exceptions.RequestException as e:
            # Fallback to local knowledge
            return self._fallback_to_local(params)
    
    def _fallback_to_local(self, params: dict = None) -> dict:
        """Fallback to local knowledge base when API is unavailable."""
        if not params:
            return {"edges": []}
        
        # Extract query parameters
        start = params.get("start", "")
        rel = params.get("rel", "")
        end = params.get("end", "")
        
        # Parse concept from URI (e.g., "/c/en/penguin" -> "penguin")
        subject = start.split("/")[-1] if start else None
        relation = rel.split("/")[-1] if rel else None
        obj = end.split("/")[-1] if end else None
        
        # Query local knowledge
        local_edges = local_knowledge.get_local_edges(
            subject=subject,
            relation=relation,
            obj=obj
        )
        
        # Convert to API-like format
        edges = []
        for s, r, o, w, text in local_edges:
            edges.append({
                "start": {"@id": f"/c/en/{s}"},
                "end": {"@id": f"/c/en/{o}"},
                "rel": {"@id": f"/r/{r}"},
                "weight": w,
                "surfaceText": text
            })
        
        if edges:
            print(f"  [Using local fallback: {len(edges)} edges found]")
        
        return {"edges": edges}
    
    def _parse_edges(self, data: dict) -> List[ConceptNetEdge]:
        """Parse API response into ConceptNetEdge objects."""
        edges = []
        for edge in data.get("edges", []):
            try:
                edges.append(ConceptNetEdge(
                    start=edge.get("start", {}).get("@id", ""),
                    end=edge.get("end", {}).get("@id", ""),
                    relation=edge.get("rel", {}).get("@id", ""),
                    weight=edge.get("weight", 0.0),
                    surface_text=edge.get("surfaceText", "")
                ))
            except (KeyError, TypeError):
                continue
        return edges
    
    def normalize_concept(self, concept: str, lang: str = "en") -> str:
        """
        Normalize a concept to ConceptNet format.
        
        Args:
            concept: Raw concept string (e.g., "penguin", "can fly")
            lang: Language code
            
        Returns:
            ConceptNet URI (e.g., "/c/en/penguin")
        """
        # Clean and normalize
        concept = concept.lower().strip()
        concept = concept.replace(' ', '_')
        return f"/c/{lang}/{concept}"
    
    def query_concept(self, concept: str, limit: int = 100) -> List[ConceptNetEdge]:
        """
        Get all edges related to a concept.
        
        Args:
            concept: Concept to query (e.g., "penguin")
            limit: Maximum number of edges to return
            
        Returns:
            List of ConceptNetEdge objects
        """
        uri = self.normalize_concept(concept)
        data = self._make_request(f"/c/en/{concept}", {"limit": limit})
        return self._parse_edges(data)
    
    def query_relation(
        self, 
        subject: str, 
        relation: str, 
        obj: Optional[str] = None,
        limit: int = 50
    ) -> List[ConceptNetEdge]:
        """
        Query for specific relations.
        
        Args:
            subject: Subject concept
            relation: Relation type (e.g., "IsA", "CapableOf")
            obj: Optional object concept to filter by
            limit: Maximum results
            
        Returns:
            List of matching edges
        """
        params = {
            "start": self.normalize_concept(subject),
            "rel": f"/r/{relation}",
            "limit": limit
        }
        if obj:
            params["end"] = self.normalize_concept(obj)
            
        data = self._make_request("/query", params)
        return self._parse_edges(data)
    
    def get_parents(self, concept: str) -> List[ConceptNetEdge]:
        """
        Get parent concepts via IsA relation.
        
        Args:
            concept: Concept to find parents for
            
        Returns:
            List of IsA edges (concept IsA parent)
        """
        return self.query_relation(concept, "IsA")
    
    def get_capabilities(self, concept: str) -> Tuple[List[ConceptNetEdge], List[ConceptNetEdge]]:
        """
        Get capabilities and incapabilities for a concept.
        
        Returns:
            Tuple of (CapableOf edges, NotCapableOf edges)
        """
        capable = self.query_relation(concept, "CapableOf")
        not_capable = self.query_relation(concept, "NotCapableOf")
        return capable, not_capable
    
    def check_relation_exists(
        self, 
        subject: str, 
        relation: str, 
        obj: str,
        min_weight: float = 0.5
    ) -> Tuple[bool, Optional[ConceptNetEdge]]:
        """
        Check if a specific relation exists in ConceptNet.
        
        Args:
            subject: Subject concept
            relation: Relation type
            obj: Object concept
            min_weight: Minimum weight threshold
            
        Returns:
            (exists, matching_edge)
        """
        edges = self.query_relation(subject, relation, obj)
        for edge in edges:
            if edge.weight >= min_weight:
                return True, edge
        return False, None
    
    def get_inherited_relations(
        self, 
        concept: str, 
        relation: str,
        max_depth: int = 3
    ) -> List[Tuple[ConceptNetEdge, str]]:
        """
        Get relations that might be inherited from parent concepts.
        
        Args:
            concept: Starting concept
            relation: Relation to look for
            max_depth: Maximum inheritance depth
            
        Returns:
            List of (edge, inherited_from) tuples
        """
        inherited = []
        visited = {concept}
        queue = [(concept, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            
            # Get parents
            parents = self.get_parents(current)
            for parent_edge in parents:
                parent = parent_edge.end_label
                if parent in visited:
                    continue
                visited.add(parent)
                
                # Check if parent has the relation
                relations = self.query_relation(parent, relation)
                for rel in relations:
                    inherited.append((rel, parent))
                
                queue.append((parent, depth + 1))
        
        return inherited
    
    # =========================================================================
    # SG-CL Specific Methods
    # =========================================================================
    
    def detect_conflict(
        self, 
        subject: str, 
        relation: str, 
        obj: str
    ) -> ConflictResult:
        """
        Detect if a claim conflicts with existing ConceptNet knowledge.
        
        This is the core conflict detection method for SG-CL.
        
        Args:
            subject: Subject of the claim (e.g., "penguin")
            relation: Relation type (e.g., "CapableOf")
            obj: Object of the claim (e.g., "fly")
            
        Returns:
            ConflictResult with conflict details
        """
        conflicting_edges = []
        
        # 1. Check for direct contradiction
        opposite_relation = self.OPPOSITE_RELATIONS.get(relation)
        if opposite_relation:
            exists, edge = self.check_relation_exists(subject, opposite_relation, obj)
            if exists:
                return ConflictResult(
                    has_conflict=True,
                    conflict_type="direct",
                    conflicting_edges=[edge],
                    explanation=f"Direct contradiction: {subject} has {opposite_relation}({obj}) in ConceptNet"
                )
        
        # 2. Check existing same relation with different polarity
        # e.g., claim "penguin CapableOf fly" but "penguin NotCapableOf fly" exists
        existing, edge = self.check_relation_exists(subject, relation, obj)
        if existing:
            # The relation already exists - no conflict, reinforcement
            return ConflictResult(
                has_conflict=False,
                conflict_type="none",
                conflicting_edges=[],
                explanation=f"Relation already exists in ConceptNet with weight {edge.weight:.2f}"
            )
        
        # 3. Check for inherited conflicts
        if relation in self.INHERITABLE_RELATIONS:
            inherited = self.get_inherited_relations(subject, relation)
            opposite_inherited = []
            if opposite_relation:
                opposite_inherited = self.get_inherited_relations(subject, opposite_relation)
            
            # Check if opposite relation is inherited
            for edge, parent in opposite_inherited:
                if edge.end_label.lower() == obj.lower():
                    conflicting_edges.append(edge)
            
            if conflicting_edges:
                return ConflictResult(
                    has_conflict=True,
                    conflict_type="inherited",
                    conflicting_edges=conflicting_edges,
                    explanation=f"Inherited conflict from parent concepts"
                )
            
            # Check for exception pattern (parent has capability, child doesn't)
            for edge, parent in inherited:
                if edge.end_label.lower() == obj.lower():
                    # Parent can do it, but we're claiming child can too
                    # Check if child is explicitly marked as exception
                    _, not_capable = self.get_capabilities(subject)
                    for nc in not_capable:
                        if nc.end_label.lower() == obj.lower():
                            return ConflictResult(
                                has_conflict=True,
                                conflict_type="exception",
                                conflicting_edges=[nc, edge],
                                explanation=f"{subject} is an exception to {parent}'s ability to {obj}"
                            )
        
        # No conflict found
        return ConflictResult(
            has_conflict=False,
            conflict_type="none",
            conflicting_edges=[],
            explanation="No conflict detected with ConceptNet knowledge"
        )
    
    def get_guardrail_knowledge(
        self, 
        subject: str, 
        relation: str, 
        obj: str
    ) -> List[str]:
        """
        Retrieve guard-rail knowledge statements for a conflicting claim.
        
        These statements can be used to augment training data to preserve
        existing knowledge while allowing new learning.
        
        Args:
            subject: Subject of the claim
            relation: Relation type
            obj: Object of the claim
            
        Returns:
            List of natural language guard-rail statements
        """
        guardrails = []
        
        # 1. Get parent concepts
        parents = self.get_parents(subject)
        for parent in parents[:3]:  # Limit to top 3 parents
            if parent.surface_text:
                guardrails.append(parent.surface_text)
            else:
                guardrails.append(f"A {subject} is a {parent.end_label}.")
        
        # 2. Get existing capabilities/properties
        capable, not_capable = self.get_capabilities(subject)
        
        for edge in not_capable[:2]:
            if edge.surface_text:
                guardrails.append(edge.surface_text)
            else:
                guardrails.append(f"A {subject} cannot {edge.end_label}.")
        
        # 3. Get parent capabilities if this is an exception case
        for parent in parents[:2]:
            parent_capable, _ = self.get_capabilities(parent.end_label)
            for edge in parent_capable:
                if edge.end_label.lower() == obj.lower():
                    guardrails.append(f"Most {parent.end_label}s can {obj}, but {subject}s are an exception.")
        
        # 4. Get related constraints
        opposite = self.OPPOSITE_RELATIONS.get(relation)
        if opposite:
            exists, edge = self.check_relation_exists(subject, opposite, obj)
            if exists and edge.surface_text:
                guardrails.append(edge.surface_text)
        
        return guardrails
    
    def analyze_claim(self, claim_triple: Tuple[str, str, str]) -> Dict:
        """
        Full analysis of a claim for SG-CL processing.
        
        Args:
            claim_triple: (subject, relation, object)
            
        Returns:
            Dictionary with conflict analysis and guard-rails
        """
        subject, relation, obj = claim_triple
        
        # Detect conflicts
        conflict_result = self.detect_conflict(subject, relation, obj)
        
        # Get guard-rails if conflict exists
        guardrails = []
        if conflict_result.has_conflict:
            guardrails = self.get_guardrail_knowledge(subject, relation, obj)
        
        return {
            "claim": {
                "subject": subject,
                "relation": relation,
                "object": obj
            },
            "conflict": {
                "has_conflict": conflict_result.has_conflict,
                "type": conflict_result.conflict_type,
                "explanation": conflict_result.explanation,
                "conflicting_edges": [str(e) for e in conflict_result.conflicting_edges]
            },
            "guardrails": guardrails,
            "gating_decision": "gated_training" if conflict_result.has_conflict else "normal_training"
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_client(cache_dir: str = None, local_only: bool = False) -> ConceptNetClient:
    """Create a ConceptNet client with default settings."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "conceptnet_cache")
    return ConceptNetClient(cache_dir=cache_dir, local_only=local_only)


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    # Create client
    client = create_client()
    
    print("=" * 60)
    print("ConceptNet Client for SG-CL - Test Suite")
    print("=" * 60)
    
    # Test 1: Query concept
    print("\n1. Querying 'penguin'...")
    edges = client.query_concept("penguin")
    print(f"   Found {len(edges)} edges")
    for edge in edges[:5]:
        print(f"   - {edge}")
    
    # Test 2: Get parents
    print("\n2. Getting parents of 'penguin'...")
    parents = client.get_parents("penguin")
    for parent in parents[:5]:
        print(f"   - {parent}")
    
    # Test 3: Get capabilities
    print("\n3. Getting capabilities of 'penguin'...")
    capable, not_capable = client.get_capabilities("penguin")
    print(f"   CapableOf: {len(capable)} edges")
    for edge in capable[:3]:
        print(f"   - {edge}")
    print(f"   NotCapableOf: {len(not_capable)} edges")
    for edge in not_capable[:3]:
        print(f"   - {edge}")
    
    # Test 4: Conflict detection (the core SG-CL use case)
    print("\n4. Testing conflict detection...")
    
    # Case A: Conflicting claim
    print("\n   Case A: 'Penguins can fly' (should conflict)")
    result = client.detect_conflict("penguin", "CapableOf", "fly")
    print(f"   Conflict: {result.has_conflict}")
    print(f"   Type: {result.conflict_type}")
    print(f"   Explanation: {result.explanation}")
    
    # Case B: Non-conflicting claim
    print("\n   Case B: 'Penguins can swim' (should not conflict)")
    result = client.detect_conflict("penguin", "CapableOf", "swim")
    print(f"   Conflict: {result.has_conflict}")
    print(f"   Explanation: {result.explanation}")
    
    # Test 5: Full claim analysis
    print("\n5. Full claim analysis for 'Penguins can fly'...")
    analysis = client.analyze_claim(("penguin", "CapableOf", "fly"))
    print(f"   Gating Decision: {analysis['gating_decision']}")
    print(f"   Guard-rails:")
    for gr in analysis['guardrails']:
        print(f"   - {gr}")
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)
