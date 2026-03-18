"""
Guard-Rail Generator for SG-CL
==============================

The Guard-Rail Generator synthesizes constraint-preserving knowledge statements
that are added to training batches when conflicts are detected.

Purpose of Guard-Rails:
1. Preserve existing knowledge during gradient updates
2. Encode exceptions explicitly
3. Prevent over-generalization
4. Act as semantic anchors during training

Guard-rails are NOT prompts — they are training data that shapes gradients.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.conceptnet_client import ConceptNetClient, ConceptNetEdge, create_client
from sid.semantic_inconsistency_detector import (
    SemanticInconsistencyDetector,
    ClaimClassification,
    RelationType,
    create_sid
)


class GuardRailType(Enum):
    """Types of guard-rail statements."""
    HIERARCHICAL = "hierarchical"      # IsA relationships (penguin is a bird)
    CONSTRAINT = "constraint"          # Explicit constraints (penguins cannot fly)
    EXCEPTION = "exception"            # Exception patterns (unlike most birds...)
    PROPERTY = "property"              # Properties to preserve (birds have feathers)
    CONTEXT = "context"                # Contextual facts (penguins live in Antarctica)
    REINFORCEMENT = "reinforcement"    # Reinforce correct knowledge


@dataclass
class GuardRail:
    """Represents a single guard-rail statement."""
    text: str                          # Natural language statement
    rail_type: GuardRailType          # Type of guard-rail
    source_relation: str              # Source relation from knowledge base
    weight: float                     # Importance weight for training
    metadata: Dict = field(default_factory=dict)
    
    def __repr__(self):
        return f"GuardRail({self.rail_type.value}: '{self.text[:50]}...')"


@dataclass
class GuardRailBatch:
    """A batch of guard-rails for a conflicting claim."""
    original_claim: str               # The conflicting claim
    claim_triple: Tuple[str, str, str]  # (subject, relation, object)
    conflict_type: str                # Type of conflict detected
    guard_rails: List[GuardRail]      # Generated guard-rails
    
    def get_training_texts(self) -> List[str]:
        """Get all texts for training (claim + guard-rails)."""
        texts = [self.original_claim]
        texts.extend([gr.text for gr in self.guard_rails])
        return texts
    
    def get_weighted_texts(self) -> List[Tuple[str, float]]:
        """Get texts with their importance weights."""
        weighted = [(self.original_claim, 1.0)]
        weighted.extend([(gr.text, gr.weight) for gr in self.guard_rails])
        return weighted


class GuardRailTemplates:
    """
    Natural language templates for generating guard-rail statements.
    
    Templates are designed to be:
    - Declarative (factual statements)
    - Explicit (clear constraints)
    - Training-friendly (good for gradient updates)
    """
    
    # Hierarchical templates (IsA)
    HIERARCHICAL = [
        "A {subject} is a {parent}.",
        "{subject}s are {parent}s.",
        "The {subject} belongs to the category of {parent}s.",
    ]
    
    # Constraint templates (NotCapableOf, etc.)
    CONSTRAINT = [
        "{subject}s cannot {action}.",
        "A {subject} is not able to {action}.",
        "It is not possible for {subject}s to {action}.",
        "{subject}s lack the ability to {action}.",
    ]
    
    # Exception templates
    EXCEPTION = [
        "Unlike most {parent}s, {subject}s cannot {action}.",
        "While {parent}s typically can {action}, {subject}s are an exception.",
        "{subject}s are {parent}s that cannot {action}.",
        "Most {parent}s can {action}, but {subject}s cannot.",
        "Although {subject}s are {parent}s, they cannot {action}.",
    ]
    
    # Property templates
    PROPERTY = [
        "{subject}s have {property}.",
        "A {subject} has {property}.",
        "{subject}s are characterized by {property}.",
    ]
    
    # Capability templates (CapableOf - for reinforcement)
    CAPABILITY = [
        "{subject}s can {action}.",
        "A {subject} is able to {action}.",
        "{subject}s have the ability to {action}.",
    ]
    
    # Context/Location templates
    CONTEXT = [
        "{subject}s are found in {location}.",
        "{subject}s live in {location}.",
        "The natural habitat of {subject}s is {location}.",
    ]
    
    # Contrastive templates (for emphasizing differences)
    CONTRASTIVE = [
        "While {subject}s cannot {action}, they can {alt_action}.",
        "{subject}s cannot {action}, but they are excellent at {alt_action}.",
        "Instead of {action}, {subject}s {alt_action}.",
    ]


class GuardRailGenerator:
    """
    Main Guard-Rail Generator Class
    
    Generates constraint-preserving statements for training when
    the SID detects a conflict between new data and existing knowledge.
    """
    
    def __init__(
        self, 
        conceptnet_client: Optional[ConceptNetClient] = None,
        max_rails_per_type: int = 2,
        include_contrastive: bool = True
    ):
        """
        Initialize Guard-Rail Generator.
        
        Args:
            conceptnet_client: ConceptNet client for knowledge retrieval
            max_rails_per_type: Maximum guard-rails per type
            include_contrastive: Include contrastive statements
        """
        self.conceptnet = conceptnet_client or create_client()
        self.max_rails_per_type = max_rails_per_type
        self.include_contrastive = include_contrastive
        self.templates = GuardRailTemplates()
    
    def generate(
        self, 
        claim: str,
        claim_triple: Tuple[str, str, str],
        conflict_type: str
    ) -> GuardRailBatch:
        """
        Generate guard-rails for a conflicting claim.
        
        Args:
            claim: Original claim text
            claim_triple: (subject, relation, object) triple
            conflict_type: Type of conflict ("direct", "inherited", "exception")
            
        Returns:
            GuardRailBatch with generated guard-rails
        """
        subject, relation, obj = claim_triple
        guard_rails = []
        
        # 1. Generate hierarchical guard-rails (IsA)
        hierarchical = self._generate_hierarchical(subject)
        guard_rails.extend(hierarchical)
        
        # 2. Generate constraint guard-rails
        constraints = self._generate_constraints(subject, relation, obj)
        guard_rails.extend(constraints)
        
        # 3. Generate exception guard-rails if applicable
        if conflict_type in ["inherited", "exception"]:
            exceptions = self._generate_exceptions(subject, relation, obj)
            guard_rails.extend(exceptions)
        
        # 4. Generate contrastive guard-rails
        if self.include_contrastive:
            contrastive = self._generate_contrastive(subject, relation, obj)
            guard_rails.extend(contrastive)
        
        # 5. Generate property guard-rails
        properties = self._generate_properties(subject)
        guard_rails.extend(properties)
        
        return GuardRailBatch(
            original_claim=claim,
            claim_triple=claim_triple,
            conflict_type=conflict_type,
            guard_rails=guard_rails
        )
    
    def _generate_hierarchical(self, subject: str) -> List[GuardRail]:
        """Generate hierarchical (IsA) guard-rails."""
        rails = []
        parents = self.conceptnet.get_parents(subject)
        
        for i, parent_edge in enumerate(parents[:self.max_rails_per_type]):
            parent = parent_edge.end_label
            
            # Use surface text if available, otherwise template
            if parent_edge.surface_text:
                text = parent_edge.surface_text
            else:
                template = self.templates.HIERARCHICAL[i % len(self.templates.HIERARCHICAL)]
                text = template.format(subject=subject, parent=parent)
            
            rails.append(GuardRail(
                text=text,
                rail_type=GuardRailType.HIERARCHICAL,
                source_relation=f"IsA({subject}, {parent})",
                weight=parent_edge.weight / 4.0,  # Normalize to ~1.0
                metadata={"parent": parent}
            ))
        
        return rails
    
    def _generate_constraints(
        self, 
        subject: str, 
        relation: str, 
        obj: str
    ) -> List[GuardRail]:
        """Generate constraint guard-rails (NotCapableOf, etc.)."""
        rails = []
        
        # Get the opposite relation
        opposite_relations = {
            "CapableOf": "NotCapableOf",
            "NotCapableOf": "CapableOf",
        }
        
        opposite = opposite_relations.get(relation)
        if opposite:
            # Check if constraint exists
            exists, edge = self.conceptnet.check_relation_exists(subject, opposite, obj)
            if exists:
                if edge.surface_text:
                    text = edge.surface_text
                else:
                    if opposite == "NotCapableOf":
                        template = self.templates.CONSTRAINT[0]
                        text = template.format(subject=subject, action=obj)
                    else:
                        template = self.templates.CAPABILITY[0]
                        text = template.format(subject=subject, action=obj)
                
                rails.append(GuardRail(
                    text=text,
                    rail_type=GuardRailType.CONSTRAINT,
                    source_relation=f"{opposite}({subject}, {obj})",
                    weight=edge.weight / 4.0,
                    metadata={"opposite_relation": opposite}
                ))
        
        # Also get other NotCapableOf relations for context
        _, not_capable = self.conceptnet.get_capabilities(subject)
        for edge in not_capable[:self.max_rails_per_type]:
            if edge.end_label.lower() != obj.lower():  # Skip the one we already added
                if edge.surface_text:
                    text = edge.surface_text
                else:
                    text = f"{subject.capitalize()}s cannot {edge.end_label}."
                
                rails.append(GuardRail(
                    text=text,
                    rail_type=GuardRailType.CONSTRAINT,
                    source_relation=f"NotCapableOf({subject}, {edge.end_label})",
                    weight=edge.weight / 4.0,
                    metadata={}
                ))
        
        return rails
    
    def _generate_exceptions(
        self, 
        subject: str, 
        relation: str, 
        obj: str
    ) -> List[GuardRail]:
        """Generate exception pattern guard-rails."""
        rails = []
        
        # Get parents
        parents = self.conceptnet.get_parents(subject)
        
        for parent_edge in parents[:self.max_rails_per_type]:
            parent = parent_edge.end_label
            
            # Check if parent has the capability
            parent_capable, _ = self.conceptnet.get_capabilities(parent)
            for cap_edge in parent_capable:
                if cap_edge.end_label.lower() == obj.lower():
                    # Parent can do it, but subject cannot - exception pattern!
                    template = self.templates.EXCEPTION[
                        len(rails) % len(self.templates.EXCEPTION)
                    ]
                    text = template.format(
                        subject=subject,
                        parent=parent,
                        action=obj
                    )
                    
                    rails.append(GuardRail(
                        text=text,
                        rail_type=GuardRailType.EXCEPTION,
                        source_relation=f"Exception({subject}, {parent}, {obj})",
                        weight=1.2,  # Higher weight for exceptions
                        metadata={"parent": parent, "action": obj}
                    ))
                    break
        
        return rails
    
    def _generate_contrastive(
        self, 
        subject: str, 
        relation: str, 
        obj: str
    ) -> List[GuardRail]:
        """Generate contrastive guard-rails (what it CAN do instead)."""
        rails = []
        
        if relation != "CapableOf":
            return rails
        
        # Get what the subject CAN do
        capable, _ = self.conceptnet.get_capabilities(subject)
        
        for cap_edge in capable[:self.max_rails_per_type]:
            alt_action = cap_edge.end_label
            
            template = self.templates.CONTRASTIVE[
                len(rails) % len(self.templates.CONTRASTIVE)
            ]
            text = template.format(
                subject=subject,
                action=obj,
                alt_action=alt_action
            )
            
            rails.append(GuardRail(
                text=text,
                rail_type=GuardRailType.REINFORCEMENT,
                source_relation=f"Contrastive({subject}, {obj}, {alt_action})",
                weight=0.8,
                metadata={"alt_action": alt_action}
            ))
        
        return rails
    
    def _generate_properties(self, subject: str) -> List[GuardRail]:
        """Generate property guard-rails."""
        rails = []
        
        # Query for HasProperty relations
        properties = self.conceptnet.query_relation(subject, "HasProperty")
        
        for prop_edge in properties[:self.max_rails_per_type]:
            prop = prop_edge.end_label
            
            if prop_edge.surface_text:
                text = prop_edge.surface_text
            else:
                template = self.templates.PROPERTY[
                    len(rails) % len(self.templates.PROPERTY)
                ]
                text = template.format(subject=subject, property=prop)
            
            rails.append(GuardRail(
                text=text,
                rail_type=GuardRailType.PROPERTY,
                source_relation=f"HasProperty({subject}, {prop})",
                weight=prop_edge.weight / 4.0,
                metadata={"property": prop}
            ))
        
        return rails
    
    def generate_from_classification(
        self, 
        classification: ClaimClassification
    ) -> GuardRailBatch:
        """
        Generate guard-rails from a SID classification result.
        
        This is the main integration point with SID.
        """
        return self.generate(
            claim=classification.claim.source_text,
            claim_triple=classification.claim.to_triple(),
            conflict_type=classification.conflict_result.conflict_type
        )


class GatedBatchConstructor:
    """
    Constructs training batches with gating applied.
    
    This is the final stage before training, combining:
    - SID analysis
    - Guard-rail generation
    - Batch construction logic
    """
    
    def __init__(
        self,
        sid: Optional[SemanticInconsistencyDetector] = None,
        generator: Optional[GuardRailGenerator] = None
    ):
        """
        Initialize batch constructor.
        
        Args:
            sid: Semantic Inconsistency Detector
            generator: Guard-Rail Generator
        """
        self.sid = sid or create_sid()
        self.generator = generator or GuardRailGenerator(self.sid.conceptnet)
    
    def construct_batch(
        self, 
        claims: List[str],
        include_weights: bool = False
    ) -> Dict:
        """
        Construct a gated training batch from claims.
        
        Args:
            claims: List of input claims
            include_weights: Include importance weights
            
        Returns:
            Dictionary with batch data and statistics
        """
        safe_texts = []
        gated_texts = []
        gated_weights = []
        
        stats = {
            "total_claims": len(claims),
            "safe_claims": 0,
            "gated_claims": 0,
            "failed_extraction": 0,
            "total_guard_rails": 0,
            "guard_rail_types": {}
        }
        
        for claim in claims:
            # Analyze with SID
            result = self.sid.analyze(claim)
            
            if result is None:
                stats["failed_extraction"] += 1
                continue
            
            if result.gating_decision == "normal_training":
                # Safe claim - add directly
                safe_texts.append(claim)
                stats["safe_claims"] += 1
            else:
                # Conflicting claim - generate guard-rails
                batch = self.generator.generate_from_classification(result)
                
                if include_weights:
                    weighted = batch.get_weighted_texts()
                    for text, weight in weighted:
                        gated_texts.append(text)
                        gated_weights.append(weight)
                else:
                    gated_texts.extend(batch.get_training_texts())
                
                stats["gated_claims"] += 1
                stats["total_guard_rails"] += len(batch.guard_rails)
                
                # Track guard-rail types
                for gr in batch.guard_rails:
                    type_name = gr.rail_type.value
                    stats["guard_rail_types"][type_name] = \
                        stats["guard_rail_types"].get(type_name, 0) + 1
        
        result = {
            "safe_texts": safe_texts,
            "gated_texts": gated_texts,
            "all_texts": safe_texts + gated_texts,
            "stats": stats
        }
        
        if include_weights:
            # Safe claims get weight 1.0
            safe_weights = [1.0] * len(safe_texts)
            result["weights"] = safe_weights + gated_weights
        
        return result


# =============================================================================
# Factory Functions
# =============================================================================

def create_generator(
    conceptnet_client: Optional[ConceptNetClient] = None
) -> GuardRailGenerator:
    """Create a Guard-Rail Generator with default settings."""
    return GuardRailGenerator(conceptnet_client)


def create_batch_constructor(
    sid: Optional[SemanticInconsistencyDetector] = None,
    generator: Optional[GuardRailGenerator] = None
) -> GatedBatchConstructor:
    """Create a Gated Batch Constructor with default settings."""
    return GatedBatchConstructor(sid, generator)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Guard-Rail Generator - Test Suite")
    print("=" * 70)
    
    # Create components
    generator = create_generator()
    batch_constructor = create_batch_constructor()
    
    # Test 1: Generate guard-rails for a single claim
    print("\n1. Testing guard-rail generation for 'Penguins can fly':\n")
    
    batch = generator.generate(
        claim="Penguins can fly.",
        claim_triple=("penguin", "CapableOf", "fly"),
        conflict_type="direct"
    )
    
    print(f"Original claim: {batch.original_claim}")
    print(f"Conflict type: {batch.conflict_type}")
    print(f"\nGenerated guard-rails ({len(batch.guard_rails)}):")
    for gr in batch.guard_rails:
        print(f"  [{gr.rail_type.value}] {gr.text}")
        print(f"    Source: {gr.source_relation}, Weight: {gr.weight:.2f}")
    
    print("\nTraining texts:")
    for text in batch.get_training_texts():
        print(f"  - {text}")
    
    # Test 2: Full batch construction
    print("\n" + "=" * 70)
    print("\n2. Testing full batch construction:\n")
    
    test_claims = [
        "Penguins can fly.",
        "Penguins can swim.",
        "Dogs can fly.",
        "Cats can climb.",
        "Fish can walk.",
        "Birds can fly.",
    ]
    
    result = batch_constructor.construct_batch(test_claims, include_weights=True)
    
    print(f"Statistics:")
    for key, value in result["stats"].items():
        print(f"  {key}: {value}")
    
    print(f"\nSafe texts ({len(result['safe_texts'])}):")
    for text in result["safe_texts"]:
        print(f"  ✓ {text}")
    
    print(f"\nGated texts ({len(result['gated_texts'])}):")
    for i, text in enumerate(result["gated_texts"]):
        weight = result["weights"][len(result["safe_texts"]) + i]
        print(f"  ⚠ [{weight:.2f}] {text}")
    
    # Test 3: From SID classification
    print("\n" + "=" * 70)
    print("\n3. Testing integration with SID:\n")
    
    sid = create_sid()
    classification = sid.analyze("Dogs can fly.")
    
    if classification and classification.conflict_result.has_conflict:
        batch = generator.generate_from_classification(classification)
        print(f"Claim: {batch.original_claim}")
        print(f"Guard-rails generated: {len(batch.guard_rails)}")
        for gr in batch.guard_rails[:5]:
            print(f"  - {gr.text}")
    
    print("\n" + "=" * 70)
    print("Guard-Rail Generator tests completed!")
    print("=" * 70)
