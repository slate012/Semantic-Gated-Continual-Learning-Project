"""
Semantic Inconsistency Detector (SID) for SG-CL
================================================

SID is a pre-training semantic analysis module that:
1. Normalizes incoming text (linguistic normalization)
2. Extracts entities (subject/object)
3. Extracts semantic relations
4. Maps to canonical symbolic form (ConceptNet-compatible)
5. Detects conflicts with existing knowledge

This implementation uses pattern-based extraction for robustness.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.conceptnet_client import ConceptNetClient, ConflictResult, create_client


class RelationType(Enum):
    """Supported semantic relation types."""
    CAPABLE_OF = "CapableOf"
    NOT_CAPABLE_OF = "NotCapableOf"
    IS_A = "IsA"
    HAS_PROPERTY = "HasProperty"
    AT_LOCATION = "AtLocation"
    USED_FOR = "UsedFor"
    CAUSES = "Causes"
    DESIRES = "Desires"
    UNKNOWN = "Unknown"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from text."""
    text: str                    # Original text span
    normalized: str              # Normalized form
    entity_type: str            # Type (e.g., "SUBJECT", "OBJECT")
    start_idx: int              # Start position in text
    end_idx: int                # End position in text
    
    def __repr__(self):
        return f"Entity({self.normalized}, type={self.entity_type})"


@dataclass  
class ExtractedRelation:
    """Represents an extracted semantic relation."""
    subject: ExtractedEntity
    relation_type: RelationType
    object: ExtractedEntity
    confidence: float           # Extraction confidence [0, 1]
    source_text: str           # Original sentence
    
    def to_triple(self) -> Tuple[str, str, str]:
        """Convert to (subject, relation, object) triple."""
        return (
            self.subject.normalized,
            self.relation_type.value,
            self.object.normalized
        )
    
    def __repr__(self):
        return f"({self.subject.normalized}, {self.relation_type.value}, {self.object.normalized})"


@dataclass
class ClaimClassification:
    """Classification of a claim for SG-CL gating."""
    claim: ExtractedRelation
    conflict_result: ConflictResult
    classification: str         # "safe", "hard_conflict", "exception_conflict", "conditional"
    guard_rails: List[str]
    gating_decision: str        # "normal_training", "gated_training", "reject"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "claim": str(self.claim),
            "triple": self.claim.to_triple(),
            "has_conflict": self.conflict_result.has_conflict,
            "conflict_type": self.conflict_result.conflict_type,
            "classification": self.classification,
            "guard_rails": self.guard_rails,
            "gating_decision": self.gating_decision
        }


class TextNormalizer:
    """
    Stage 1: Linguistic Normalization
    
    Normalizes incoming text for consistent processing:
    - Resolve tense
    - Remove stylistic variation
    - Reduce to propositional content
    """
    
    # Common contractions
    CONTRACTIONS = {
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "couldn't": "could not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "they're": "they are",
        "we're": "we are",
        "you're": "you are",
        "i'm": "i am",
        "he's": "he is",
        "she's": "she is",
    }
    
    # Articles and determiners to remove for normalization
    REMOVABLE_WORDS = {"a", "an", "the", "some", "any", "this", "that", "these", "those"}
    
    def normalize(self, text: str) -> str:
        """
        Normalize text for semantic analysis.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower().strip()
        
        # Expand contractions
        for contraction, expansion in self.CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove trailing punctuation for analysis
        text = text.rstrip('.')
        
        return text
    
    def normalize_entity(self, entity: str) -> str:
        """
        Normalize an entity name to canonical form.
        
        Args:
            entity: Raw entity string
            
        Returns:
            Normalized entity (lowercase, singular, underscores)
        """
        entity = entity.lower().strip()
        
        # Remove articles
        words = entity.split()
        words = [w for w in words if w not in self.REMOVABLE_WORDS]
        
        # Simple plural to singular conversion
        words = [self._singularize(w) for w in words]
        
        # Join with underscores for ConceptNet format
        return "_".join(words)
    
    def _singularize(self, word: str) -> str:
        """
        Simple rule-based singularization.
        
        Handles common English plural patterns.
        """
        # Irregular plurals
        irregulars = {
            "penguins": "penguin",
            "children": "child",
            "people": "person",
            "men": "men",
            "women": "woman",
            "mice": "mouse",
            "geese": "goose",
            "teeth": "tooth",
            "feet": "foot",
            "fish": "fish",
            "sheep": "sheep",
            "deer": "deer",
            "species": "species",
            "aircraft": "aircraft",
        }
        
        if word in irregulars:
            return irregulars[word]
        
        # Common patterns (order matters)
        if word.endswith("ies") and len(word) > 3:
            return word[:-3] + "y"  # flies -> fly
        if word.endswith("ves"):
            return word[:-3] + "f"  # wolves -> wolf
        if word.endswith("xes") or word.endswith("ches") or word.endswith("shes") or word.endswith("sses"):
            return word[:-2]  # boxes -> box, watches -> watch
        if word.endswith("oes") and len(word) > 3:
            return word[:-2]  # potatoes -> potato
        if word.endswith("s") and len(word) > 2 and not word.endswith("ss"):
            return word[:-1]  # dogs -> dog, cats -> cat
        
        return word


class EntityExtractor:
    """
    Stage 2: Entity Extraction
    
    Extracts subject and object entities from text using
    pattern-based rules and heuristics.
    """
    
    def __init__(self, normalizer: TextNormalizer):
        self.normalizer = normalizer
    
    def extract_entities(self, text: str) -> Tuple[Optional[ExtractedEntity], Optional[ExtractedEntity]]:
        """
        Extract subject and object entities from a sentence.
        
        Args:
            text: Normalized text
            
        Returns:
            (subject_entity, object_entity) or (None, None) if extraction fails
        """
        normalized = self.normalizer.normalize(text)
        
        # Try various patterns
        for pattern_func in [
            self._extract_can_pattern,
            self._extract_cannot_pattern,
            self._extract_is_a_pattern,
            self._extract_is_adj_pattern,
            self._extract_have_pattern,
            self._extract_are_pattern,
        ]:
            result = pattern_func(normalized)
            if result:
                return result
        
        return None, None
    
    def _create_entity(self, text: str, entity_type: str, start: int = 0, end: int = -1) -> ExtractedEntity:
        """Create an ExtractedEntity from text."""
        normalized = self.normalizer.normalize_entity(text)
        return ExtractedEntity(
            text=text,
            normalized=normalized,
            entity_type=entity_type,
            start_idx=start,
            end_idx=end if end > 0 else len(text)
        )
    
    def _extract_can_pattern(self, text: str) -> Optional[Tuple[ExtractedEntity, ExtractedEntity]]:
        """Extract from 'X can Y' pattern."""
        # Match: "penguins can swim", "a dog can run"
        match = re.match(r'^(.+?)\s+can\s+(.+)$', text)
        if match:
            subject = self._create_entity(match.group(1), "SUBJECT")
            obj = self._create_entity(match.group(2), "OBJECT")
            return subject, obj
        return None
    
    def _extract_cannot_pattern(self, text: str) -> Optional[Tuple[ExtractedEntity, ExtractedEntity]]:
        """Extract from 'X cannot Y' pattern."""
        match = re.match(r'^(.+?)\s+cannot\s+(.+)$', text)
        if match:
            subject = self._create_entity(match.group(1), "SUBJECT")
            obj = self._create_entity(match.group(2), "OBJECT")
            return subject, obj
        return None
    
    def _extract_is_a_pattern(self, text: str) -> Optional[Tuple[ExtractedEntity, ExtractedEntity]]:
        """Extract from 'X is a Y' pattern."""
        match = re.match(r'^(.+?)\s+is\s+(?:a|an)\s+(.+)$', text)
        if match:
            subject = self._create_entity(match.group(1), "SUBJECT")
            obj = self._create_entity(match.group(2), "OBJECT")
            return subject, obj
        return None
    
    def _extract_is_adj_pattern(self, text: str) -> Optional[Tuple[ExtractedEntity, ExtractedEntity]]:
        """Extract from 'X is ADJ' pattern (for HasProperty)."""
        match = re.match(r'^(.+?)\s+is\s+(\w+)$', text)
        if match:
            subject = self._create_entity(match.group(1), "SUBJECT")
            obj = self._create_entity(match.group(2), "PROPERTY")
            return subject, obj
        return None
    
    def _extract_have_pattern(self, text: str) -> Optional[Tuple[ExtractedEntity, ExtractedEntity]]:
        """Extract from 'X has/have Y' pattern."""
        match = re.match(r'^(.+?)\s+(?:has|have)\s+(.+)$', text)
        if match:
            subject = self._create_entity(match.group(1), "SUBJECT")
            obj = self._create_entity(match.group(2), "OBJECT")
            return subject, obj
        return None
    
    def _extract_are_pattern(self, text: str) -> Optional[Tuple[ExtractedEntity, ExtractedEntity]]:
        """Extract from 'X are Y' pattern."""
        match = re.match(r'^(.+?)\s+are\s+(.+)$', text)
        if match:
            subject = self._create_entity(match.group(1), "SUBJECT")
            obj = self._create_entity(match.group(2), "OBJECT")
            return subject, obj
        return None


class RelationExtractor:
    """
    Stage 3: Relation Extraction
    
    Identifies the semantic relation type between entities.
    """
    
    # Patterns for relation detection
    RELATION_PATTERNS = {
        RelationType.CAPABLE_OF: [
            r'\bcan\b(?!\s*not)',
            r'\bis able to\b',
            r'\bcapable of\b',
            r'\bhas the ability to\b',
        ],
        RelationType.NOT_CAPABLE_OF: [
            r'\bcannot\b',
            r'\bcan not\b',
            r'\bcannot\b',
            r'\bis not able to\b',
            r'\bis unable to\b',
            r'\bincapable of\b',
        ],
        RelationType.IS_A: [
            r'\bis a\b',
            r'\bis an\b',
            r'\bare\b.*\b(?:type|kind|form)\b',
        ],
        RelationType.HAS_PROPERTY: [
            r'\bis\s+\w+$',  # "X is ADJ" at end
            r'\bhas property\b',
            r'\bis characterized by\b',
        ],
        RelationType.AT_LOCATION: [
            r'\bis (?:found |located )?(?:in|at|on)\b',
            r'\blives? in\b',
            r'\bexists? in\b',
        ],
        RelationType.USED_FOR: [
            r'\bis used for\b',
            r'\bserves to\b',
            r'\bis designed for\b',
        ],
        RelationType.CAUSES: [
            r'\bcauses?\b',
            r'\bleads? to\b',
            r'\bresults? in\b',
        ],
    }
    
    def extract_relation(self, text: str) -> Tuple[RelationType, float]:
        """
        Extract the relation type from text.
        
        Args:
            text: Normalized text
            
        Returns:
            (RelationType, confidence)
        """
        text_lower = text.lower()
        
        # Check patterns in order of specificity
        for rel_type, patterns in self.RELATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return rel_type, 0.9
        
        # Default fallback
        return RelationType.UNKNOWN, 0.3

    def determine_polarity(self, text: str) -> bool:
        """
        Determine if the statement is positive or negative.
        
        Returns:
            True if positive, False if negative
        """
        negation_patterns = [
            r'\bcannot\b', r'\bcan\'t\b', r'\bnot\b', r'\bnever\b',
            r'\bno\b', r'\bnone\b', r'\bunable\b', r'\bincapable\b'
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, text.lower()):
                return False
        return True


class SemanticInconsistencyDetector:
    """
    Main SID Class
    
    Orchestrates the full pipeline:
    1. Text normalization
    2. Entity extraction
    3. Relation extraction
    4. Symbolic mapping
    5. Conflict detection
    """
    
    def __init__(self, conceptnet_client: Optional[ConceptNetClient] = None):
        """
        Initialize SID.
        
        Args:
            conceptnet_client: ConceptNet client for conflict detection.
                             If None, creates a default client.
        """
        self.normalizer = TextNormalizer()
        self.entity_extractor = EntityExtractor(self.normalizer)
        self.relation_extractor = RelationExtractor()
        self.conceptnet = conceptnet_client or create_client()
    
    def analyze(self, text: str) -> Optional[ClaimClassification]:
        """
        Full analysis pipeline for a text claim.
        
        Args:
            text: Input text (e.g., "Penguins can fly.")
            
        Returns:
            ClaimClassification with conflict analysis and gating decision
        """
        # Stage 1: Normalize
        normalized = self.normalizer.normalize(text)
        
        # Stage 2: Extract entities
        subject, obj = self.entity_extractor.extract_entities(text)
        if not subject or not obj:
            return None
        
        # Stage 3: Extract relation
        rel_type, confidence = self.relation_extractor.extract_relation(normalized)
        
        # Adjust for negation
        is_positive = self.relation_extractor.determine_polarity(normalized)
        if rel_type == RelationType.CAPABLE_OF and not is_positive:
            rel_type = RelationType.NOT_CAPABLE_OF
        elif rel_type == RelationType.NOT_CAPABLE_OF and not is_positive:
            rel_type = RelationType.CAPABLE_OF  # Double negative
        
        # Create extracted relation
        relation = ExtractedRelation(
            subject=subject,
            relation_type=rel_type,
            object=obj,
            confidence=confidence,
            source_text=text
        )
        
        # Stage 4 & 5: Symbolic mapping and conflict detection
        subject_norm = subject.normalized
        rel_str = rel_type.value
        obj_norm = obj.normalized
        
        # Detect conflict
        conflict_result = self.conceptnet.detect_conflict(subject_norm, rel_str, obj_norm)
        
        # Classify and determine gating
        classification, gating = self._classify_conflict(conflict_result)
        
        # Get guard-rails if needed
        guard_rails = []
        if conflict_result.has_conflict:
            guard_rails = self.conceptnet.get_guardrail_knowledge(
                subject_norm, rel_str, obj_norm
            )
        
        return ClaimClassification(
            claim=relation,
            conflict_result=conflict_result,
            classification=classification,
            guard_rails=guard_rails,
            gating_decision=gating
        )
    
    def _classify_conflict(self, conflict_result: ConflictResult) -> Tuple[str, str]:
        """
        Classify the conflict and determine gating decision.
        
        Returns:
            (classification, gating_decision)
        """
        if not conflict_result.has_conflict:
            return "safe", "normal_training"
        
        if conflict_result.conflict_type == "direct":
            return "hard_conflict", "gated_training"
        
        if conflict_result.conflict_type == "inherited":
            return "inherited_conflict", "gated_training"
        
        if conflict_result.conflict_type == "exception":
            return "exception_conflict", "gated_training"
        
        return "conditional", "gated_training"
    
    def analyze_batch(self, texts: List[str]) -> List[Optional[ClaimClassification]]:
        """
        Analyze multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of ClaimClassification results
        """
        return [self.analyze(text) for text in texts]
    
    def get_training_batch(
        self, 
        claims: List[str]
    ) -> Tuple[List[str], List[str], Dict]:
        """
        Construct a training batch with gating applied.
        
        This is the core interface for SG-CL training.
        
        Args:
            claims: List of claim texts to process
            
        Returns:
            (safe_claims, gated_claims_with_guardrails, stats)
        """
        safe_claims = []
        gated_claims = []
        stats = {
            "total": len(claims),
            "safe": 0,
            "gated": 0,
            "failed_extraction": 0
        }
        
        for claim in claims:
            result = self.analyze(claim)
            
            if result is None:
                stats["failed_extraction"] += 1
                continue
            
            if result.gating_decision == "normal_training":
                safe_claims.append(claim)
                stats["safe"] += 1
            else:
                # Include original claim + guard-rails
                gated_batch = [claim] + result.guard_rails
                gated_claims.extend(gated_batch)
                stats["gated"] += 1
        
        return safe_claims, gated_claims, stats


# =============================================================================
# Package Init 
# =============================================================================

def create_sid(conceptnet_client: Optional[ConceptNetClient] = None) -> SemanticInconsistencyDetector:
    """Create a SID instance with default settings."""
    return SemanticInconsistencyDetector(conceptnet_client)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Semantic Inconsistency Detector (SID) - Test Suite")
    print("=" * 70)
    
    # Create SID
    sid = create_sid()
    
    # Test cases
    test_cases = [
        # Conflicting claims
        "Penguins can fly.",
        "Dogs can fly.",
        "Fish can walk.",
        
        # Non-conflicting claims
        "Penguins can swim.",
        "Birds can fly.",
        "Dogs can run.",
        
        # IsA relations
        "A penguin is a bird.",
        "A dog is a mammal.",
        
        # Properties
        "Ice is cold.",
        "Fire is hot.",
        
        # Exception patterns
        "Bats can fly.",  # Mammal that CAN fly (exception)
    ]
    
    print("\nAnalyzing claims:\n")
    
    for claim in test_cases:
        print(f"Claim: \"{claim}\"")
        result = sid.analyze(claim)
        
        if result:
            print(f"  Triple: {result.claim.to_triple()}")
            print(f"  Conflict: {result.conflict_result.has_conflict}")
            print(f"  Classification: {result.classification}")
            print(f"  Gating: {result.gating_decision}")
            if result.guard_rails:
                print(f"  Guard-rails: {result.guard_rails[:2]}...")
        else:
            print("  [Extraction failed]")
        print()
    
    # Test batch processing
    print("=" * 70)
    print("Testing batch processing for training:\n")
    
    batch_claims = [
        "Penguins can fly.",
        "Penguins can swim.", 
        "Dogs can bark.",
        "Cats can fly.",
    ]
    
    safe, gated, stats = sid.get_training_batch(batch_claims)
    
    print(f"Stats: {stats}")
    print(f"\nSafe claims (normal training): {safe}")
    print(f"\nGated claims (with guard-rails): {gated}")
    
    print("\n" + "=" * 70)
    print("SID tests completed!")
    print("=" * 70)
