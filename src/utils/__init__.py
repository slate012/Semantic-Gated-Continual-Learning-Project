"""
SG-CL Utilities Package
"""

from .conceptnet_client import (
    ConceptNetClient,
    ConceptNetEdge,
    ConflictResult,
    create_client
)

from .local_knowledge import (
    get_local_edges,
    concept_exists,
    get_all_concepts,
    get_all_relations,
    LOCAL_KNOWLEDGE
)

__all__ = [
    'ConceptNetClient',
    'ConceptNetEdge', 
    'ConflictResult',
    'create_client',
    'get_local_edges',
    'concept_exists',
    'get_all_concepts',
    'get_all_relations',
    'LOCAL_KNOWLEDGE'
]
