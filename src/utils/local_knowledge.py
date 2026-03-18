"""
Local Fallback Knowledge Base for ConceptNet
=============================================
This module provides a curated local knowledge base as a fallback when
the ConceptNet API is unavailable. It contains key facts needed for
SG-CL conflict detection and guard-rail generation.
"""

# Format: (subject, relation, object, weight, surface_text)
LOCAL_KNOWLEDGE = [
    # =========================================================================
    # PENGUINS (Classic exception case)
    # =========================================================================
    ("penguin", "IsA", "bird", 4.0, "A penguin is a bird."),
    ("penguin", "IsA", "animal", 3.5, "A penguin is an animal."),
    ("penguin", "IsA", "flightless_bird", 3.0, "A penguin is a flightless bird."),
    ("penguin", "CapableOf", "swim", 3.5, "Penguins can swim."),
    ("penguin", "CapableOf", "dive", 3.0, "Penguins can dive."),
    ("penguin", "CapableOf", "catch_fish", 2.5, "Penguins can catch fish."),
    ("penguin", "NotCapableOf", "fly", 4.0, "Penguins cannot fly."),
    ("penguin", "HasProperty", "black_and_white", 2.5, "Penguins are black and white."),
    ("penguin", "AtLocation", "antarctica", 3.0, "Penguins are found in Antarctica."),
    
    # =========================================================================
    # BIRDS (General category)
    # =========================================================================
    ("bird", "IsA", "animal", 4.0, "A bird is an animal."),
    ("bird", "IsA", "vertebrate", 3.0, "A bird is a vertebrate."),
    ("bird", "CapableOf", "fly", 4.0, "Birds can fly."),
    ("bird", "CapableOf", "lay_eggs", 3.5, "Birds can lay eggs."),
    ("bird", "CapableOf", "build_nest", 3.0, "Birds can build nests."),
    ("bird", "HasProperty", "feathers", 4.0, "Birds have feathers."),
    ("bird", "HasProperty", "wings", 3.5, "Birds have wings."),
    
    # =========================================================================
    # OTHER FLIGHTLESS BIRDS
    # =========================================================================
    ("ostrich", "IsA", "bird", 4.0, "An ostrich is a bird."),
    ("ostrich", "IsA", "flightless_bird", 3.0, "An ostrich is a flightless bird."),
    ("ostrich", "NotCapableOf", "fly", 4.0, "Ostriches cannot fly."),
    ("ostrich", "CapableOf", "run", 3.5, "Ostriches can run."),
    
    ("emu", "IsA", "bird", 4.0, "An emu is a bird."),
    ("emu", "IsA", "flightless_bird", 3.0, "An emu is a flightless bird."),
    ("emu", "NotCapableOf", "fly", 4.0, "Emus cannot fly."),
    
    ("kiwi", "IsA", "bird", 4.0, "A kiwi is a bird."),
    ("kiwi", "NotCapableOf", "fly", 4.0, "Kiwis cannot fly."),
    
    # =========================================================================
    # FISH
    # =========================================================================
    ("fish", "IsA", "animal", 4.0, "A fish is an animal."),
    ("fish", "CapableOf", "swim", 4.0, "Fish can swim."),
    ("fish", "NotCapableOf", "walk", 3.5, "Fish cannot walk."),
    ("fish", "NotCapableOf", "fly", 3.0, "Fish cannot fly."),
    ("fish", "HasProperty", "gills", 4.0, "Fish have gills."),
    ("fish", "AtLocation", "water", 4.0, "Fish live in water."),
    
    ("salmon", "IsA", "fish", 4.0, "A salmon is a fish."),
    ("salmon", "CapableOf", "swim", 3.5, "Salmon can swim."),
    ("salmon", "CapableOf", "jump", 2.5, "Salmon can jump."),
    
    ("flying_fish", "IsA", "fish", 3.5, "A flying fish is a fish."),
    ("flying_fish", "CapableOf", "glide", 3.0, "Flying fish can glide."),
    
    # =========================================================================
    # MAMMALS
    # =========================================================================
    ("mammal", "IsA", "animal", 4.0, "A mammal is an animal."),
    ("mammal", "CapableOf", "give_birth", 3.5, "Mammals can give birth to live young."),
    ("mammal", "HasProperty", "warm_blooded", 4.0, "Mammals are warm-blooded."),
    
    ("dog", "IsA", "mammal", 4.0, "A dog is a mammal."),
    ("dog", "IsA", "pet", 3.5, "A dog is a pet."),
    ("dog", "CapableOf", "bark", 4.0, "Dogs can bark."),
    ("dog", "CapableOf", "run", 3.5, "Dogs can run."),
    ("dog", "NotCapableOf", "fly", 3.5, "Dogs cannot fly."),
    
    ("cat", "IsA", "mammal", 4.0, "A cat is a mammal."),
    ("cat", "IsA", "pet", 3.5, "A cat is a pet."),
    ("cat", "CapableOf", "climb", 3.5, "Cats can climb."),
    ("cat", "CapableOf", "meow", 4.0, "Cats can meow."),
    ("cat", "NotCapableOf", "fly", 3.5, "Cats cannot fly."),
    
    ("bat", "IsA", "mammal", 4.0, "A bat is a mammal."),
    ("bat", "CapableOf", "fly", 4.0, "Bats can fly."),
    ("bat", "HasProperty", "nocturnal", 3.0, "Bats are nocturnal."),
    
    ("whale", "IsA", "mammal", 4.0, "A whale is a mammal."),
    ("whale", "CapableOf", "swim", 4.0, "Whales can swim."),
    ("whale", "NotCapableOf", "walk", 3.5, "Whales cannot walk."),
    ("whale", "AtLocation", "ocean", 4.0, "Whales live in the ocean."),
    
    ("dolphin", "IsA", "mammal", 4.0, "A dolphin is a mammal."),
    ("dolphin", "CapableOf", "swim", 4.0, "Dolphins can swim."),
    ("dolphin", "CapableOf", "jump", 3.0, "Dolphins can jump."),
    
    # =========================================================================
    # HUMANS
    # =========================================================================
    ("human", "IsA", "mammal", 4.0, "A human is a mammal."),
    ("human", "IsA", "primate", 3.5, "A human is a primate."),
    ("human", "CapableOf", "think", 4.0, "Humans can think."),
    ("human", "CapableOf", "speak", 4.0, "Humans can speak."),
    ("human", "CapableOf", "walk", 4.0, "Humans can walk."),
    ("human", "CapableOf", "run", 3.5, "Humans can run."),
    ("human", "CapableOf", "swim", 3.0, "Humans can swim."),
    ("human", "NotCapableOf", "fly", 4.0, "Humans cannot fly."),
    ("human", "NotCapableOf", "breathe_underwater", 4.0, "Humans cannot breathe underwater."),
    
    # =========================================================================
    # VEHICLES
    # =========================================================================
    ("car", "IsA", "vehicle", 4.0, "A car is a vehicle."),
    ("car", "CapableOf", "drive", 3.5, "Cars can drive."),
    ("car", "NotCapableOf", "fly", 3.5, "Cars cannot fly."),
    ("car", "UsedFor", "transportation", 4.0, "Cars are used for transportation."),
    
    ("airplane", "IsA", "vehicle", 4.0, "An airplane is a vehicle."),
    ("airplane", "CapableOf", "fly", 4.0, "Airplanes can fly."),
    ("airplane", "UsedFor", "transportation", 4.0, "Airplanes are used for transportation."),
    
    ("boat", "IsA", "vehicle", 4.0, "A boat is a vehicle."),
    ("boat", "CapableOf", "float", 4.0, "Boats can float."),
    ("boat", "NotCapableOf", "fly", 3.0, "Boats cannot fly."),
    
    # =========================================================================
    # PLANTS
    # =========================================================================
    ("plant", "IsA", "living_thing", 4.0, "A plant is a living thing."),
    ("plant", "CapableOf", "grow", 4.0, "Plants can grow."),
    ("plant", "CapableOf", "photosynthesize", 3.5, "Plants can photosynthesize."),
    ("plant", "NotCapableOf", "move", 3.0, "Plants cannot move."),
    
    ("tree", "IsA", "plant", 4.0, "A tree is a plant."),
    ("tree", "CapableOf", "grow", 4.0, "Trees can grow."),
    
    # =========================================================================
    # ABSTRACT CONCEPTS
    # =========================================================================
    ("ice", "HasProperty", "cold", 4.0, "Ice is cold."),
    ("ice", "HasProperty", "solid", 3.5, "Ice is solid."),
    ("fire", "HasProperty", "hot", 4.0, "Fire is hot."),
    ("water", "HasProperty", "wet", 4.0, "Water is wet."),
    ("water", "HasProperty", "liquid", 3.5, "Water is liquid."),
    
    # =========================================================================
    # EXCEPTION PATTERNS (for testing SG-CL)
    # =========================================================================
    ("platypus", "IsA", "mammal", 4.0, "A platypus is a mammal."),
    ("platypus", "CapableOf", "lay_eggs", 3.5, "Platypuses can lay eggs."),
    ("platypus", "CapableOf", "swim", 3.0, "Platypuses can swim."),
    ("platypus", "HasProperty", "venomous", 2.5, "Platypuses are venomous."),
]


def get_local_edges(subject=None, relation=None, obj=None):
    """
    Query the local knowledge base.
    
    Args:
        subject: Filter by subject (optional)
        relation: Filter by relation (optional)  
        obj: Filter by object (optional)
        
    Returns:
        List of matching tuples
    """
    results = []
    for entry in LOCAL_KNOWLEDGE:
        s, r, o, w, text = entry
        
        if subject and s.lower() != subject.lower():
            continue
        if relation and r != relation:
            continue
        if obj and o.lower() != obj.lower():
            continue
            
        results.append(entry)
    
    return results


def concept_exists(concept):
    """Check if a concept exists in local knowledge."""
    concept = concept.lower()
    for s, r, o, w, text in LOCAL_KNOWLEDGE:
        if s.lower() == concept or o.lower() == concept:
            return True
    return False


def get_all_concepts():
    """Get all unique concepts in the knowledge base."""
    concepts = set()
    for s, r, o, w, text in LOCAL_KNOWLEDGE:
        concepts.add(s)
        concepts.add(o)
    return sorted(concepts)


def get_all_relations():
    """Get all unique relations in the knowledge base."""
    return sorted(set(r for s, r, o, w, text in LOCAL_KNOWLEDGE))


if __name__ == "__main__":
    print("Local Knowledge Base Statistics:")
    print(f"  Total facts: {len(LOCAL_KNOWLEDGE)}")
    print(f"  Unique concepts: {len(get_all_concepts())}")
    print(f"  Relations: {get_all_relations()}")
    
    print("\nSample queries:")
    print("\nPenguin facts:")
    for fact in get_local_edges(subject="penguin"):
        print(f"  {fact[0]} --[{fact[1]}]--> {fact[2]}")
    
    print("\nAll NotCapableOf facts:")
    for fact in get_local_edges(relation="NotCapableOf"):
        print(f"  {fact[0]} cannot {fact[2]}")
