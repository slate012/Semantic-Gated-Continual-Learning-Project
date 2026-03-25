import json
import random
import os
import sys

# Add src to path to import local_knowledge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from utils.local_knowledge import LOCAL_KNOWLEDGE

# -----------------------------
# CONFIG
# -----------------------------
TOTAL_TASKS = 5
SAMPLES_PER_TASK = 1000
SAFE_RATIO = 0.6
CONFLICT_RATIO = 0.4
EVAL_SIZE = 1000

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)

# -----------------------------
# DATA GENERATION UTILS
# -----------------------------
def generate_safe_variations(kb, target_count):
    """Generate safe claims using local knowledge with variations to multiply data."""
    base_claims = [item[4] for item in kb]
    prefixes = ["", "It is known that ", "Fact: ", "Generally, ", "In reality, ", "It is true that "]
    
    variations = []
    while len(variations) < target_count:
        claim = random.choice(base_claims)
        prefix = random.choice(prefixes)
        
        # Adjust casing if applying a prefix
        if prefix != "":
            claim = claim[0].lower() + claim[1:]
            
        variations.append(prefix + claim)
        
    return variations

def generate_synthetic_conflicts(kb, target_count):
    """Generate conflicting claims to trigger SG-CL guard-rails."""
    conflicts = []
    
    # Extract entities and relations
    subjects = list(set([item[0] for item in kb]))
    objects = list(set([item[2] for item in kb]))
    
    conflict_templates = {
        "IsA": [
            "{subject} is a type of {obj}.",
            "A {subject} is a {obj}."
        ],
        "CapableOf": [
            "{subject}s can {obj}.",
            "The {subject} is capable of {obj}."
        ],
        "NotCapableOf": [
            "{subject}s are easily able to {obj}.",
            "A {subject} can {obj}."
        ],
        "HasProperty": [
            "{subject}s are generally {obj}.",
            "The {subject} has the property of being {obj}."
        ],
        "AtLocation": [
            "{subject}s are usually found in {obj}."
        ]
    }
    
    while len(conflicts) < target_count:
        # Pick a random true fact
        true_fact = random.choice(kb)
        s, r, o, _, _ = true_fact
        
        # Corrupt it deliberately
        corruption_type = random.choice(["wrong_subject", "wrong_object"])
        
        new_s, new_o = s, o
        if corruption_type == "wrong_subject":
            new_s = random.choice([subj for subj in subjects if subj != s])
        elif corruption_type == "wrong_object":
            new_o = random.choice([obj for obj in objects if obj != o])
            
        # Format strings for presentation
        fmt_s = new_s.replace("_", " ").capitalize()
        fmt_o = new_o.replace("_", " ")
        
        # Prevent "Fishs" or "Penguins" plural issues by basic replace, though simple for this demo
        if fmt_s.endswith("s"):
            fmt_s = fmt_s[:-1]
            
        template = random.choice(conflict_templates.get(r, ["{subject}s and {obj}s are related."]))
        sentence = template.format(subject=fmt_s, obj=fmt_o)
        
        # Avoid accidentally generating a true fact that exists in KB
        if not any(entry[4].lower() == sentence.lower() for entry in kb):
            
            # Create the prompt string missing the object
            prompt_str = template.format(subject=fmt_s, obj="").replace(" .", "").replace("..", ".").strip()
            
            # Store the conflict along with triple metadata for evaluation
            conflicts.append({
                "sentence": sentence,
                "prompt": prompt_str,
                "expected": fmt_o,
                "subject": new_s,
                "relation": r,
                "object": new_o,
            })
            
    return conflicts

# -----------------------------
# BUILD TASKS
# -----------------------------
def build_tasks(safe_variations, conflicts):
    tasks = []
    
    for t in range(TOTAL_TASKS):
        safe_count = int(SAMPLES_PER_TASK * SAFE_RATIO)
        conflict_count = int(SAMPLES_PER_TASK * CONFLICT_RATIO)

        # progressively increase conflicts simulating harder tasks
        conflict_boost = int((t / TOTAL_TASKS) * 200)
        conflict_count += conflict_boost
        safe_count -= conflict_boost

        # Randomly sample to build the task list
        task_safe = random.sample(safe_variations, safe_count)
        task_conflicts = [c["sentence"] for c in random.sample(conflicts, conflict_count)]
        
        task_data = task_safe + task_conflicts
        random.shuffle(task_data)
        tasks.append(task_data)

    return tasks

# -----------------------------
# RELATION-TO-QUESTION TEMPLATES
# -----------------------------
QUESTION_TEMPLATES = {
    "IsA":          "Is a {subject} a type of {object}?",
    "CapableOf":    "Can a {subject} {object}?",
    "NotCapableOf": "Can a {subject} {object}?",
    "HasProperty":  "Is a {subject} {object}?",
    "AtLocation":   "Is a {subject} found in {object}?",
    "UsedFor":      "Is a {subject} used for {object}?",
}

def _make_question(subject, relation, obj):
    """Convert a (subject, relation, object) triple into a natural-language question."""
    fmt_s = subject.replace("_", " ")
    fmt_o = obj.replace("_", " ")
    template = QUESTION_TEMPLATES.get(relation, "Is {subject} related to {object}?")
    return template.format(subject=fmt_s, object=fmt_o)

def _expected_answer(relation):
    """Return the expected yes/no answer for a given relation type."""
    if relation == "NotCapableOf":
        return "no"
    return "yes"

# -----------------------------
# BUILD EVAL SET
# -----------------------------
def build_eval_set(kb, conflicts):
    """
    Build an evaluation set in the format expected by evaluate_model.py:
    {
      "old_knowledge": [{"question": ..., "expected": ..., "category": ..., "subject": ..., "relation": ..., "object": ...}, ...],
      "new_knowledge": [...]
    }
    """
    old_knowledge = []
    new_knowledge = []

    # --- Old knowledge (preservation testing) ---
    # Use real facts from the knowledge base
    for _ in range(EVAL_SIZE // 2):
        fact = random.choice(kb)
        s, r, o, _, _ = fact

        old_knowledge.append({
            "question": _make_question(s, r, o),
            "expected": _expected_answer(r),
            "category": r,
            "subject": s,
            "relation": r,
            "object": o,
        })

    # --- New knowledge (conflict acquisition testing) ---
    # These are the corrupted facts the model was trained on;
    # we want to check if the model blindly memorised them.
    conflict_sample = random.sample(conflicts, min(EVAL_SIZE // 2, len(conflicts)))

    for c in conflict_sample:
        # Reverse-engineer subject/object from the sentence
        new_knowledge.append({
            "question": c["sentence"].rstrip(".") + "?",
            "expected": "yes",
            "category": "conflict",
            "subject": c.get("subject", ""),
            "relation": c.get("relation", ""),
            "object": c.get("object", ""),
        })

    return {"old_knowledge": old_knowledge, "new_knowledge": new_knowledge}

# -----------------------------
# SAVE FILES
# -----------------------------
def save_tasks(tasks):
    for i, task in enumerate(tasks):
        path = os.path.join(OUTPUT_DIR, f"train_task_{i+1}.txt")
        with open(path, "w") as f:
            for line in task:
                f.write(line + "\n")

def save_eval(eval_data):
    path = os.path.join(OUTPUT_DIR, "evaluation_set.json")
    with open(path, "w") as f:
        json.dump(eval_data, f, indent=2)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("Loading Local Knowledge Base from src/utils/local_knowledge.py...")
    
    # We generate a large pool (double the required total size) to safely sample from without crashing
    pool_size = (TOTAL_TASKS * SAMPLES_PER_TASK) * 2
    
    print(f"Generating {pool_size} synthetic safe facts via rule variations...")
    safe_variations = generate_safe_variations(LOCAL_KNOWLEDGE, pool_size)

    print(f"Generating {pool_size} synthetic contradictions/counterfactuals...")
    conflict_data = generate_synthetic_conflicts(LOCAL_KNOWLEDGE, pool_size)

    print("Building progressively difficult training tasks...")
    tasks = build_tasks(safe_variations, conflict_data)

    print("Building evaluation question/answer JSON...")
    eval_data = build_eval_set(LOCAL_KNOWLEDGE, conflict_data)

    print("Saving files to data/...")
    save_tasks(tasks)
    save_eval(eval_data)

    print(f"✅ SG-CL Synthetic Dataset Generation complete! Check the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()