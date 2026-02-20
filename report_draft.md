# Coursework Reconstruction - Draft Sections

## Task 1b: Design, Incremental Development, and Testing

### Design & Assumptions
The implementation follows a **Top-Down Functional Design** approach. The problem is decomposed into a main match engine (`simulate_match`) and a subordinate possession engine (`run_possession`). 
**Key Assumptions**:
1. **Discrete Time**: Match duration is modeled in "rolls" rather than linear clock time, where each roll represents a significant event or second of play.
2. **Independence**: Each dice roll is an independent stochastic event, representing the chaotic nature of football.
3. **State Consistency**: The "next die color" represents the tactical transition (e.g., from Red/Midfield to Blue/Wings).

### Incremental Development Process
1. **Phase 1: Minimum Viable Simulation**: Implemented the basic `red` dice logic and the loops for a single possession. Simple print statements verified that the simulation could terminate.
2. **Phase 2: Tactical Complexity**: Added the `black`, `blue`, and `yellow` dice logic. Refactored the code into `run_possession` to maintain high cohesion.
3. **Phase 3: Storytelling Integration**: Replaced generic output with a dedicated `get_commentary` function to translate raw dice actions into descriptive football narratives.
4. **Phase 4: Rule Enforcement**: Iteratively removed all `break` and `continue` statements, replacing them with state-controlled boolean flags (`possession_ended`) to meet strict University coding standards.

### Testing Strategy
Testing was performed using **White-Box Verification**:
- **Boundary Testing**: Verified that total elapsed time never exceeds the match limit.
- **Path Testing**: Forced specific dice colors using temporary seeds to ensure transitions (e.g., 'to yellow') mapped correctly to the next roll's logic.
- **Assertion Validation**: Used manual trace checks of the storytelling output to ensure "Goal" outcomes correctly reset possession to the other player.

---

## Task 1c: Modification Analysis (Shot Clock & Hail Mary)

### The "Hail Mary" Implementation
The modification introduces a **Shot Clock** (limiting possession duration) and a high-risk **Hail Mary** shot.
**Modified Logic**:
```python
# [HIGHLIGHTED MODIFICATION]
if roll_count >= shot_clock_limit:
    # Force immediate attempt on Green dice
    hail_face = random.choice(green)
    # Result determined by Green outcome
```
**Evaluation**:
In "Scenario 2" (Shot Clock = 5), total goals increased from 2 to 4. While the shot clock forces more turnovers, the "Hail Mary" mechanic creates high-variance goal-scoring opportunities that would otherwise be lost to endless passing loops. This mirrors modern football strategies where teams are pressured to "get the ball in the box" under time constraints.

---

## Task 2b: Teaching the Sentiment Algorithms

### VADER: The Lexicon Specialist
VADER is a **lexicon-based** model. It uses a "gold-standard" dictionary where words are pre-scored by humans (e.g., "tragedy" is -3.4, "great" is 3.1). 
**How it works**:
1. **Tokenization**: Breaks the sentence into words.
2. **Scoring**: Sums up the scores of each word.
3. **Heuristics**: It applies rules for "intensifiers" (e.g., VERY good increases the score) and "negations" (e.g., NOT good flips the score).
*Think of it as a human with a dictionary and a set of simple grammar rules.*

### DistilBERT: The Contextual Scholar
DistilBERT is a **Transformer** model. Unlike VADER, it doesn't just look at word scores; it looks at word *relationships*.
**The Attention Mechanism**:
When the model reads "The fresher was not happy," the "Attention" mechanism focuses on the relationship between "not" and "happy." While a simple model might see "happy" and rate it positive, DistilBERT "attends" to the negation to understand the full context.
*Think of it as an expert reader who understands nuance, sarcasm, and sentence structure.*

---

## Task 2c: Critical Assessment of Libraries

### Difficulty of Coding
- **VADER**: Extremely low. Requires a simple import and one function call.
- **Transformers/Hugging Face**: Moderate. Requires understanding of pipelines, tokenizers, and model checkpoints. Fine-tuning adds significant complexity regarding tensors and training loops.

### Adaptability
- **VADER**: Limited. You can add words to the lexicon, but you cannot change its underlying "understanding" of language.
- **Transformers**: High. Transfer learning allows us to take a model trained on all of Wikipedia and specialize it for "Coventry Freshers" with relatively little data.

### Quality of Solution
- **VADER** is transparent (you see why it scored a word) but brittle (fails on complex context).
- **Transformers** achieve near-human accuracy (MLO3) but are "Black Boxes"—it is difficult to explain *exactly* why a specific neuron fired.

### Recommendation
For quick, descriptive tasks, I recommend **VADER**. For high-stakes or context-heavy data science problems (e.g., identifying student mental health trends), the **Transformer-based** approach is the superior Choice (Bird et al., 2009; Vaswani et al., 2017).
