"""
task1_simulation.py

Dice-based "football match" simulation (Task 1: parts a & c).
No classes used. Uses only functions and plain dicts.

Run:
    python task1_simulation.py

This will run:
 - a seeded example match (storytelling)
 - a Monte Carlo comparison (default 500 simulations) of:
     * no shot clock
     * shot clock (limit configurable)
"""

import random
import statistics
from collections import defaultdict

# ---------------------------
# Dice definitions (DO NOT EDIT)
# ---------------------------
red = ['through ball/to yellow','inside pass/to blue',
       'dribble/throw again','short pass/to black',
       'through ball/to green','tackled and lost']

black = ['pass back/to red','throw in/to blue','shoot',
         'free kick/to yellow','long shot at goal','tackled and lost']

blue = ['header at goal','shoot','opponent shown yellow card',
        'pass back/to red','long shot at goal','tackled and lost']

yellow = ['pass back/to black','off side','tackled and lost',
          'penalty','shoot','shoot']

green = ['goal','wide','goal','over bar','saved','corner/to yellow']

DICE = {"red": red, "black": black, "blue": blue, "yellow": yellow, "green": green}

# ---------------------------
# Utility functions
# ---------------------------
def roll_dice(color: str, rng: random.Random) -> str:
    """Return one face outcome from the dice of given color."""
    return rng.choice(DICE[color])

def is_pass_face(face: str) -> bool:
    return "/to " in face

def pass_to_color(face: str) -> str:
    # Assumes the face contains "/to <color>"
    return face.split("/to ")[1].strip()

def is_shot_face(face: str) -> bool:
    lower = face.lower()
    return ("shoot" in lower) or ("header" in lower) or ("penalty" in lower) or ("long shot" in lower)

def is_turnover_face(face: str) -> bool:
    lower = face.lower()
    return ("tackled and lost" in lower) or ("off side" in lower)

# ---------------------------
# Single-match simulator
# ---------------------------
def simulate_single_match(turns: int = 30,
                          rng_seed: int = None,
                          shot_clock_limit: int | None = None,
                          hail_success_factor: float = 0.35,
                          verbose: bool = False):
    """
    Simulate a single match.
    - turns: number of dice-roll units (dice rolls) to simulate.
    - rng_seed: for reproducibility.
    - shot_clock_limit: if int, enforce shot clock (None disables it).
    - hail_success_factor: probability modifier for Hail Mary (applies only if shot clock expires).
    Returns a dict with stats.
    """
    rng = random.Random(rng_seed)

    state = {
        "score": {"Player 1": 0, "Player 2": 0},
        "shots": {"Player 1": 0, "Player 2": 0},
        "possession_rolls": {"Player 1": 0, "Player 2": 0},
        "current_player": "Player 1",
        "current_dice": "red",
        "shot_clock": 0
    }

    for t in range(1, turns + 1):
        player = state["current_player"]
        state["possession_rolls"][player] += 1
        state["shot_clock"] += 1

        face = roll_dice(state["current_dice"], rng)
        if verbose:
            print(f"[Turn {t}] {player} rolls {state['current_dice'].upper()} -> {face}")

        # 1) If the face indicates a pass to a colour (e.g., '/to yellow')
        if is_pass_face(face):
            next_color = pass_to_color(face)
            state["current_dice"] = next_color
            if verbose:
                print(f"    PASS to {next_color.upper()} (possession kept)")

        # 2) If it's a shot-triggering face
        elif is_shot_face(face):
            # Count the shot
            state["shots"][player] += 1
            state["shot_clock"] = 0  # reset shot clock on taking a shot
            if verbose:
                print(f"    {player} attempts a shot!")

            green_face = roll_dice("green", rng)
            if verbose:
                print(f"       GREEN -> {green_face}")

            # If green shows a goal (literal 'goal'), score
            if green_face == "goal":
                state["score"][player] += 1
                if verbose:
                    print(f"       ⚽ GOAL for {player}! Score now: {state['score']}")
                # after a goal, possession changes
                state["current_player"] = "Player 2" if player == "Player 1" else "Player 1"
                state["current_dice"] = "red"
            elif is_pass_face(green_face):
                # e.g., corner/to yellow -- attacker keeps possession and dice moves to that color
                next_color = pass_to_color(green_face)
                state["current_dice"] = next_color
                if verbose:
                    print(f"       CORNER -> play continues with {player} on {next_color.upper()}")
                # possession stays same
            else:
                # saved/wide/over bar -> possession changes
                state["current_player"] = "Player 2" if player == "Player 1" else "Player 1"
                state["current_dice"] = "red"
                if verbose:
                    print(f"       Miss/Save -> possession to {state['current_player']}")

        # 3) Direct 'goal' face (if that occurs on some dice; treat as goal)
        elif "goal" in face.lower():
            state["score"][player] += 1
            if verbose:
                print(f"    ⚽ Direct GOAL for {player}! Score now: {state['score']}")
            state["current_player"] = "Player 2" if player == "Player 1" else "Player 1"
            state["current_dice"] = "red"
            state["shot_clock"] = 0

        # 4) Turnovers
        elif is_turnover_face(face):
            state["current_player"] = "Player 2" if player == "Player 1" else "Player 1"
            state["current_dice"] = "red"
            state["shot_clock"] = 0
            if verbose:
                print(f"    Turnover -> possession to {state['current_player']}")

        # 5) Dribble/throw again (we interpret as keep and re-roll same color)
        elif "throw again" in face.lower():
            if verbose:
                print("    Dribble / keep possession -> immediate re-roll next turn same player")

        # 6) Other faces (pass back, opponent shown yellow card etc.)
        else:
            # default: change possession if face suggests lost play? otherwise keep possession and move on
            if verbose:
                print("    No special rule -> continue with current possession and dice reset to red")
            state["current_dice"] = "red"

        # -------------------------
        # Shot clock enforcement: Hail Mary when shot_clock >= limit
        # -------------------------
        if shot_clock_limit is not None and state["shot_clock"] >= shot_clock_limit:
            # Force a Hail Mary shot
            attacker = state["current_player"]
            if verbose:
                print(f"    ⏱ Shot clock expired for {attacker}! Hail Mary attempt.")
            state["shots"][attacker] += 1
            state["shot_clock"] = 0
            hail_face = roll_dice("green", rng)
            # Only a green 'goal' can be considered; apply hail success modifier (realism)
            if hail_face == "goal" and rng.random() < hail_success_factor:
                state["score"][attacker] += 1
                if verbose:
                    print(f"    ⚽ HAIL MARY GOAL for {attacker}! Score: {state['score']}")
            else:
                # miss -> possession changes
                if verbose:
                    print(f"    Hail Mary missed ({hail_face}). Possession changes.")
                state["current_player"] = "Player 2" if attacker == "Player 1" else "Player 1"
                state["current_dice"] = "red"

    # Compile stats
    stats = {
        "score": dict(state["score"]),
        "shots": dict(state["shots"]),
        "possession_rolls": dict(state["possession_rolls"])
    }
    return stats

# ---------------------------
# Monte Carlo aggregator
# ---------------------------
def simulate_n_matches(n: int = 500,
                       turns: int = 60,
                       shot_clock_limit: int | None = None,
                       hail_success_factor: float = 0.35,
                       base_seed: int = 0):
    """
    Run n simulations and aggregate:
        - average goals per match per player
        - average shots per match per player
        - average possession% per player
        - conversion rates (goals/shots)
    """
    agg_goals = {"Player 1": [], "Player 2": []}
    agg_shots = {"Player 1": [], "Player 2": []}
    agg_possession_pct = {"Player 1": [], "Player 2": []}

    for i in range(n):
        seed = base_seed + i
        stats = simulate_single_match(turns=turns,
                                      rng_seed=seed,
                                      shot_clock_limit=shot_clock_limit,
                                      hail_success_factor=hail_success_factor,
                                      verbose=False)
        # append numeric stats
        g1, g2 = stats["score"]["Player 1"], stats["score"]["Player 2"]
        s1, s2 = stats["shots"]["Player 1"], stats["shots"]["Player 2"]
        p1_rolls = stats["possession_rolls"]["Player 1"]
        p2_rolls = stats["possession_rolls"]["Player 2"]
        total_rolls = p1_rolls + p2_rolls
        agg_goals["Player 1"].append(g1)
        agg_goals["Player 2"].append(g2)
        agg_shots["Player 1"].append(s1)
        agg_shots["Player 2"].append(s2)
        agg_possession_pct["Player 1"].append(p1_rolls / total_rolls * 100)
        agg_possession_pct["Player 2"].append(p2_rolls / total_rolls * 100)

    # compute summary statistics
    summary = {}
    for p in ("Player 1", "Player 2"):
        summary[p] = {
            "avg_goals": statistics.mean(agg_goals[p]),
            "std_goals": statistics.pstdev(agg_goals[p]),
            "avg_shots": statistics.mean(agg_shots[p]),
            "std_shots": statistics.pstdev(agg_shots[p]),
            "avg_possession_pct": statistics.mean(agg_possession_pct[p]),
            "std_possession_pct": statistics.pstdev(agg_possession_pct[p]),
        }
        # conversion rate: use total sums to avoid divide-by-zero
        total_goals = sum(agg_goals[p])
        total_shots = sum(agg_shots[p])
        summary[p]["overall_conversion"] = (total_goals / total_shots) if total_shots > 0 else 0.0

    return summary

# ---------------------------
# Driver: example runs & comparison
# ---------------------------
if __name__ == "__main__":
    print("=== Task 1 — Single example match (verbose) ===\n")
    example_stats = simulate_single_match(turns=20, rng_seed=42, shot_clock_limit=None, verbose=True)
    print("\nExample match stats:", example_stats)

    print("\n=== Task 1 — Monte Carlo comparison ===")
    N = 400  # changeable; 400 is reasonable for a quick run
    turns = 40

    print(f"\nRunning {N} sims without shot clock...")
    res_no_clock = simulate_n_matches(n=N, turns=turns, shot_clock_limit=None, base_seed=1000)
    print("No shot clock summary (averages):")
    for p in res_no_clock:
        print(p, res_no_clock[p])

    print(f"\nRunning {N} sims with shot clock = 5 and hail_success_factor = 0.35 ...")
    res_with_clock = simulate_n_matches(n=N, turns=turns, shot_clock_limit=5, hail_success_factor=0.35, base_seed=5000)
    print("Shot clock summary (averages):")
    for p in res_with_clock:
        print(p, res_with_clock[p])

    # Simple comparison
    print("\nSimple comparison (avg goals):")
    print("Player 1: no_clock avg goals =", res_no_clock["Player 1"]["avg_goals"],
          " vs shot_clock avg goals =", res_with_clock["Player 1"]["avg_goals"])
    print("Player 2: no_clock avg goals =", res_no_clock["Player 2"]["avg_goals"],
          " vs shot_clock avg goals =", res_with_clock["Player 2"]["avg_goals"])

