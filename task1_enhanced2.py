"""
task1_enhanced.py

Enhanced Task 1:
 - deterministic/storytelling single-match simulator (functions only)
 - Monte Carlo aggregator to compute avg goals/shots/possession
 - Markov-chain (absorbing) analysis of a possession starting from a dice color:
    --> compute probability that possession results in a GOAL vs TURNOVER analytically
 - Compare analytic values to Monte Carlo estimates.

No classes used.
"""

import random
import numpy as np
import math
from collections import defaultdict
import statistics
import os
import json

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
# Utility helpers
# ---------------------------
def roll_from(color, rng):
    return rng.choice(DICE[color])

def is_pass(face):
    return "/to " in face

def pass_color(face):
    return face.split("/to ")[1].strip()

def is_shot(face):
    f = face.lower()
    return ("shoot" in f) or ("header" in f) or ("penalty" in f) or ("long shot" in f)

def is_turnover(face):
    f = face.lower()
    return ("tackled and lost" in f) or ("off side" in f)

# ---------------------------
# Part A: simulator (storytelling version)
# ---------------------------
def simulate_single_match(turns=30, rng_seed=None, shot_clock_limit=None, hail_success=0.35, verbose=True):
    """
    Simulate a match with storytelling prints (verbose=True).
    Returns stats dict: score, shots, possession_rolls.
    """
    rng = random.Random(rng_seed)
    score = {"Player 1": 0, "Player 2": 0}
    shots = {"Player 1": 0, "Player 2": 0}
    poss_rolls = {"Player 1": 0, "Player 2": 0}

    current_player = "Player 1"
    current_die = "red"
    shot_clock = 0

    for t in range(1, turns+1):
        poss_rolls[current_player] += 1
        shot_clock += 1
        face = roll_from(current_die, rng)
        if verbose:
            print(f"[Turn {t}] {current_player} rolls {current_die.upper()} -> {face}")

        # passes
        if is_pass(face):
            current_die = pass_color(face)
            if verbose:
                print(f"  PASS to {current_die.upper()}")
            # possession stays

        elif is_shot(face):
            shots[current_player] += 1
            shot_clock = 0
            if verbose:
                print(f"  {current_player} SHOOTS!")
            green = roll_from("green", rng)
            if verbose:
                print(f"    GREEN -> {green}")
            if green == "goal":
                score[current_player] += 1
                if verbose:
                    print(f"    ⚽ GOAL for {current_player} (score {score})")
                current_player = "Player 2" if current_player == "Player 1" else "Player 1"
                current_die = "red"
            elif is_pass(green):
                current_die = pass_color(green)
                if verbose:
                    print(f"    SET PIECE -> keep possession, next die {current_die}")
            else:
                # missed/saved/wide
                current_player = "Player 2" if current_player == "Player 1" else "Player 1"
                current_die = "red"
                if verbose:
                    print("    Miss -> possession changes")

        elif "goal" in face.lower():
            # Occurs rarely on non-green faces; treat as direct goal
            score[current_player] += 1
            if verbose:
                print(f"  ⚽ Direct GOAL for {current_player} (score {score})")
            current_player = "Player 2" if current_player == "Player 1" else "Player 1"
            current_die = "red"
            shot_clock = 0

        elif is_turnover(face):
            if verbose:
                print("  Turnover")
            current_player = "Player 2" if current_player == "Player 1" else "Player 1"
            current_die = "red"
            shot_clock = 0

        elif "throw again" in face.lower():
            if verbose:
                print("  Dribble -> keep possession (re-roll next)")

        else:
            # fallback: reset to red but keep possession
            current_die = "red"

        # shot clock enforcement
        if shot_clock_limit is not None and shot_clock >= shot_clock_limit:
            # Hail Mary
            if verbose:
                print(f"  ⏱ Shot clock expired for {current_player}: Hail Mary!")
            shots[current_player] += 1
            shot_clock = 0
            green = roll_from("green", rng)
            if green == "goal" and rng.random() < hail_success:
                score[current_player] += 1
                if verbose:
                    print(f"    ⚽ HAIL MARY GOAL for {current_player}!")
            else:
                if verbose:
                    print(f"    Hail Mary missed ({green}). Possession changes.")
                current_player = "Player 2" if current_player == "Player 1" else "Player 1"
                current_die = "red"

    if verbose:
        print("\nFinal score:", score)
    return {"score": score, "shots": shots, "possession_rolls": poss_rolls}

# ---------------------------
# Part B: Monte Carlo aggregation
# ---------------------------
def simulate_n_matches(n=500, turns=60, shot_clock_limit=None, hail_success=0.35, base_seed=0):
    results = {"goals": defaultdict(list), "shots": defaultdict(list), "pos_pct": defaultdict(list)}
    for i in range(n):
        seed = base_seed + i
        stats = simulate_single_match(turns=turns, rng_seed=seed, shot_clock_limit=shot_clock_limit,
                                      hail_success=hail_success, verbose=False)
        for p in ("Player 1", "Player 2"):
            results["goals"][p].append(stats["score"][p])
            results["shots"][p].append(stats["shots"][p])
            pr = stats["possession_rolls"][p]
            total = sum(stats["possession_rolls"].values())
            results["pos_pct"][p].append(pr / total * 100 if total>0 else 0.0)

    summary = {}
    for p in ("Player 1", "Player 2"):
        summary[p] = {
            "avg_goals": statistics.mean(results["goals"][p]),
            "avg_shots": statistics.mean(results["shots"][p]),
            "avg_possession_pct": statistics.mean(results["pos_pct"][p]),
            "conv_rate": (sum(results["goals"][p]) / sum(results["shots"][p])) if sum(results["shots"][p])>0 else 0.0
        }
    return summary

# ---------------------------
# Part C: Markov chain (absorbing) analysis
# ---------------------------
def build_transition_matrix_for_possession(rng_sample_size=5000):
    """
    Build empirical transition probabilities P(state -> state') for a single roll in possession.
    States:
      red, black, blue, yellow, green, GOAL (absorb), TURNOVER (absorb)
    We'll sample transitions by enumerating faces exactly (since dice are uniform discrete),
    derive exact probabilities from face counts (no randomness).
    """
    states = ["red", "black", "blue", "yellow", "green", "GOAL", "TURNOVER"]
    idx = {s:i for i,s in enumerate(states)}
    P = np.zeros((len(states), len(states)), dtype=float)

    # For each non-absorbing color, go through its faces and add transitions
    for color in ["red","black","blue","yellow","green"]:
        faces = DICE[color]
        for face in faces:
            if color != "green" and "goal" in face.lower():
                # treat direct goal on non-green
                P[idx[color], idx["GOAL"]] += 1.0/len(faces)
            elif is_pass(face):
                to = pass_color(face)
                P[idx[color], idx[to]] += 1.0/len(faces)
            elif is_shot(face) and color != "green":
                # shot -> resolved by green die: handle probabilities using green faces
                gf = DICE["green"]
                for gface in gf:
                    if gface == "goal":
                        P[idx[color], idx["GOAL"]] += (1.0/len(faces))*(1.0/len(gf))
                    elif is_pass(gface):
                        P[idx[color], idx[pass_color(gface)]] += (1.0/len(faces))*(1.0/len(gf))
                    else:
                        P[idx[color], idx["TURNOVER"]] += (1.0/len(faces))*(1.0/len(gf))
            elif color == "green":
                # resolve green face directly
                if face == "goal":
                    P[idx["green"], idx["GOAL"]] += 1.0/len(faces)
                elif is_pass(face):
                    P[idx["green"], idx[pass_color(face)]] += 1.0/len(faces)
                else:
                    # saved/wide/over bar -> turnover
                    P[idx["green"], idx["TURNOVER"]] += 1.0/len(faces)
            elif is_turnover(face):
                P[idx[color], idx["TURNOVER"]] += 1.0/len(faces)
            elif "throw again" in face.lower():
                # keep same color: this is important
                P[idx[color], idx[color]] += 1.0/len(faces)
            else:
                # fallback: reset to red (possession kept)
                P[idx[color], idx["red"]] += 1.0/len(faces)

    # Ensure rows sum to 1 for non-absorbing, absorbing rows identity
    for s in ["GOAL","TURNOVER"]:
        i = idx[s]
        P[i,i] = 1.0

    return states, P

def compute_absorption_probabilities(states, P):
    """
    Given transition matrix P with absorbing states GOAL and TURNOVER,
    compute fundamental matrix N = (I - Q)^-1 and B = N * R
    where Q = transitions among transient states, R = transitions from transient to absorbing.
    Returns mapping: for each transient state, probability of absorption in GOAL and TURNOVER.
    """
    idx = {s:i for i,s in enumerate(states)}
    # identify transient states (non-absorbing)
    absorbing = ["GOAL","TURNOVER"]
    trans_states = [s for s in states if s not in absorbing]
    t = len(trans_states)
    Q = np.zeros((t,t))
    R = np.zeros((t, len(absorbing)))
    for i, si in enumerate(trans_states):
        for j, sj in enumerate(trans_states):
            Q[i,j] = P[idx[si], idx[sj]]
        for j, ab in enumerate(absorbing):
            R[i,j] = P[idx[si], idx[ab]]
    # Fundamental matrix
    I = np.eye(t)
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        # Handle near-singularity with pseudo-inverse
        N = np.linalg.pinv(I - Q)
    B = N.dot(R)
    # B[i,0] is probability start at trans_states[i] -> GOAL
    result = {}
    for i, s in enumerate(trans_states):
        result[s] = {"P(goal)": float(B[i,0]), "P(turnover)": float(B[i,1])}
    return result

# ---------------------------
# Driver: run storytelling, analytic & Monte Carlo and compare
# ---------------------------
if __name__ == "__main__":
    # Prepare output folder
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)

    # Helper to write output to file
    def write_output(filename, content, mode="w"):
        with open(os.path.join(output_dir, filename), mode, encoding="utf-8") as f:
            f.write(content)

    # 1. Storytelling Example
    print("\n=== STORYTELLING EXAMPLE (Task 1 part a) ===\n")
    ex = simulate_single_match(turns=90, rng_seed=123, shot_clock_limit=None, verbose=True)
    print("\nExample stats:", ex)
    # Save storytelling stats to file
    write_output("storytelling_stats.json", json.dumps(ex, indent=2))

    # 2. Markov Chain Analysis
    print("\n=== MARKOV CHAIN ANALYSIS ===")
    states, P = build_transition_matrix_for_possession()
    analytic = compute_absorption_probabilities(states, P)
    print("Analytic absorption probabilities (start at color -> P(goal), P(turnover)):")
    analytic_lines = []
    for s in analytic:
        line = f"  {s}: {analytic[s]}"
        print(line)
        analytic_lines.append(line)
    # Save analytic absorption probabilities to file
    write_output("analytic_absorption.txt", "\n".join(analytic_lines))

    # 3. Monte Carlo Comparison
    print("\n=== MONTE CARLO COMPARISON ===")
    mc_no_clock = simulate_n_matches(n=400, turns=40, shot_clock_limit=None, base_seed=100)
    mc_with_clock = simulate_n_matches(n=400, turns=40, shot_clock_limit=5, base_seed=1000)
    print("Monte Carlo (no shot clock) summary (avg goals per match):", mc_no_clock)
    print("Monte Carlo (shot clock=5) summary (avg goals per match):", mc_with_clock)
    # Save Monte Carlo summaries to file
    write_output("monte_carlo_no_shot_clock.json", json.dumps(mc_no_clock, indent=2))
    write_output("monte_carlo_shot_clock_5.json", json.dumps(mc_with_clock, indent=2))

    # 4. Analytic vs Empirical for initial red possession
    print("\n=== Analytic vs Empirical for initial red possession ===")
    p_goal_red = analytic["red"]["P(goal)"]
    print(f"Analytic P(goal | start red) = {p_goal_red:.4f}")

    # Empirical: simulate many single possessions by forcing starting red and running until absorption
    def simulate_possession_once(rng):
        # returns 'goal' or 'turnover'
        current = "red"
        while True:
            face = rng.choice(DICE[current])
            if is_pass(face):
                current = pass_color(face)
            elif is_shot(face):
                green_face = rng.choice(DICE["green"])
                if green_face == "goal":
                    return "goal"
                elif is_pass(green_face):
                    current = pass_color(green_face)
                else:
                    return "turnover"
            elif current != "green" and "goal" in face.lower():
                return "goal"
            elif is_turnover(face):
                return "turnover"
            elif "throw again" in face.lower():
                # stay in same state
                continue
            elif current == "green":
                if face == "goal":
                    return "goal"
                elif is_pass(face):
                    current = pass_color(face)
                else:
                    return "turnover"
            else:
                current = "red"

    rng = random.Random(999)
    trials = 20000
    cnt_goal = 0
    for _ in range(trials):
        if simulate_possession_once(rng) == "goal":
            cnt_goal += 1
    emp_p = cnt_goal / trials
    print(f"Empirical P(goal | start red) approx = {emp_p:.4f} ({trials} trials)")

    # Save analytic and empirical comparison to file
    analytic_vs_empirical = {
        "analytic_P_goal_start_red": p_goal_red,
        "empirical_P_goal_start_red": emp_p,
        "trials": trials
    }
    write_output("analytic_vs_empirical.json", json.dumps(analytic_vs_empirical, indent=2))

    print("\n--- End of Task 1 enhanced script ---")
