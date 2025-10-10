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


import matplotlib.pyplot as plt

# Modified plotting functions to save images to output_dir
def plot_match_events(events, score, output_dir=None, filename="match_events.png"):
    # Map event types to numeric positions
    event_map = {"Pass":1, "Shot":2, "Goal":3, "Turnover":4}
    colors = {"Player 1":"blue", "Player 2":"orange"}

    x = [e["turn"] for e in events]
    y = [event_map[e["event"]] for e in events]
    c = [colors[e["player"]] for e in events]

    plt.figure(figsize=(12,4))
    plt.scatter(x, y, c=c, s=100, marker="o", edgecolor="k")
    plt.yticks(list(event_map.values()), list(event_map.keys()))
    plt.xlabel("Turn (Minute)")
    plt.ylabel("Event")
    plt.title(f"Match Timeline (Final Score P1 {score['Player 1']} - P2 {score['Player 2']})")
    if output_dir is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    else:
        plt.show()

def plot_score_progression(events, output_dir=None, filename="score_progression.png"):
    p1_goals, p2_goals, turns = [], [], []
    g1=g2=0
    for e in events:
        if e["event"]=="Goal":
            if e["player"]=="Player 1":
                g1+=1
            else:
                g2+=1
        turns.append(e["turn"])
        p1_goals.append(g1)
        p2_goals.append(g2)

    plt.figure()
    plt.plot(turns, p1_goals, label="Player 1", color="blue")
    plt.plot(turns, p2_goals, label="Player 2", color="orange")
    plt.xlabel("Turn")
    plt.ylabel("Cumulative Goals")
    plt.title("Score Progression During Match")
    plt.legend()
    if output_dir is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    else:
        plt.show()

def plot_possession(events, output_dir=None, filename="possession_timeline.png"):
    colors = {"Player 1":"blue","Player 2":"orange"}
    bar_colors = [colors[e["player"]] for e in events]

    plt.figure(figsize=(12,2))
    plt.bar(range(len(events)), [1]*len(events), color=bar_colors, edgecolor="k")
    plt.xlabel("Turn")
    plt.yticks([])
    plt.title("Possession Timeline")
    if output_dir is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    else:
        plt.show()

# ---------------------------
# Part A: simulator (storytelling version)
# ---------------------------
def simulate_single_match(turns=30, rng_seed=None, shot_clock_limit=None, hail_success=0.35, verbose=True):
    """
    Simulate a match with storytelling prints (verbose=True).
    Returns stats dict: score, shots, possession_rolls, events.
    """
    rng = random.Random(rng_seed)
    score = {"Player 1": 0, "Player 2": 0}
    shots = {"Player 1": 0, "Player 2": 0}
    poss_rolls = {"Player 1": 0, "Player 2": 0}
    events = []  # store structured events

    current_player = "Player 1"
    current_die = "red"
    shot_clock = 0

    for t in range(1, turns+1):
        poss_rolls[current_player] += 1
        shot_clock += 1
        face = roll_from(current_die, rng)

        if verbose:
            print(f"[Turn {t}] {current_player} rolls {current_die.upper()} -> {face}")

        # --- Pass
        if is_pass(face):
            current_die = pass_color(face)
            events.append({"turn": t, "player": current_player, "event": "Pass"})
            if verbose:
                print(f"  PASS to {current_die.upper()}")

        # --- Shot
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
                events.append({"turn": t, "player": current_player, "event": "Goal"})
                if verbose:
                    print(f"    ⚽ GOAL for {current_player} (score {score})")
                current_player = "Player 2" if current_player == "Player 1" else "Player 1"
                current_die = "red"
            elif is_pass(green):
                current_die = pass_color(green)
                events.append({"turn": t, "player": current_player, "event": "Pass"})
                if verbose:
                    print(f"    SET PIECE -> keep possession, next die {current_die}")
            else:
                events.append({"turn": t, "player": current_player, "event": "Shot"})
                current_player = "Player 2" if current_player == "Player 1" else "Player 1"
                current_die = "red"
                if verbose:
                    print("    Miss -> possession changes")

        # --- Direct goal (rare)
        elif "goal" in face.lower():
            score[current_player] += 1
            events.append({"turn": t, "player": current_player, "event": "Goal"})
            if verbose:
                print(f"  ⚽ Direct GOAL for {current_player} (score {score})")
            current_player = "Player 2" if current_player == "Player 1" else "Player 1"
            current_die = "red"
            shot_clock = 0

        # --- Turnover
        elif is_turnover(face):
            events.append({"turn": t, "player": current_player, "event": "Turnover"})
            if verbose:
                print("  Turnover")
            current_player = "Player 2" if current_player == "Player 1" else "Player 1"
            current_die = "red"
            shot_clock = 0

        # --- Dribble
        elif "throw again" in face.lower():
            events.append({"turn": t, "player": current_player, "event": "Pass"})
            if verbose:
                print("  Dribble -> keep possession (re-roll next)")

        else:
            current_die = "red"
            events.append({"turn": t, "player": current_player, "event": "Pass"})

        # --- Shot clock enforcement
        if shot_clock_limit is not None and shot_clock >= shot_clock_limit:
            if verbose:
                print(f"  ⏱ Shot clock expired for {current_player}: Hail Mary!")
            shots[current_player] += 1
            shot_clock = 0
            green = roll_from("green", rng)
            if green == "goal" and rng.random() < hail_success:
                score[current_player] += 1
                events.append({"turn": t, "player": current_player, "event": "Goal"})
                if verbose:
                    print(f"    ⚽ HAIL MARY GOAL for {current_player}!")
            else:
                events.append({"turn": t, "player": current_player, "event": "Shot"})
                if verbose:
                    print(f"    Hail Mary missed ({green}). Possession changes.")
                current_player = "Player 2" if current_player == "Player 1" else "Player 1"
                current_die = "red"

    if verbose:
        print("\nFinal score:", score)

    return {"score": score, "shots": shots, "possession_rolls": poss_rolls, "events": events}

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

    events = ex["events"]
    score = ex["score"]

    # Save plots to output_dir as images
    plot_match_events(events, score, output_dir=output_dir, filename="match_events.png")
    plot_score_progression(events, output_dir=output_dir, filename="score_progression.png")
    plot_possession(events, output_dir=output_dir, filename="possession_timeline.png")
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
