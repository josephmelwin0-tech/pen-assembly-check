"""
Shared logic: comparing a live embedding against your reference states,
and tracking assembly progress / flagging errors.

You don't run this file directly — validate.py and live_demo.py both import it.
"""

import numpy as np


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def match_state(live_embedding, state_embeddings, threshold=0.80):
    """
    Compares one live embedding against every state's reference embeddings.

    state_embeddings: { "state_name": [vec1, vec2, ...], ... }
    Returns (best_state_name, best_score). best_state_name is None if the
    best score didn't clear the threshold (probably mid-motion / hand blocking view).
    """
    best_state = None
    best_score = -1.0

    for state_name, vectors in state_embeddings.items():
        if not vectors:
            continue
        score = max(cosine_sim(live_embedding, v) for v in vectors)
        if score > best_score:
            best_state = state_name
            best_score = score

    if best_score < threshold:
        return None, best_score
    return best_state, best_score


DEFAULT_PEN_STATES = [
    "state_0_barrel_only",
    "state_1_backcap_on",
    "state_2_refill_inserted",
    "state_3_conecap_on",
    "state_4_topcap_on",
]


from collections import deque, Counter


class AssemblyStateMachine:
    """
    Tracks progress through an ordered sequence of states and flags
    out-of-order or reverted assembly. Includes rolling-window temporal
    smoothing to filter momentary hand occlusions or motion blur in live video.
    """

    def __init__(self, state_order=None, window_size=7, min_consensus=4):
        """
        state_order: ordered list of state names. Defaults to DEFAULT_PEN_STATES.
        window_size: number of recent frames to buffer for majority voting.
        min_consensus: minimum frame votes required to confirm state transition.
        """
        self.state_order = state_order or DEFAULT_PEN_STATES
        self.current_index = 0
        self.window_size = window_size
        self.min_consensus = min_consensus
        self.history = deque(maxlen=window_size)

    def reset(self):
        """Resets assembly tracker back to the initial step."""
        self.current_index = 0
        self.history.clear()

    def get_consensus(self):
        """Returns the most frequent valid state in the rolling window and its vote count."""
        valid_votes = [s for s in self.history if s is not None and s in self.state_order]
        if not valid_votes:
            return None, 0
        counts = Counter(valid_votes)
        top_state, count = counts.most_common(1)[0]
        return top_state, count

    def update_smoothed(self, raw_state):
        """
        Pushes a new frame prediction into the rolling buffer, performs
        majority voting, and only triggers a state advance/error once
        consensus is reached across consecutive frames.

        Returns (status, detail, consensus_state, votes_ratio)
        """
        self.history.append(raw_state)
        consensus_state, votes = self.get_consensus()
        ratio = f"{votes}/{len(self.history)}"

        if consensus_state is None or votes < self.min_consensus:
            # Stabilizing / waiting for consensus
            return "holding", "stabilizing", consensus_state, ratio

        status, detail = self.update(consensus_state)
        return status, detail, consensus_state, ratio

    def current_state(self):
        return self.state_order[self.current_index]

    def update(self, matched_state):
        """
        Call this every time you get a new matched_state (or None) from match_state().
        Returns (status, detail):
            status: "holding" | "advanced" | "error"
            detail: extra info string, or the new state name on "advanced"
        """
        if matched_state is None:
            return "holding", None  # no confident match — treat as mid-motion, not an error

        if matched_state not in self.state_order:
            return "error", f"unrecognized state name: {matched_state}"

        matched_index = self.state_order.index(matched_state)

        if matched_index == self.current_index:
            return "holding", None
        elif matched_index == self.current_index + 1:
            self.current_index = matched_index
            return "advanced", self.current_state()
        elif matched_index < self.current_index:
            return "error", f"reverted to '{matched_state}' (was on '{self.current_state()}')"
        else:
            return "error", (
                f"skipped ahead to '{matched_state}', "
                f"expected the step after '{self.current_state()}'"
            )

    def verify_components(self, components):
        """
        Validates logical rules on physical components:
            components: {"barrel": bool, "back_cap": bool, "refill": bool, "cone_cap": bool, "top_cap": bool}
        Returns (is_valid, message)
        """
        if not components.get("barrel", True):
            return False, "Missing pen barrel"
        if components.get("top_cap", False) and not components.get("cone_cap", False):
            return False, "Top cap placed without cone cap"
        if components.get("cone_cap", False) and not components.get("refill", False):
            return False, "Cone cap assembled without inserting refill"
        return True, "Components logically consistent"

    def is_complete(self):
        return self.current_index == len(self.state_order) - 1