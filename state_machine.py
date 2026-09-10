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


STEP_TITLES = {
    "state_0_barrel_only": "1. Barrel Body",
    "state_1_backcap_on": "2. Rear Back Cap",
    "state_2_refill_inserted": "3. Ink Refill",
    "state_3_conecap_on": "4. Front Cone Cap",
    "state_4_topcap_on": "5. Blue Top Cap",
}


class AssemblyStateMachine:
    """
    Tracks progress through an ordered sequence of states and flags
    out-of-order or reverted assembly. Includes rolling-window temporal
    smoothing to filter momentary hand occlusions or motion blur in live video.
    Strictly latches errors when out-of-order steps occur.
    """

    def __init__(self, state_order=None, window_size=12, min_consensus=8):
        """
        state_order: ordered list of state names. Defaults to DEFAULT_PEN_STATES.
        window_size: number of recent frames to buffer for majority voting (~0.4s @ 30fps).
        min_consensus: minimum frame votes required to confirm state transition.
        """
        self.state_order = state_order or DEFAULT_PEN_STATES
        self.current_index = 0
        self.window_size = window_size
        self.min_consensus = min_consensus
        self.history = deque(maxlen=window_size)
        
        # Strict Error Latching
        self.error_active = False
        self.error_detail = ""
        self.skipped_step_index = None

    def reset(self):
        """Resets assembly tracker back to the initial step."""
        self.current_index = 0
        self.history.clear()
        self.error_active = False
        self.error_detail = ""
        self.skipped_step_index = None

    def get_consensus(self):
        """Returns the most frequent valid state in the rolling window and its vote count."""
        valid_votes = [s for s in self.history if s is not None and s in self.state_order]
        if not valid_votes:
            return None, 0
        counts = Counter(valid_votes)
        top_state, count = counts.most_common(1)[0]
        return top_state, count

    def get_step_title(self, index):
        if 0 <= index < len(self.state_order):
            s = self.state_order[index]
            return STEP_TITLES.get(s, s)
        return "Unknown Step"

    def update_smoothed(self, raw_state):
        """
        Pushes a new frame prediction into the rolling buffer, performs
        majority voting, and only triggers a state advance/error once
        consensus is reached across consecutive frames.
        """
        self.history.append(raw_state)
        consensus_state, votes = self.get_consensus()
        ratio = f"{votes}/{len(self.history)}"

        if consensus_state is None or votes < self.min_consensus:
            # If an error is already active, keep showing the latched error
            if self.error_active:
                return "error", self.error_detail, consensus_state, ratio
            return "holding", "stabilizing", consensus_state, ratio

        status, detail = self.update(consensus_state)
        return status, detail, consensus_state, ratio

    def current_state(self):
        return self.state_order[self.current_index]

    def update(self, matched_state):
        """
        Processes a consensus state prediction and enforces sequential assembly.
        Latches an ERROR when a step is skipped.
        """
        if matched_state is None:
            if self.error_active:
                return "error", self.error_detail
            return "holding", None

        if matched_state not in self.state_order:
            self.error_active = True
            self.error_detail = f"Unrecognized part: {matched_state}"
            return "error", self.error_detail

        matched_index = self.state_order.index(matched_state)

        # Case 1: Same as current step
        if matched_index == self.current_index:
            if self.error_active:
                return "error", self.error_detail
            return "holding", None

        # Case 2: Valid sequential advance to the EXACT next step
        elif matched_index == self.current_index + 1:
            self.current_index = matched_index
            self.error_active = False
            self.error_detail = ""
            self.skipped_step_index = None
            return "advanced", self.current_state()

        # Case 3: Reverted to an earlier step
        elif matched_index < self.current_index:
            self.error_active = True
            self.error_detail = (
                f"REVERTED to [{self.get_step_title(matched_index)}] "
                f"(was on [{self.get_step_title(self.current_index)}])"
            )
            return "error", self.error_detail

        # Case 4: Skipped ahead out-of-order!
        else:
            self.error_active = True
            self.skipped_step_index = self.current_index + 1
            expected = self.get_step_title(self.current_index + 1)
            detected = self.get_step_title(matched_index)
            self.error_detail = f"SKIPPED STEP! Missing [{expected}], but found [{detected}]"
            return "error", self.error_detail

    def get_steps_for_hud(self):
        """
        Returns the sequential inspection checklist:
        List of dicts with name, state, and status ('verified', 'current', 'skipped', 'pending').
        """
        steps = []
        for i, sname in enumerate(self.state_order):
            title = STEP_TITLES.get(sname, sname)
            if i < self.current_index:
                st = "verified"
            elif self.error_active and i == self.skipped_step_index:
                st = "skipped"
            elif i == self.current_index:
                st = "error_current" if self.error_active else "current"
            else:
                st = "pending"
            steps.append({"title": title, "state": sname, "status": st})
        return steps

    def verify_components(self, components):
        """
        Validates logical rules on physical components.
        """
        if not components.get("barrel", True):
            return False, "Missing pen barrel"
        if components.get("top_cap", False) and not components.get("cone_cap", False):
            return False, "Top cap placed without cone cap"
        if components.get("cone_cap", False) and not components.get("refill", False):
            return False, "Cone cap assembled without inserting refill"
        return True, "Components logically consistent"

    def is_complete(self):
        return self.current_index == len(self.state_order) - 1 and not self.error_active