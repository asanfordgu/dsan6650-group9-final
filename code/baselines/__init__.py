import numpy as np
import traci


class FixedTimeController:
    """
    Baseline 1: fixed-time traffic signal.
    Ignores EMS, just cycles phases every K simulation steps.
    """

    def __init__(self, tls_ids, phase_duration_steps=20, max_phases=4):
        self.tls_ids = tls_ids
        self.phase_duration_steps = phase_duration_steps
        self.max_phases = max_phases
        self.current_phase_idx = {tls_id: 0 for tls_id in tls_ids}
        self.step_counter = 0

    def select_action(self, obs=None):
        """
        Returns action compatible with GymTrafficEnv:
        - scalar if one TLS
        - list of actions if multiple
        """
        # advance phase every phase_duration_steps
        if self.step_counter % self.phase_duration_steps == 0 and self.step_counter > 0:
            for tls_id in self.tls_ids:
                self.current_phase_idx[tls_id] = (
                    self.current_phase_idx[tls_id] + 1
                ) % self.max_phases

        self.step_counter += 1

        if len(self.tls_ids) == 1:
            return self.current_phase_idx[self.tls_ids[0]]
        else:
            return [self.current_phase_idx[tls_id] for tls_id in self.tls_ids]


class GreedyEVPreemptionController:
    """
    Baseline 2: simple emergency preemption.
    If any EV is detected on an incoming edge for this TLS, switch/hold green toward that EV.
    Otherwise behaves like a fixed-time controller.
    """

    def __init__(
        self,
        tls_ids,
        ev_prefix="EV_",
        phase_duration_steps=20,
        max_phases=4,
        tls_phase_map=None,
    ):
        """
        tls_phase_map: optional dict mapping
            tls_id -> {phase_index: [incoming_edge_ids_for_that_phase]}
        If not provided, this example just uses phase index 0 as "EV direction".
        """
        self.tls_ids = tls_ids
        self.ev_prefix = ev_prefix
        self.phase_duration_steps = phase_duration_steps
        self.max_phases = max_phases
        self.tls_phase_map = tls_phase_map or {}

        self.current_phase_idx = {tls_id: 0 for tls_id in tls_ids}
        self.step_counter = 0

    def _ev_present_for_phase(self, tls_id, phase_idx):
        """
        Check if an EV exists on any incoming edge mapped to this phase.
        """
        phase_map = self.tls_phase_map.get(tls_id, {})
        incoming_edges = phase_map.get(phase_idx, [])

        if not incoming_edges:
            return False

        for vid in traci.vehicle.getIDList():
            if not vid.startswith(self.ev_prefix):
                continue
            # get the edge where vehicle is now
            edge_id = traci.vehicle.getRoadID(vid)
            if edge_id in incoming_edges:
                return True

        return False

    def select_action(self, obs=None):
        """
        Returns action compatible with GymTrafficEnv.
        """
        actions = []

        for tls_id in self.tls_ids:
            # 1) Check if any phase has an EV present; if so, preempt to that phase
            preempt_phase = None
            phase_map = self.tls_phase_map.get(tls_id, {})

            if phase_map:
                for phase_idx in phase_map.keys():
                    if self._ev_present_for_phase(tls_id, phase_idx):
                        preempt_phase = phase_idx
                        break

            if preempt_phase is not None:
                self.current_phase_idx[tls_id] = preempt_phase
            else:
                # 2) No EV: fixed-time cycling
                if (
                    self.step_counter % self.phase_duration_steps == 0
                    and self.step_counter > 0
                ):
                    self.current_phase_idx[tls_id] = (
                        self.current_phase_idx[tls_id] + 1
                    ) % self.max_phases

            actions.append(self.current_phase_idx[tls_id])

        self.step_counter += 1

        if len(self.tls_ids) == 1:
            return actions[0]
        return actions
