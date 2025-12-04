# %%
# integration_pems_ems_sumo.py

import os
import glob
from datetime import datetime, date

import numpy as np
import pandas as pd
import time


try:
    from sklearn.neighbors import KDTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import traci
from traci.exceptions import TraCIException


# ============================================================
# 1. NEAREST-STATION MAPPING: EMS <-> PeMS METADATA
# ============================================================

def build_station_kdtree(meta_rl: pd.DataFrame):
    """
    Build a KD-tree (or return coordinates) for nearest-station lookup.
    meta_rl must have: 'ID', 'Latitude', 'Longitude'
    """
    coords = meta_rl[["Latitude", "Longitude"]].to_numpy(dtype=float)
    if SKLEARN_AVAILABLE:
        tree = KDTree(coords, metric="euclidean")
        return tree, coords
    else:
        # fallback: return None, we'll do brute-force
        return None, coords


def attach_nearest_station_to_ems(ems_rl: pd.DataFrame,
                                  meta_rl: pd.DataFrame,
                                  tree=None,
                                  station_coords=None):
    """
    For each EMS event, find nearest PeMS station by lat/lon.
    Adds columns: 'nearest_station_id', 'station_lat', 'station_lon', 'station_freeway'
    """
    ems = ems_rl.copy()

    if not {"lat", "lon"}.issubset(ems.columns):
        raise ValueError("ems_rl must have 'lat' and 'lon' columns")

    if not {"ID", "Latitude", "Longitude"}.issubset(meta_rl.columns):
        raise ValueError("meta_rl must have 'ID', 'Latitude', 'Longitude' columns")

    # EMS coords
    ems_coords = ems[["lat", "lon"]].to_numpy(dtype=float)

    station_ids = meta_rl["ID"].to_numpy()
    station_lats = meta_rl["Latitude"].to_numpy()
    station_lons = meta_rl["Longitude"].to_numpy()
    station_fwy = meta_rl["Freeway"].to_numpy() if "Freeway" in meta_rl.columns else None

    if tree is not None and station_coords is not None:
        # KDTree lookup (fast)
        dists, idx = tree.query(ems_coords, k=1)
        idx = idx.flatten()
    else:
        # brute-force: compute distance to every station
        # (OK for a few hundred stations)
        station_coords = np.column_stack([station_lats, station_lons])
        idx = []
        for lat, lon in ems_coords:
            d = np.sqrt((station_coords[:, 0] - lat) ** 2 +
                        (station_coords[:, 1] - lon) ** 2)
            idx.append(np.argmin(d))
        idx = np.array(idx, dtype=int)

    ems["nearest_station_id"] = station_ids[idx]
    ems["station_lat"] = station_lats[idx]
    ems["station_lon"] = station_lons[idx]
    if station_fwy is not None:
        ems["station_freeway"] = station_fwy[idx]

    return ems


# ============================================================
# 2. PEMS TRAFFIC LOOKUP: LOAD TRANSFORMED DAY FOR RL
# ============================================================

def load_transformed_pems_day(transformed_dir: str,
                              day: date) -> pd.DataFrame:
    """
    Load the RL-transformed PeMS day parquet from 4_transformed_dataset.
    Expects filenames like: d04_text_station_5min_YYYY_MM_DD_rl.parquet
    """
    pattern = os.path.join(
        transformed_dir,
        f"d04_text_station_5min_{day.year:04d}_{day.month:02d}_{day.day:02d}_rl.parquet",
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No transformed PeMS file for {day} at {pattern}")

    df = pd.read_parquet(matches[0])
    # ensure types
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for col in ["Total Flow", "Avg Speed", "Avg Occupancy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_station_traffic_time_series(pems_day_rl: pd.DataFrame,
                                    station_id,
                                    time_col="Timestamp"):
    """
    Extract traffic time series for a single station from RL-transformed day.
    Returns df sorted by time with columns at least:
      [Timestamp, Total Flow, Avg Speed, Avg Occupancy]
    """
    df = pems_day_rl.copy()
    if "Station" not in df.columns:
        raise ValueError("pems_day_rl must have 'Station' column")

    station_str = str(station_id)
    df = df[df["Station"].astype(str) == station_str].copy()

    if time_col in df.columns:
        df = df.sort_values(time_col)

    return df


def lookup_traffic_at_time(pems_station_ts: pd.DataFrame,
                           t: pd.Timestamp,
                           time_col="Timestamp"):
    """
    Given a station time series and a timestamp t,
    return the nearest traffic record (flow/speed/occupancy).
    """
    if pems_station_ts.empty:
        return None

    times = pems_station_ts[time_col].values
    # argmin on absolute time difference
    diffs = np.abs(times - np.datetime64(t))
    idx = diffs.argmin()
    row = pems_station_ts.iloc[idx]

    return {
        "Timestamp": row[time_col],
        "Total Flow": row.get("Total Flow", np.nan),
        "Avg Speed": row.get("Avg Speed", np.nan),
        "Avg Occupancy": row.get("Avg Occupancy", np.nan),
    }


# ============================================================
# 3. BUILD EV SCHEDULE FROM EMS EVENTS (PER DAY)
# ============================================================

def filter_ems_for_day(ems_rl: pd.DataFrame, day: date):
    """
    Filter EMS RL table to events occurring on a given calendar date.
    """
    df = ems_rl.copy()
    if "event_time" not in df.columns:
        raise ValueError("ems_rl must have 'event_time' column (datetime)")

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    mask = df["event_time"].dt.date == day
    return df[mask].sort_values("event_time")


def build_ev_schedule_seconds(ems_day: pd.DataFrame):
    """
    Given EMS events for one day (with 'event_time'),
    return an array of simulation seconds when EVs should spawn.
    """
    if ems_day.empty:
        return np.array([], dtype=int)

    times = pd.to_datetime(ems_day["event_time"], errors="coerce").sort_values()
    start_of_day = times.iloc[0].normalize()  # midnight
    sim_times = (times - start_of_day).dt.total_seconds().astype(int)
    return sim_times.values


# ============================================================
# 4. SUMO ENVIRONMENT WITH PEMS + EMS INTEGRATION (SKELETON)
# ============================================================

class SUTrafficEnv:
    """
    SUMO traffic environment integrating:
    - EMS events (EV spawn schedule)
    - PeMS traffic features (for state)
    """

    def __init__(
        self,
        sumo_cfg: str,
        ems_day: pd.DataFrame,
        pems_day_rl: pd.DataFrame,
        meta_rl: pd.DataFrame,
        tls_ids,
        station_edge_map=None,  # mapping: station_id -> list of SUMO edge IDs (optional)
        use_gui: bool = False,
        step_length: float = 1.0,
        sim_duration_s: int = 3600,
    ):
        """
        sumo_cfg: path to .sumocfg
        ems_day: EMS events for a single day (after nearest-station assignment and transform_ems_for_rl)
        pems_day_rl: RL-transformed PeMS data for that day
        meta_rl: station metadata (for mapping, maybe debugging)
        tls_ids: list of traffic light IDs to control
        station_edge_map: dict mapping station IDs (str/int) -> SUMO edge IDs (not strictly required)
        """
        self.sumo_cfg = sumo_cfg
        self.ems_day = ems_day.copy()
        self.pems_day_rl = pems_day_rl.copy()
        self.meta_rl = meta_rl.copy()
        self.tls_ids = tls_ids
        self.station_edge_map = station_edge_map or {}
        self.use_gui = use_gui
        self.step_length = step_length
        self.sim_duration_s = sim_duration_s

        self.current_step = 0

        # Precompute EV schedule
        self.ev_schedule = build_ev_schedule_seconds(self.ems_day)
        self.next_ev_idx = 0

        # Precompute per-station traffic time series (dict of DataFrame)
        self.station_traffic = self._build_station_traffic_dict(self.pems_day_rl)

    def _build_station_traffic_dict(self, pems_day_rl):
        """
        Build dict: station_id_str -> station time series DF.
        """
        d = {}
        if "Station" not in pems_day_rl.columns:
            return d

        for station_id, df_station in pems_day_rl.groupby("Station"):
            df_station = df_station.sort_values("Timestamp")
            d[str(station_id)] = df_station
        return d
    

    def _ensure_traci_closed(self):
        """Safely close any existing TraCI connection."""
        try:
            if traci.isLoaded():
                print("[SUTrafficEnv] TraCI is loaded, closing existing connection...")
                traci.close()
                time.sleep(0.1)
        except TraCIException as e:
            print(f"[SUTrafficEnv] TraCIException while closing: {e}")
        except Exception as e:
            print(f"[SUTrafficEnv] Unexpected error while closing TraCI: {e}")

    def _start_sumo(self):
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [sumo_binary, "-c", self.sumo_cfg, "--step-length", str(self.step_length)]
        print("[SUTrafficEnv] Launching:", " ".join(cmd))
        traci.start(cmd, label="default")

    def reset(self, **kwargs):
        """Reset SUMO simulation and environment state."""
        self._ensure_traci_closed()
        self._start_sumo()
        self.current_step = 0
        self.next_ev_idx = 0
        obs = self._get_observation()
        return obs

    def _spawn_emergency_vehicle(self):
        """
        Spawn one emergency vehicle.
        This uses a simple placeholder route 'ev_route_0' and type 'emergency'.
        You must define those in your SUMO route/additional files.
        """
        def _spawn_emergency_vehicle(self):
            veh_id = f"EV_{self.current_step}"

            # TODO: replace these with REAL IDs from routes.rou.xml
            EV_ROUTE_ID = "route0"         
            EV_TYPE_ID = "car"             

            traci.vehicle.add(veh_id, routeID="ev_route_0", typeID="emergency")
            traci.vehicle.setColor(veh_id, (255, 0, 0, 255))

    def _maybe_spawn_ev(self):
        while (
            self.next_ev_idx < len(self.ev_schedule)
            and self.ev_schedule[self.next_ev_idx] <= self.current_step
        ):
            self._spawn_emergency_vehicle()
            self.next_ev_idx += 1

    def _get_time_of_day_features(self):
        tod = (self.current_step % 86400) / 86400.0  # [0,1)
        return tod

    def _get_next_ev_feature(self):
        if self.next_ev_idx < len(self.ev_schedule):
            dt_next = self.ev_schedule[self.next_ev_idx] - self.current_step
        else:
            dt_next = 99999.0
        return dt_next / 3600.0  # hours until next EV

    def _get_pems_traffic_features(self):
        """
        Example: average traffic features across all stations for this time step.
        You can make this more specific (e.g., only stations associated with EMS events).
        """
        if not self.station_traffic:
            return (0.0, 0.0, 0.0)

        # We approximate real time in day as start_of_day + current_step seconds.
        # For simplicity, assume pems_day_rl['Timestamp'] already covers the whole day
        any_station_df = next(iter(self.station_traffic.values()))
        start_ts = any_station_df["Timestamp"].min().normalize()
        current_ts = start_ts + pd.to_timedelta(self.current_step, unit="s")

        flows = []
        speeds = []
        occs = []

        for st_id, df_station in self.station_traffic.items():
            traffic = lookup_traffic_at_time(df_station, current_ts)
            if traffic is None:
                continue
            flows.append(traffic.get("Total Flow", np.nan))
            speeds.append(traffic.get("Avg Speed", np.nan))
            occs.append(traffic.get("Avg Occupancy", np.nan))

        def safe_mean(arr):
            arr = np.array(arr, dtype=float)
            arr = arr[~np.isnan(arr)]
            return float(arr.mean()) if len(arr) > 0 else 0.0

        return safe_mean(flows), safe_mean(speeds), safe_mean(occs)

    def _get_observation(self):
        """
        Example observation:
        [time_of_day, hours_to_next_EV, avg_flow, avg_speed, avg_occupancy, phases...]
        """
        tod = self._get_time_of_day_features()
        h_to_next_ev = self._get_next_ev_feature()
        avg_flow, avg_speed, avg_occ = self._get_pems_traffic_features()

        phases = []
        for tls_id in self.tls_ids:
            phase = traci.trafficlight.getPhase(tls_id)
            phases.append(float(phase))

        obs = np.array(
            [tod, h_to_next_ev, avg_flow, avg_speed, avg_occ, *phases], dtype=float
        )
        return obs

    def _compute_reward(self):
        """
        Reward: minimize waiting time, with higher weight on EV waiting.
        """
        total_wait = 0.0
        ev_wait = 0.0

        for vid in traci.vehicle.getIDList():
            w = traci.vehicle.getWaitingTime(vid)
            total_wait += w
            if vid.startswith("EV_"):
                ev_wait += w

        # weight EV waiting time more
        reward = - (total_wait + 5.0 * ev_wait)
        return reward
    def _get_num_phases(self, tls_id):
        """
        Get number of signal phases for a given TLS, compatible with older TraCI APIs.
        """
        # If your TraCI has getPhaseNumber, use it
        if hasattr(traci.trafficlight, "getPhaseNumber"):
            return traci.trafficlight.getPhaseNumber(tls_id)

        # Fallback: infer from the signal program definition
        # getCompleteRedYellowGreenDefinition returns a list of Logic objects
        logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
        if not logics:
            raise RuntimeError(f"No program logic found for TLS '{tls_id}'")
        return len(logics[0].phases)

    def step(self, action):
        """
        action: either scalar (for single TL) or list/array for multiple tls_ids
        """
        # apply action to TLS
        if isinstance(action, (list, np.ndarray)):
            for a, tls_id in zip(action, self.tls_ids):
                n_phases = traci.trafficlight.getPhaseNumber(tls_id)
                traci.trafficlight.setPhase(tls_id, int(a) % n_phases)
        else:
            # one action for all (coarse)
            for tls_id in self.tls_ids:
                n_phases = self._get_num_phases(tls_id)
                traci.trafficlight.setPhase(tls_id, int(action) % n_phases)

        # spawn EVs if scheduling says so
        self._maybe_spawn_ev()

        # advance SUMO
        traci.simulationStep()
        self.current_step += self.step_length

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self.current_step >= self.sim_duration_s
        info = {}
        return obs, reward, done, info

    def close(self):
        if traci.isLoaded():
            traci.close()



