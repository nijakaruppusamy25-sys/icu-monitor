"""
Multi-Clinician HR & RR Alert, Care Team & Patient Detail Approval Manager
- Zero fake pre-populated staff names.
- Individual login for Doctors, Nurses, and Patients.
- Multiple Doctors and Nurses can join a patient's care team and see co-treating staff.
- Real HR & RR threshold alerts are broadcast to all Doctors and Nurses treating the patient.
- Patient detail updates display who made the change and allow co-treating staff to Accept or Deny.
"""

import time
import uuid
import threading
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Callable

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VitalsAlertManager")


class AlertState(str, Enum):
    TRIGGERED = "TRIGGERED"
    ESCALATED = "ESCALATED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    CARE_IN_PROGRESS = "CARE_IN_PROGRESS"
    RESOLVED = "RESOLVED"


class VitalsAlert:
    def __init__(
        self,
        alert_id: str,
        patient_id: str,
        patient_name: str,
        room_bed: str,
        notified_team: List[Dict[str, str]],
        timeout_seconds: int = 30,
        vitals_details: Optional[Dict[str, Any]] = None,
    ):
        self.alert_id = alert_id
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.room_bed = room_bed
        self.created_at = time.time()
        self.timeout_seconds = timeout_seconds
        self.state = AlertState.TRIGGERED
        self.vitals_details = vitals_details or {}
        self.notified_team = notified_team
        self.responder_name: Optional[str] = None
        self.responder_role: Optional[str] = None
        self.care_by_name: Optional[str] = None
        self.resolver_name: Optional[str] = None

    def remaining_seconds(self) -> int:
        if self.state != AlertState.TRIGGERED:
            return 0
        elapsed = time.time() - self.created_at
        return max(0, int(round(self.timeout_seconds - elapsed)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "room_bed": self.room_bed,
            "remaining_seconds": self.remaining_seconds(),
            "timeout_seconds": self.timeout_seconds,
            "state": self.state.value,
            "vitals_details": self.vitals_details,
            "notified_team": self.notified_team,
            "responder_name": self.responder_name,
            "responder_role": self.responder_role,
            "care_by_name": self.care_by_name,
            "resolver_name": self.resolver_name,
        }


class VitalsAlertManager:
    def __init__(self, timeout_seconds: int = 30):
        self.response_timeout_seconds = timeout_seconds
        self.alert_counter = 1
        self.update_counter = 1

        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, str] = {}

        # Patient details start blank/neutral until entered & approved by clinicians
        self.patient_info: Dict[str, Any] = {
            "patient_id": "Not Entered",
            "name": "Not Entered",
            "age_gender": "Not Entered",
            "room_bed": "Not Entered",
            "last_updated_by": "None",
        }

        # Only real logged-in Doctors & Nurses who join this patient appear here
        self.patient_care_team: Dict[str, Dict[str, Any]] = {}
        self.pending_patient_update: Optional[Dict[str, Any]] = None
        self.active_alerts: Dict[str, VitalsAlert] = {}

        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        self._last_auto_alert_time: Dict[str, float] = {}

        self._timer_thread = threading.Thread(target=self._escalation_loop, daemon=True)
        self._timer_thread.start()

    def register_listener(self, callback: Callable[[Dict[str, Any]], None]):
        self._listeners.append(callback)

    def _broadcast(self, event_type: str):
        payload = {
            "event": event_type,
            "data": self.get_dashboard_state(),
        }
        for cb in self._listeners:
            try:
                cb(payload)
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # 1. AUTHENTICATION (DOCTOR / NURSE / PATIENT)
    # -----------------------------------------------------------------------
    def login_or_register(self, user_id: str, name: str, role: str, password: str) -> Dict[str, Any]:
        uid = user_id.strip().lower()
        role_clean = role.strip().capitalize()
        if role_clean not in ("Doctor", "Nurse", "Patient"):
            raise ValueError("Role must be Doctor, Nurse, or Patient")

        with self._lock:
            if uid in self.users:
                user = self.users[uid]
                if user["password"] != password:
                    raise ValueError("Incorrect password for this User ID")
                user["name"] = name.strip() or user["name"]
                user["role"] = role_clean
            else:
                user = {
                    "user_id": uid,
                    "name": name.strip(),
                    "role": role_clean,
                    "password": password,
                }
                self.users[uid] = user

            token = str(uuid.uuid4())
            self.sessions[token] = uid

            if uid in self.patient_care_team:
                self.patient_care_team[uid]["name"] = user["name"]
                self.patient_care_team[uid]["role"] = user["role"]

        self._broadcast("USER_LOGGED_IN")
        return {
            "token": token,
            "user": {"user_id": user["user_id"], "name": user["name"], "role": user["role"]},
        }

    def get_user_by_token(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        uid = self.sessions.get(token)
        return self.users.get(uid) if uid else None

    # -----------------------------------------------------------------------
    # 2. CARE TEAM SELF-ASSIGNMENT (MULTIPLE DOCTORS & NURSES PER PATIENT)
    # -----------------------------------------------------------------------
    def join_patient_care_team(self, user: Dict[str, Any]) -> Dict[str, Any]:
        if user["role"] not in ("Doctor", "Nurse"):
            raise ValueError("Only Doctors and Nurses can add themselves to a patient's care team.")

        uid = user["user_id"]
        with self._lock:
            member = {
                "user_id": uid,
                "name": user["name"],
                "role": user["role"],
                "joined_at": time.strftime("%H:%M:%S"),
                "status": "Monitoring Patient",
            }
            self.patient_care_team[uid] = member

            for alert in self.active_alerts.values():
                if not any(m["user_id"] == uid for m in alert.notified_team):
                    alert.notified_team.append({
                        "user_id": uid,
                        "name": user["name"],
                        "role": user["role"],
                    })

        self._broadcast("CARE_TEAM_UPDATED")
        return member

    def leave_patient_care_team(self, user: Dict[str, Any]):
        uid = user["user_id"]
        with self._lock:
            self.patient_care_team.pop(uid, None)
        self._broadcast("CARE_TEAM_UPDATED")

    # -----------------------------------------------------------------------
    # 3. PATIENT DETAILS UPDATE WITH ACCEPT / DENY BY CARE TEAM
    # -----------------------------------------------------------------------
    def propose_patient_update(self, user: Dict[str, Any], new_details: Dict[str, str]) -> Dict[str, Any]:
        with self._lock:
            uid = user["user_id"]
            if user["role"] in ("Doctor", "Nurse") and uid not in self.patient_care_team:
                self.patient_care_team[uid] = {
                    "user_id": uid,
                    "name": user["name"],
                    "role": user["role"],
                    "joined_at": time.strftime("%H:%M:%S"),
                    "status": "Monitoring Patient",
                }

            proposal = {
                "proposal_id": f"UPD-{self.update_counter:03d}",
                "proposed_by_id": uid,
                "proposed_by_name": user["name"],
                "proposed_by_role": user["role"],
                "proposed_at": time.strftime("%H:%M:%S"),
                "proposed_info": {
                    "patient_id": new_details.get("patient_id", "").strip() or "Not Entered",
                    "name": new_details.get("name", "").strip() or "Not Entered",
                    "age_gender": new_details.get("age_gender", "").strip() or "Not Entered",
                    "room_bed": new_details.get("room_bed", "").strip() or "Not Entered",
                },
            }
            self.pending_patient_update = proposal

        self._broadcast("PATIENT_UPDATE_PROPOSED")
        return proposal

    def review_patient_update(self, user: Dict[str, Any], decision: str) -> Dict[str, Any]:
        decision_upper = decision.strip().upper()
        if decision_upper not in ("ACCEPT", "DENY"):
            raise ValueError("Decision must be ACCEPT or DENY")

        with self._lock:
            if not self.pending_patient_update:
                raise ValueError("No pending update found")

            prop = self.pending_patient_update
            if decision_upper == "ACCEPT":
                self.patient_info.update(prop["proposed_info"])
                self.patient_info["last_updated_by"] = (
                    f"Updated by {prop['proposed_by_name']} ({prop['proposed_by_role']}) • "
                    f"Accepted by {user['name']} ({user['role']})"
                )
            self.pending_patient_update = None

        self._broadcast("PATIENT_UPDATE_REVIEWED")
        return prop

    # -----------------------------------------------------------------------
    # 4. REAL HR & RR THRESHOLD ALERTS SENT TO ALL ASSIGNED STAFF
    # -----------------------------------------------------------------------
    def _escalation_loop(self):
        while True:
            time.sleep(1.0)
            changed = False
            with self._lock:
                for alert in self.active_alerts.values():
                    if alert.state == AlertState.TRIGGERED:
                        changed = True
                        if alert.remaining_seconds() <= 0:
                            alert.state = AlertState.ESCALATED
            if changed:
                self._broadcast("TIMER_TICK")

    def evaluate_live_vitals(
        self,
        hr: Optional[float],
        rr: Optional[float],
        hr_status: str,
        rr_status: str,
    ):
        now = time.time()
        anomalies = []

        if hr is not None:
            if hr > 115.0:
                anomalies.append(("TACHYCARDIA", f"High Heart Rate: {hr:.1f} BPM (> 115 BPM)"))
            elif 0 < hr < 50.0:
                anomalies.append(("BRADYCARDIA", f"Low Heart Rate: {hr:.1f} BPM (< 50 BPM)"))

        if rr is not None:
            if rr == 0.0 or "APNEA" in rr_status:
                anomalies.append(("APNEA", "Apnea / Breath-Hold Detected (0.0 BrPM)"))
            elif rr > 22.0:
                anomalies.append(("TACHYPNEA", f"High Respiratory Rate: {rr:.1f} BrPM (> 22 BrPM)"))
            elif rr < 9.0:
                anomalies.append(("BRADYPNEA", f"Low Respiratory Rate: {rr:.1f} BrPM (< 9 BrPM)"))

        for anomaly_type, desc in anomalies:
            already_active = any(
                a.vitals_details.get("anomaly_type") == anomaly_type
                for a in self.active_alerts.values()
            )
            last_t = self._last_auto_alert_time.get(anomaly_type, 0.0)
            if not already_active and (now - last_t) > 45.0:
                self._last_auto_alert_time[anomaly_type] = now
                with self._lock:
                    alert_id = f"ALT-{self.alert_counter:03d}"
                    self.alert_counter += 1
                    notified = [
                        {"user_id": m["user_id"], "name": m["name"], "role": m["role"]}
                        for m in self.patient_care_team.values()
                    ]
                    self.active_alerts[alert_id] = VitalsAlert(
                        alert_id=alert_id,
                        patient_id=self.patient_info["patient_id"],
                        patient_name=self.patient_info["name"],
                        room_bed=self.patient_info["room_bed"],
                        notified_team=notified,
                        timeout_seconds=self.response_timeout_seconds,
                        vitals_details={
                            "anomaly_type": anomaly_type,
                            "description": desc,
                            "hr_bpm": round(hr, 1) if hr is not None else None,
                            "rr_brpm": round(rr, 1) if rr is not None else None,
                        },
                    )
                self._broadcast("ALERT_CREATED")

    def respond(self, alert_id: str, user: Dict[str, Any]) -> Optional[VitalsAlert]:
        with self._lock:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return None
            alert.state = AlertState.ACKNOWLEDGED
            alert.responder_name = user["name"]
            alert.responder_role = user["role"]
            if user["user_id"] in self.patient_care_team:
                self.patient_care_team[user["user_id"]]["status"] = f"Responding to {alert_id}"
        self._broadcast("ALERT_ACKNOWLEDGED")
        return alert

    def start_care(self, alert_id: str, user: Dict[str, Any]) -> Optional[VitalsAlert]:
        with self._lock:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return None
            alert.state = AlertState.CARE_IN_PROGRESS
            alert.care_by_name = f"{user['name']} ({user['role']})"
            if user["user_id"] in self.patient_care_team:
                self.patient_care_team[user["user_id"]]["status"] = f"Treating ({alert_id})"
        self._broadcast("CARE_IN_PROGRESS")
        return alert

    def resolve(self, alert_id: str, user: Dict[str, Any]) -> Optional[VitalsAlert]:
        with self._lock:
            alert = self.active_alerts.pop(alert_id, None)
            if not alert:
                return None
            for m in self.patient_care_team.values():
                if alert_id in m.get("status", ""):
                    m["status"] = "Monitoring Patient"
        self._broadcast("ALERT_RESOLVED")
        return alert

    def get_dashboard_state(self) -> Dict[str, Any]:
        return {
            "patient": self.patient_info,
            "pending_patient_update": self.pending_patient_update,
            "care_team": list(self.patient_care_team.values()),
            "active_alerts": [a.to_dict() for a in self.active_alerts.values()],
        }


_singleton_manager: Optional[VitalsAlertManager] = None


def get_alert_manager(timeout_seconds: int = 30) -> VitalsAlertManager:
    global _singleton_manager
    if _singleton_manager is None:
        _singleton_manager = VitalsAlertManager(timeout_seconds=timeout_seconds)
    return _singleton_manager