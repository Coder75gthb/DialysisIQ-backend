import urllib.request
import urllib.parse
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def log_pass(name, details=""):
    print(f"  [PASS] {name} {f'-> {details}' if details else ''}")

def log_fail(name, err):
    print(f"  [FAIL] {name} -> {err}")

def run_tests():
    print("==================================================")
    print("Testing All DialysisIQ Backend Endpoints & ML Modules")
    print("==================================================")
    
    passed = 0
    failed = 0

    # 1. GET /
    try:
        res = urllib.request.urlopen(f"{BASE_URL}/")
        data = json.loads(res.read())
        assert data.get("message") == "DialysisIQ backend is running"
        log_pass("Root Healthcheck (GET /)", data.get("message"))
        passed += 1
    except Exception as e:
        log_fail("Root Healthcheck (GET /)", e)
        failed += 1

    # 2. GET /patients
    try:
        res = urllib.request.urlopen(f"{BASE_URL}/patients?query=100001")
        data = json.loads(res.read())
        patients = data.get("patients", [])
        assert len(patients) > 0
        log_pass("Search Patients (GET /patients?query=100001)", f"Found PID {patients[0]['pid']} ({patients[0].get('name')})")
        passed += 1
    except Exception as e:
        log_fail("Search Patients (GET /patients)", e)
        failed += 1

    # 3. GET /sessions
    try:
        res = urllib.request.urlopen(f"{BASE_URL}/sessions")
        data = json.loads(res.read())
        sessions = data.get("sessions", [])
        assert len(sessions) > 0
        log_pass("Get Sessions (GET /sessions)", f"Fetched {len(sessions)} recent sessions")
        passed += 1
    except Exception as e:
        log_fail("Get Sessions (GET /sessions)", e)
        failed += 1

    # 4. POST /sessions
    try:
        payload = {
            "pid": 100001,
            "pre_sbp": 128.0,
            "pre_dbp": 75.0,
            "weightstart": 74.5,
            "dryweight": 70.0,
            "duration_min": 240,
            "avg_conductivity": 14.0,
            "avg_dia_temp": 36.5
        }
        req = urllib.request.Request(
            f"{BASE_URL}/sessions",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        res = urllib.request.urlopen(req)
        data = json.loads(res.read())
        assert "session_id" in data
        log_pass("Create Session (POST /sessions)", f"Created session {data['session_id']}")
        passed += 1
    except Exception as e:
        log_fail("Create Session (POST /sessions)", e)
        failed += 1

    # 5. GET /module1/predict/{session_id}
    try:
        res = urllib.request.urlopen(f"{BASE_URL}/module1/predict/100001_latest")
        data = json.loads(res.read())
        assert "predicted_qb" in data
        log_pass("Module 1 Prediction (GET /module1/predict/100001_latest)", f"Predicted Qb: {data['predicted_qb']} mL/min")
        passed += 1
    except Exception as e:
        log_fail("Module 1 Prediction", e)
        failed += 1

    # 6. GET /module2/predict/{session_id}
    try:
        res = urllib.request.urlopen(f"{BASE_URL}/module2/predict/100001_latest")
        data = json.loads(res.read())
        assert "hypotension_probability" in data
        log_pass("Module 2 Prediction (GET /module2/predict/100001_latest)", f"Prob: {data['hypotension_probability']:.2f}, Tier: {data['hypotension_tier']}")
        passed += 1
    except Exception as e:
        log_fail("Module 2 Prediction", e)
        failed += 1

    # 7. POST /module3/predict
    try:
        m3_payload = {
            "session_id": "100001_latest",
            "pid": 100001,
            "event_time": "2026-08-27T02:00:00Z",
            "event": {}
        }
        req = urllib.request.Request(
            f"{BASE_URL}/module3/predict",
            data=json.dumps(m3_payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        res = urllib.request.urlopen(req)
        data = json.loads(res.read())
        assert "predicted_event" in data or "module" in data
        log_pass("Module 3 Event Classification (POST /module3/predict)", f"Predicted event: {data.get('predicted_event', 'Normal')}")
        passed += 1
    except Exception as e:
        log_fail("Module 3 Event Classification", e)
        failed += 1

    # 8. GET /module4/predict/{pid}
    try:
        res = urllib.request.urlopen(f"{BASE_URL}/module4/predict/100001")
        data = json.loads(res.read())
        assert "component_a" in data
        drift_flag = data["component_a"].get("drift_detected")
        log_pass("Module 4 Drift Detection (GET /module4/predict/100001)", f"Drift Detected: {drift_flag}")
        passed += 1
    except Exception as e:
        log_fail("Module 4 Drift Detection", e)
        failed += 1

    # 9. PATCH /sessions/{session_id}/qb-intervention
    try:
        qb_payload = {"qb_intervention": True}
        req = urllib.request.Request(
            f"{BASE_URL}/sessions/100001_latest/qb-intervention",
            data=json.dumps(qb_payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='PATCH'
        )
        res = urllib.request.urlopen(req)
        data = json.loads(res.read())
        assert data.get("qb_intervention") is True
        log_pass("Qb Intervention Annotation (PATCH /sessions/100001_latest/qb-intervention)", "Annotated qb_intervention: True")
        passed += 1
    except Exception as e:
        log_fail("Qb Intervention Annotation", e)
        failed += 1

    # 10. GET /module5/predict
    try:
        res = urllib.request.urlopen(f"{BASE_URL}/module5/predict")
        data = json.loads(res.read())
        summary = data.get("summary", {})
        assert summary.get("n_total") == 15
        log_pass("Module 5 Morning Briefing (GET /module5/predict)", f"Summary: Total {summary.get('n_total')}, High {summary.get('n_high')}, Med {summary.get('n_medium')}, Low {summary.get('n_low')}, Drift {summary.get('n_drift')}")
        passed += 1
    except Exception as e:
        log_fail("Module 5 Morning Briefing", e)
        failed += 1

    # 11. POST /module5/refresh
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/module5/refresh",
            data=b'{}',
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        res = urllib.request.urlopen(req)
        data = json.loads(res.read())
        assert "message" in data
        log_pass("Module 5 Refresh (POST /module5/refresh)", data["message"])
        passed += 1
    except Exception as e:
        log_fail("Module 5 Refresh", e)
        failed += 1

    print("==================================================")
    print(f"Results: {passed} PASSED, {failed} FAILED out of {passed + failed} tests")
    print("==================================================")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
