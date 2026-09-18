import json
import os
import time

import db_persistent_compat_app as dbapp
import compat_app as compat


def snap(label: str):
    started = time.perf_counter()
    result = compat.selftest_compat()
    result = dict(result)
    result["probe"] = label
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    print("DEEP_DB_PROBE=" + json.dumps(result, default=str, separators=(",", ":")), flush=True)
    return result


def main():
    first = snap("FIRST")
    deadline = time.time() + float(os.getenv("DEEP_PROBE_TRAIN_WAIT", "120"))
    while time.time() < deadline:
        state = dbapp.core.local_training_status("XAUUSD", "M3", 10)
        if state.get("state") in {"READY", "ERROR"}:
            print("DEEP_DB_TRAIN_STATE=" + json.dumps(state, default=str, separators=(",", ":")), flush=True)
            break
        time.sleep(2)
    second = snap("SECOND")
    art = dbapp.artifact_status()
    print("DEEP_DB_ARTIFACT_STATUS=" + json.dumps(art, default=str, separators=(",", ":")), flush=True)
    ok = bool(second.get("ok")) and second.get("patchtst_status") == "READY" and second.get("tft_status") == "READY"
    print("DEEP_DB_PERSISTENCE_RESULT=" + json.dumps({"ok": ok, "first": first.get("ensemble_weights"), "second": second.get("ensemble_weights"), "artifact_count": art.get("count")}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
