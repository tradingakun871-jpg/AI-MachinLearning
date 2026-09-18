import json

from compat_app import selftest_compat


if __name__ == "__main__":
    result = selftest_compat()
    print("DEEP_STARTUP_PROBE=" + json.dumps(result, default=str, separators=(",", ":")), flush=True)
