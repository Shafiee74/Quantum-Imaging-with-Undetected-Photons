"""End-to-end check of the Django API (or Next rewrite proxy).

    python qiup-ui/scripts/smoke_test.py [http://127.0.0.1:8000]
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
CONFIG = {
    "profiles": ["A", "B", "C", "D"],
    "modes": {"ideal": True, "noisy": True},
    "object": {"source": "mnist", "mnist_index": 5},
    "size": 64,
    "seed": 15000,
    "thermal": {
        "A": {"T_pm0_NL1_C": 83.7, "T_pm0_NL2_C": 84.7, "dT_NL1_C": 0.2, "dT_NL2_C": 0.0},
        "B": {"T_pm0_NL_C": 84.3, "dT_NL_C": 0.1},
    },
    "noisy_setup": {
        "blur_model": "double_blur",
        "image_rotation_deg": 1.5,
        "slm_misreg_x_px": 1.0,
        "slm_misreg_y_px": -0.5,
        "enable_lens_rings": True,
        "enable_etalon_fringes": True,
    },
}


def main() -> int:
    with urllib.request.urlopen(f"{BASE}/api/defaults", timeout=300) as r:
        defaults = json.load(r)
    print("defaults ok:", sorted(defaults["profiles"]))

    request = urllib.request.Request(
        f"{BASE}/api/run", data=json.dumps(CONFIG).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.time()
    errors = results = 0
    with urllib.request.urlopen(request, timeout=900) as r:
        for line in r:
            event = json.loads(line)
            t = time.time() - started
            kind = event["type"]
            if kind == "result":
                results += 1
                thermal = ", ".join(f"{x['crystal']}={x['T_crystal_C']:.2f}C" for x in event["thermal"])
                print(f"{t:6.2f}s result {event['profile']} {event['mode']:5s} mean={event['image']['mean']:.4g} [{thermal}]")
            elif kind == "object":
                print(f"{t:6.2f}s object label={event['label']}")
            elif kind == "error":
                errors += 1
                print(f"{t:6.2f}s ERROR {event.get('profile')} {event.get('mode')}: {event['message']}")
            elif kind in ("start", "done"):
                print(f"{t:6.2f}s {kind} {event.get('output_dir')}")
    print(f"results={results} errors={errors}")
    return 0 if errors == 0 and results > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
