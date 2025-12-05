import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"


def check_health():
    print("==> Checking /health")
    r = requests.get(f"{BASE_URL}/health")
    print("Status:", r.status_code)
    data = r.json()
    print("model_loaded:", data.get("model_loaded"))
    print("config:", data.get("config"))
    print()
    return r.status_code == 200 and data.get("model_loaded") is True


def check_predict():
    print("==> Checking /predict")
    payload = {"text": "Free entry in 2 a wkly comp to win FA Cup final tickets!"}
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Status:", r.status_code)
    print("Response:", r.json())
    print()
    return r.status_code == 200


def check_batch():
    print("==> Checking /batch")
    payload = {
        "texts": [
            "Your OTP code is 394021. Do not share it.",
            "Congratulations! You won a FREE vacation, click this link now!",
            "Hey, are we still meeting at 5 pm today?",
        ]
    }
    r = requests.post(f"{BASE_URL}/batch", json=payload)
    print("Status:", r.status_code)
    data = r.json()
    print("size:", data.get("size"))
    print("items sample:", data.get("items")[:2])
    print()
    return r.status_code == 200 and data.get("size") == len(payload["texts"])


def check_file_predict():
    print("==> Checking /file-predict")

    # نجهز ملف بسيط للتجربة
    test_dir = Path("data")
    test_dir.mkdir(exist_ok=True)
    csv_path = test_dir / "verify_messages.csv"

    if not csv_path.exists():
        csv_path.write_text(
            "text\n"
            "Free entry in 2 a wkly comp to win FA Cup final tickets!\n"
            "Your OTP code is 394021. Do not share this with anyone.\n"
            "Congratulations! You won a FREE vacation, click this link now!\n"
            "Hey, are we still meeting at 5 pm today?\n",
            encoding="utf-8",
        )

    with csv_path.open("rb") as f:
        files = {"file": (csv_path.name, f, "text/csv")}
        r = requests.post(f"{BASE_URL}/file-predict", files=files)

    print("Status:", r.status_code)
    out_name = f"prediction_{csv_path.stem}.csv"
    out_path = test_dir / out_name

    if r.status_code == 200:
        out_path.write_bytes(r.content)
        print(f"Saved -> {out_path}")
    else:
        print("Error body:", r.text)

    print()
    return r.status_code == 200 and out_path.exists()


def main():
    print("=== Verifying Spam Detector API ===\n")

    ok_health = check_health()
    ok_predict = check_predict()
    ok_batch = check_batch()
    ok_file = check_file_predict()

    all_ok = all([ok_health, ok_predict, ok_batch, ok_file])
    print("===================================")
    print("ALL CHECKS OK:", all_ok)
    print("===================================")


if __name__ == "__main__":
    main()
