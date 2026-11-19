import io
import sys

import requests

BASE_URL = "https://uda-census.fastapicloud.dev/"


CSV_TEXT = (
    "age, workclass, fnlgt, education, education-num, marital-status, "
    "occupation, relationship, race, sex, capital-gain, capital-loss, "
    "hours-per-week, native-country, salary\n"
    "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, "
    "Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n"
    "50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, "
    "Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, >50K\n"
)


def main() -> int:
    url = f"{BASE_URL}/predict"
    try:
        response = requests.post(
            url,
            files={
                "file": ("data.csv", io.BytesIO(CSV_TEXT.encode()), "text/csv")
            },
            timeout=60,  # free-tier hosts can take a while to wake up
        )
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return 1

    print(f"Status code: {response.status_code}")

    try:
        body = response.json()
    except ValueError:
        print(f"Non-JSON response: {response.text}")
        return 1

    if response.ok:
        print(f"Model inference result: {body['predictions']}")
        return 0

    print(f"Error response: {body}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
