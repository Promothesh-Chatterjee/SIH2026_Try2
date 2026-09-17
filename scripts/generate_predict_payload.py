import json
from pathlib import Path

# The scheduler requires 36 bands × 10 features
OBSERVATION_DIM = 36 * 10

payload = {
    "obs": [0.0] * OBSERVATION_DIM
}

# Save the JSON file in the same scripts directory
output_path = Path(__file__).parent / "predict_test.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2)

print(f"Payload saved to: {output_path}")
print(f"Number of observation values: {len(payload['obs'])}")