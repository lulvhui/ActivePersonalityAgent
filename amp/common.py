import hashlib
import json
import math
import re
from pathlib import Path

TRAITS = dict(O="Openness", C="Conscientiousness", E="Extraversion", A="Agreeableness", N="Neuroticism")
LEVELS = ["very_low", "low", "medium", "high", "very_high"]
# Observation modes: audio_video combines the audio and visual modalities.
MODALITIES = ["text", "audio", "video", "audio_video"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def normalize_id(value):
    value = str(value)
    return str(int(value)) if value.isdigit() else value


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def probabilities(values):
    values = [float(x) for x in values]
    if len(values) != 5 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Expected five finite nonnegative probabilities")
    if abs(sum(values) - 1) > 1e-3:
        raise ValueError("Probability sum must be one")
    return [x / sum(values) for x in values]


def standardize(values):
    if not values:
        return []
    mean = sum(values) / len(values)
    std = (sum((x - mean)**2 for x in values) / len(values))**0.5
    return [(x - mean) / (std + 1e-8) for x in values]


def parse_json(text):
    return json.loads(text)


def time_pair(value):
    """Numeric seconds, including negative local offsets; bounds are caller-owned."""
    if not isinstance(value,(list,tuple)) or any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in value):
        raise ValueError('Time must be an array of two numeric seconds')
    try:start,end=map(float,value)
    except (ValueError,TypeError):raise ValueError('Time must contain two numeric seconds') from None
    if not all(math.isfinite(v) for v in (start,end)) or end<=start:raise ValueError('Time must be an increasing finite interval')
    return [start,end]
