"""A synthetic trajectory fixture: never reads real people or calls hosted APIs."""
from .agent import ActiveAgent
from .backend import OfflineBackend
from .scoring import OfflineTraitScorer


class SyntheticMedia:
    def coarse(self, row, fps, chunk_seconds):
        segments=row['segments'] or [dict(start=i*10,end=i*10+5,text=f'Synthetic source interval {i}',timing='synthetic') for i in range(3)]
        return dict(duration=60, audio=[], transcript=segments,available_modalities=['text','audio','video','audio_video'])

    def observe(self, row, action):
        return dict(time=action["time"], modality=action["modality"], media=[], segments=[], timing="synthetic")


def smoke(config):
    row = dict(id="synthetic:fixture", dataset="SYNTHETIC", split="train", person_id="fixture",
               video="NOT_A_REAL_VIDEO", segments=[dict(start=0, end=10, text="Synthetic fixture", timing="synthetic")],
               labels={t: 4 for t in "OCEAN"})
    trajectory = ActiveAgent(OfflineBackend(), OfflineTraitScorer(), SyntheticMedia(), config).run(row)
    assert len(trajectory["steps"]) == 1
    assert trajectory["stopped"]
    assert trajectory["offline"]
    assert all("labels" not in str(d["messages"]) for d in trajectory["decisions"])
    return trajectory
