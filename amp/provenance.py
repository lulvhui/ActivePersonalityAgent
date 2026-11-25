"""Program-owned transcript fragment identities; unverified ASR times stay unverified."""
from copy import deepcopy
import math
from .common import digest
ALIGNED_TIMING={'media_aligned','verified_media'}

def interval(value, allow_point=False):
    if value is None:
        return None
    try:
        start, end = map(float, value)
    except (ValueError, TypeError):
        return None
    if not all(math.isfinite(v) for v in [start, end]) or start < 0 or end < start or (end == start and not allow_point):
        return None
    return [start, end]

def index_transcript(segments, media_id, duration):
    """Assign stable segment IDs; ASR annotations remain retrieval hints, not alignment.

    A future aligner may supply media_time + verified_media/media_aligned. Existing
    raw datasets have unverified_asr, so their exact evidence time remains null.
    """
    result = []
    for i, original in enumerate(segments):
        segment = deepcopy(original)
        source_time = interval([segment.get("start"), segment.get("end")])
        aligned = interval(segment.get("media_time")) if segment.get("timing") in ALIGNED_TIMING else None
        if aligned is not None and aligned[1] > duration + 1e-6:
            raise ValueError("Aligned transcript outside source media")
        segment.update(segment_id="S" + digest([media_id, i, segment.get("text", "")]),
                       media_id=media_id, source_time=source_time, media_time=aligned,
                       timing="media_aligned" if aligned else "unverified_asr" if source_time else "unlocalized")
        result.append(segment)
    return result
