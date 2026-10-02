"""Cancel a Hugging Face `generate()` call between tokens."""

from __future__ import annotations

from typing import Callable

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class _CheckCancelled(StoppingCriteria):
    """Never stops generation itself — calls the job's cancel check, which raises."""

    def __init__(self, check: Callable[[], None]) -> None:
        self.check = check

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> torch.BoolTensor:
        self.check()
        return torch.zeros((input_ids.shape[0],), dtype=torch.bool, device=input_ids.device)


def cancel_criteria(check: Callable[[], None]) -> StoppingCriteriaList:
    """Pass as `stopping_criteria=`: the check runs after every generated token,
    so a cancelled job stops in well under a second even on a slow machine."""
    return StoppingCriteriaList([_CheckCancelled(check)])
