"""
Make an attacker's randomness reproducible.

WHY THIS EXISTS
---------------
Every attacker already takes a `seed` and uses it for a numpy `RandomState`. Nothing
seeded **torch**, and the learned attackers (SAC, PPO, RLPerturb, Lambda-PIR) build torch
networks: their weight initialisation and any torch-side sampling came from torch's
global, unseeded generator. The evaluation was therefore not reproducible, which was
measured rather than suspected -- two consecutive runs of the same 4-episode evaluation,
same code, same arguments, same environment seeds, gave:

    KPI-SF-072  27.8750  vs  27.2083
    KPI-FF-070   0.2649  vs   0.2026      (a 24% swing)
    KPI-RF-078   0.7178  vs   0.7771
    KPI-AF-074  1293.89  vs  1256.66

Nine of the ten KPIs differed. A paired significance test inside one run stays valid --
every attacker and the baseline still traverse the same episode seeds -- but a reported
KPI value could not be re-derived, and a difference between two runs could not be told
apart from a difference between two agents.

WHAT IT SEEDS, AND WHY GLOBALLY
-------------------------------
`seed_attacker_rngs()` seeds torch (CPU and CUDA), numpy's global generator and Python's
`random`, because an attacker's network initialisation reads the global generators and
there is no per-module alternative in torch. It is therefore called immediately before
each attacker is constructed, with that attacker's own declared seed, so each one is
deterministic and independent of how many attackers were built before it.

Seeding globally is a deliberate choice with a consequence worth stating: it also pins
anything else that draws from those generators afterwards. That is why it is called per
attacker at construction time rather than once at import, and why the defender policy is
loaded before the attackers -- its weights come from a checkpoint, not from the RNG.
"""

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_attacker_rngs(seed: int) -> None:
    """Seed every generator an attacker's construction and sampling can draw from."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        # torch is optional; the attackers that need it fail loudly elsewhere.
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
