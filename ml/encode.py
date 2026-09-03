"""Encode a Battlegrounds board into a fixed tensor for the neural net.

A board is a *set* of up to 7 minions. We encode it as a padded [7, F] array plus
a [7] mask, so a permutation-invariant (DeepSets / attention) encoder can pool
over the real minions and ignore the padding. Combat value depends on stats +
keywords (not tribe), so those are the features.
"""

import numpy as np

MAX_MINIONS = 7
# Per-minion features: attack, health, + 6 combat keywords.
FEATURE_NAMES = ["attack", "health", "divine_shield", "taunt", "poisonous",
                 "reborn", "windfury", "cleave"]
NUM_FEATURES = len(FEATURE_NAMES)


def board_to_array(board):
    """list[Combatant] -> (features[7,F] float32, mask[7] float32)."""
    arr = np.zeros((MAX_MINIONS, NUM_FEATURES), dtype=np.float32)
    mask = np.zeros((MAX_MINIONS,), dtype=np.float32)
    for i, c in enumerate(board[:MAX_MINIONS]):
        arr[i] = [
            float(c.attack), float(c.health),
            float(c.divine_shield), float(c.taunt), float(c.poisonous),
            float(c.reborn), float(c.windfury), float(c.cleave),
        ]
        mask[i] = 1.0
    return arr, mask
