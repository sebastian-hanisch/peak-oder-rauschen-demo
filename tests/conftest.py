"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_x():
    """n = 6, x = (1, 1, 0, 0, 0, 0): X_k = 1 + exp(-2 pi i k / 6), |X_1|^2 = 3, |X_2|^2 = 1, I = |X|^2 / n = (1/2, 1/6) an den inneren Frequenzen k = 1, 2 (m = 2);
    Summe 2/3, g = 3/4. Für m = 2 ist g auf [1/2, 1] gleichverteilt: P(g' > 3/4) = 1/2 (Formel: C(2, 1) (1 - 3/4) = 1/2)."""
    return np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
