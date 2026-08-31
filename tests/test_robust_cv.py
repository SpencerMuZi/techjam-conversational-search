from __future__ import annotations

import unittest
from importlib.util import find_spec

import numpy as np

HAS_EXPERIMENT_DEPS = find_spec("sklearn") is not None
if HAS_EXPERIMENT_DEPS:
    from scripts.cv_regularized_lambdarank import grouped_folds, profile_group_ids


@unittest.skipUnless(HAS_EXPERIMENT_DEPS, "requires optional experiment dependencies")
class RobustCrossValidationTest(unittest.TestCase):
    def test_identical_profiles_share_one_group(self) -> None:
        samples = [
            {"user_profile": {"summary": "same", "preference_tags": ["x"]}},
            {"user_profile": {"preference_tags": ["x"], "summary": "same"}},
            {"user_profile": {"summary": "different", "preference_tags": []}},
        ]
        groups = profile_group_ids(samples)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_grouped_folds_never_split_a_profile(self) -> None:
        scenarios = np.asarray(
            ["buying", "browsing", "buying", "browsing"] * 5,
            dtype=object,
        )
        profiles = np.repeat(np.arange(10), 2)
        folds = grouped_folds(scenarios, profiles, n_splits=5, seed=0)
        seen_test_sessions = []
        for train_sessions, test_sessions in folds:
            self.assertFalse(
                set(profiles[train_sessions]) & set(profiles[test_sessions])
            )
            seen_test_sessions.extend(test_sessions.tolist())
        self.assertEqual(sorted(seen_test_sessions), list(range(20)))


if __name__ == "__main__":
    unittest.main()
