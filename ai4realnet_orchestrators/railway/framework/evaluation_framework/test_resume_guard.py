"""
Tests for the resume path: reusing a phase archive from a previous run.

--save-dir makes a campaign restartable, because a phase whose pickle is already on disk
is loaded instead of re-run. That is what makes a killed campaign cheap to continue. It
also means the run trusts a file it did not produce, so it has to check what is in it:
a save directory holding a 50-episode run would otherwise be reused wholesale by a
200-episode run, and the KPIs would be computed with a baseline of one length against
perturbed phases of another with nothing saying so.

A pickle truncated by a kill mid-write fails loudly inside pickle.load, so the case these
tests cover is the quiet one: a complete archive of the wrong length.
"""
import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_framework.result_getter import result_getter  # noqa: E402


class _Probe(result_getter):
    """Only the two attributes _load_or_run touches, and a run that records its calls."""

    def __init__(self, folder, n_episodes):
        self.save_folder = str(folder)
        self.n_episodes = n_episodes
        self.ran = []

    def run_episodes(self, path, attacker):
        self.ran.append(path)
        data = {"rewards": [[0.0]] * self.n_episodes}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return data


def _archive(path, n_episodes):
    with open(path, "wb") as f:
        pickle.dump({"rewards": [[0.0]] * n_episodes}, f)


def test_an_archive_of_the_requested_length_is_reused(tmp_path):
    pkl = tmp_path / "unperturbed.pkl"
    _archive(pkl, 200)
    probe = _Probe(tmp_path, 200)
    data = probe._load_or_run(str(pkl), None, "baseline")
    assert probe.ran == []
    assert len(data["rewards"]) == 200


def test_an_archive_of_a_different_length_is_rerun_not_mixed(tmp_path):
    # The defect this guards: a 200-episode archive silently serving a 50-episode run.
    pkl = tmp_path / "unperturbed.pkl"
    _archive(pkl, 200)
    probe = _Probe(tmp_path, 50)
    data = probe._load_or_run(str(pkl), None, "baseline")
    assert probe.ran == [str(pkl)]
    assert len(data["rewards"]) == 50


def test_a_shorter_archive_is_also_rerun(tmp_path):
    pkl = tmp_path / "geperturb.pkl"
    _archive(pkl, 50)
    probe = _Probe(tmp_path, 200)
    data = probe._load_or_run(str(pkl), None, "GEPerturb")
    assert probe.ran == [str(pkl)]
    assert len(data["rewards"]) == 200


def test_a_missing_archive_just_runs(tmp_path):
    pkl = tmp_path / "absent.pkl"
    probe = _Probe(tmp_path, 7)
    data = probe._load_or_run(str(pkl), None, "GEPerturb")
    assert probe.ran == [str(pkl)]
    assert len(data["rewards"]) == 7


def test_the_rerun_overwrites_the_stale_archive_so_it_is_not_reused_again(tmp_path):
    pkl = tmp_path / "unperturbed.pkl"
    _archive(pkl, 200)
    _Probe(tmp_path, 50)._load_or_run(str(pkl), None, "baseline")
    # A second 50-episode run must now hit the reuse path, not re-run again.
    second = _Probe(tmp_path, 50)
    second._load_or_run(str(pkl), None, "baseline")
    assert second.ran == []


def test_a_truncated_archive_fails_loudly_rather_than_being_reused(tmp_path):
    pkl = tmp_path / "unperturbed.pkl"
    _archive(pkl, 200)
    raw = pkl.read_bytes()
    pkl.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(Exception):
        _Probe(tmp_path, 200)._load_or_run(str(pkl), None, "baseline")
