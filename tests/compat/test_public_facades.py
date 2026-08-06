"""Regression checks for the historic root-level module imports."""


def test_cpu_package_core_matches_legacy_facade():
    import cpu_baseline
    from baseline.core import run_cpu

    assert cpu_baseline.run_cpu is run_cpu
