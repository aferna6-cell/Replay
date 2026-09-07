"""Contract tests for the causal-aligned Phase 3T execution entrypoint."""


def test_aligned_entrypoint_installs_shared_attribution_functions():
    import ml.fidelity_phase_3t_aligned as aligned
    import ml.phase_3t_alignment_adapter as adapter
    import ml.t5_incumbent_synth_diagnostic as legacy

    # Re-install to make the contract independent of import order in the suite.
    adapter.install_into_legacy_module()

    assert legacy.first_synth_component is adapter.first_synth_component_aligned
    assert (
        legacy.decompose_t5_synth_pair
        is adapter.decompose_t5_synth_pair_aligned
    )
    assert callable(aligned.run_phase_3t)
    assert callable(aligned.main)


def test_aligned_entrypoint_preserves_legacy_runner_object():
    import ml.fidelity_phase_3t as legacy_runner
    import ml.fidelity_phase_3t_aligned as aligned

    # The wrapper changes attribution routing only; simulation execution remains
    # the existing preregistered Phase 3T runner.
    assert aligned._legacy_run_phase_3t is legacy_runner.run_phase_3t
    assert aligned._legacy_main is legacy_runner.main
