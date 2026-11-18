# Data and provenance

The empirical data come from the existing Civitas experiment archives. The renderer performs read-only comparisons and uses these projections to produce figure assets.

- health.csv contains nine complete rule trajectories at N=100 and T=100 with three matched seeds. Each trajectory covers one model trading day.
- branches.csv and branch_metrics.csv give outcomes for eleven branches of one N=20 parent world. Peak infection uses the 27 event steps.
- branch_paths.csv contains all price paths, including the common pre-event checkpoint. The indexed price is 100 times the price divided by its pre-event value.
- branch_step_metrics.csv contains 308 committed state projections of infection share, mean model confidence, mean model emotion, last-trade price, and traded shares. The 28 observations per branch include u=0, for which event volume is blank. branch_step_metrics_provenance.json documents the extraction, checks on the state hashes, and diagnostic comparisons for the three policy branches.
- snapshot_cases.csv and snapshot_pairs.csv contain 54 fixed-observation outputs and 27 contrasts per backend. The signed quantity change is perturbed minus baseline, with BUY positive, SELL negative, and HOLD zero. The comparison of current state and template trace in RQ3 reads the archived request and case records at kdd27_completion_20261114/snapshot54/{requests,cases}.json. In all 54 requests, the private state and behavior records disagree on both risk and price references.
- snapshot_summary.json records batch time, token usage, and rate-based accounting.
- capacity.csv and execution_environments.json describe the runtime measurements and platforms. The ASUS record contains 12 saved simulation steps, a sampled peak, and a missing final profile. runtime_profile_evidence.json records the two Mac profiles, the 300 archived step timings, and the source code paths underlying the supported runtime interpretation.
- schedule_exposure.csv contains the offline announcement-step eligibility counts across 20 fixed seeds. The corresponding provenance gives the audit input identity and configuration.
- timing_pairs.csv contains three strict-validator timing repeats on the same N=100, T=12 rule world.

provenance.json gives archive paths relative to the engineering workspace and their SHA-256 identities. render_receipt.json records the renderer's source hashes. Figure styling follows the supplied manuscript template. Plotted measurements and offline eligibility counts use the archived data above.
