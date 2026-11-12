# `verification/phase_reconstruction`

Shared four-step / N-step phase retrieval for **simulation and experiment**.

One algorithm path only: feed four intensity arrays
(`I0`, `Iπ/2`, `Iπ`, `I3π/2`) or an `(K,H,W)` stack with phase schedule.
Do not maintain a separate experimental reconstructor.
