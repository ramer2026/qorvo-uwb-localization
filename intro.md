# UWB Indoor Localization with Bayesian Fingerprinting

**Distance and angle tell you where you might be. Fingerprinting tells you where you are, and the Bayesian model tells you when not to trust it.**

The Qorvo QM35825 measures distance to sub-10 cm accuracy and reports angle-of-arrival in real time, both visible live in the UWB Explorer GUI. The problem is that distance from one anchor is a sphere, and angle narrows it to a cone. Even with two anchors, the intersection of two cones in 3D space is not unique. Raw ranging and AoA alone cannot resolve a 3D position.

We fixed that with fingerprinting. We measured at 24 grid points under both clear and obstructed conditions, across both anchors. Every location builds up a statistical signature across distance, AoA azimuth, AoA elevation, RSSI, and signal diagnostics. The model learns those signatures, including what obstructed signals look like, without needing to solve any geometry.

AoA cuts 3D localization error by 8-9% over ranging alone, but only where the angular geometry actually separates nearby points. Where obstruction deflects the signal, the angle reading is corrupted and AoA stops helping. The Bayesian model handles this by flagging those cases with low confidence instead of silently returning a wrong answer. You get a ranked list of candidate locations and a probability score, so you know when to trust the prediction and when not to.

**Mean 3D localization error: ~0.26 m &nbsp;|&nbsp; Obstruction detection: ~77% accuracy &nbsp;|&nbsp; Live demo: UWB Explorer GUI**

---

![System block diagram](docs/figures/setup.jpg)
*Two Qorvo QM35825 anchors (fixed) ranging to a mobile tag across a 24-point 3D measurement grid, with host-side Bayesian localization.*

---

> 📹 **See it live** — [Add link to your demo video here]
