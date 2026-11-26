# Monitoring runbook

The model scores every session that shows exit intent, so it runs all day. Purchase labels come back fast (the end of
the same session), but they come back **changed by the pop-up itself**: a visitor who got a discount and bought tells us
little about whether the model was right. So the plan watches the inputs and the flag rate every day, and measures real
value with a small hold-out group that never sees pop-ups. The drift job in `src/train.py` (`drift_check_psi` in MLflow,
`reports/psi_*.csv`) is the template for the PSI checks.

| Check | Frequency | Alarm | Action | Owner |
|---|---|---|---|---|
| API errors and latency | Continuous | Error rate > 1%, or p95 latency > 200 ms (the pop-up must appear before the tab closes) | Restart the container, check logs, roll back to the previous image tag. While it is down, show no pop-ups | Web engineer on call |
| Schema and nulls | Daily | Any null, a new `Month`, `VisitorType`, browser or traffic-source code, or a rate outside 0 to 1 | Find the tracking change. A new code is scored as "infrequent", so it works but is not trusted until retraining | Data analyst |
| PSI on top inputs (`ExitRates`, `ProductRelated_Duration`, `Month`, `ProductRelated`, `VisitorType`, `TrafficType`) and on the score | Weekly, last 7 days against the training sessions | Any PSI > 0.25 | Review in the Monday marketing meeting. Find out what changed (a campaign, a site redesign, the season) before trusting the flag rate | Data analyst, then marketing manager |
| Pop-up (flag) rate | Daily | Outside 7% to 13% of sessions (±30% of the 10% plan) | Re-set the threshold to the 90th percentile of the last 7 days of scores. If it moves a lot, look at the PSI check first | Data analyst |
| Realised value, hold-out group | Weekly, as orders settle | Flagged sessions in the hold-out group (no pop-up) buy less than 2x the site average, or the pop-up group does not buy more than the hold-out group | Below 2x: the ranking has decayed, so retrain. No lift: the pop-up itself does not work, so stop it and tell the marketing manager. This is the charter's kill criterion | Marketing manager |
| Discount cost | Monthly | Discount given away is larger than the margin of the extra sales measured against the hold-out group | Lower the discount or the 10% cap | Marketing manager |

**What the current check shows.** Between the training sessions and the random test sessions every input is stable
(largest PSI 0.01), which is the "no alarm" picture. A simulated holiday shift, November and December sessions against
February to October training sessions, fires the alarm on three things: traffic source (PSI 0.81), closeness to a
holiday (0.78, the dataset only marks spring holidays) and the model's own score (0.35). With one fixed threshold the flag
rate would jump from 4% in spring to 21% in the holidays, blowing the 10% budget. That is exactly what the flag-rate row
is for, and the threshold should be re-set before the season, not after. A model trained only on February to October
still ranks November and December sessions better than Rule B (PR-AUC 0.34 against 0.29), so the season alarm means
"re-set and watch", not "switch off".

**Hold-out group.** 10% of the sessions the model would flag never get a pop-up, picked at random. This is the only way
to tell a "save" from a discount given to someone who would have bought anyway.

**Retrain** every quarter on the last twelve months of sessions, and also when a PSI alarm is explained by a lasting
change (a redesign, a new traffic source). Before the holiday season, retrain or at least re-set the threshold on the
previous holiday season's sessions. **Roll back** to the previous registered version (MLflow alias `champion` points at
the version in use) if a new version flags outside 7% to 13% of sessions on its first day, if its round-trip check in
`train.py` fails, or if the hold-out comparison is worse after two weeks than it was for the old version.
