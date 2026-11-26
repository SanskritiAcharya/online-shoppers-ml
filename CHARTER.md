# Project Charter: Who should see the discount pop-up?

Written on 2026-11-26, before any model was trained. The success and kill criteria below were fixed at that point and were not changed after seeing results.

| Field | What it says |
|---|---|
| **Business problem** | An online store has about 12,000 browsing sessions in its log and only about 1 in 6 ends in a purchase (15.6%). When a visitor moves to leave, the store can show a discount pop-up. Today it either shows the pop-up to everyone, which annoys shoppers and gives away margin, or to nobody, which lets close-to-buying visitors walk away. |
| **Decision changed** | At the moment a visitor shows **exit intent** (the cursor heads for the close button or the back button), the marketing system decides **whether to show a discount pop-up** to this session. Two actions: show the pop-up, or let the visitor leave. |
| **Decision owner** | The e-commerce marketing manager. They own the pop-up budget and set how many sessions may get it. |
| **Capacity** | Pop-ups go to at most **10% of sessions**. More than that annoys shoppers and spends discount margin on people who were never going to buy. |
| **ML task** | Binary classification. For each session, predict the probability that it ends in a purchase. The top 10% of sessions by probability get the pop-up. |
| **Target** | `Revenue` = 1 if the session ended in a purchase. It is known at the end of the same session. |
| **Unit of prediction** | One browsing session, scored once, at the moment of exit intent. The features are treated as the state of the session at that moment. |
| **Baseline to beat** | Measured on the same split, not assumed. (1) Constant "nobody buys". (2) Rule A: "the session has a page value above zero". (3) Rule B, without page value: "20 or more product pages and an exit rate below 0.025". (4) Logistic regression on four obvious features (product pages, time on product pages, bounce rate, exit rate). |
| **Offline criterion** | On the held-out 20% of sessions, the model that would be deployed must reach **PR-AUC of at least 0.30**, and at a **10% pop-up rate its precision must be at least 0.35** (more than twice the 15.6% base rate). It must also beat the best rule that does not use page value by **at least 0.05 PR-AUC**. |
| **Business criterion** | For every 100 pop-ups, at least 35 should land on visitors who were close to buying, against about 16 if the 10% were picked at random. |
| **Guardrails** | No personal data (the dataset has none). Accuracy is never the headline, since "nobody buys" is already 84% accurate. New visitors must not be shut out: their precision at the 10% rate should not fall below half of the overall figure. Every prediction comes with a plain-words reason. Scoring takes under 200 ms, because the pop-up has to appear before the tab closes. |
| **Kill criteria** | Do not deploy if (a) the model does not beat Rule B on both PR-AUC and precision at 10%, or (b) the criteria are only met when `PageValues` is included and the site cannot supply page values at the moment of exit intent. After launch, stop if a hold-out group that never sees pop-ups buys just as often as the group that does. |
| **Known risks** | **Leakage:** `PageValues` is a Google Analytics metric worth the average value of pages seen before a transaction. A buying session's own transaction can feed it, so it may carry the answer. **Point in time:** all counters cover the whole session, including pages seen after the exit-intent moment. **Season:** November has a quarter of all buyers. January and April are missing. **Data quality:** 125 duplicate rows, "June" spelled out next to three-letter months, and integer codes that are really categories. |

## Is this really an ML problem?

Partly. If page value were available at exit intent, the one-line rule "page value above zero" would already do most of the work, and a dashboard rule would be enough. Page value is the suspect column, so the honest question is what can be done without it. Without it, no single counter separates buyers well, and a 10% cap needs a ranking of every session, not a yes or no. That is a job for a model. One more limit should be said up front. The model predicts who will buy, not who will buy *because of* the pop-up. Some flagged visitors would have bought anyway, and the data cannot tell them apart. Only a live test with a hold-out group can measure that.
