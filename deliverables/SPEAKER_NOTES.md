# Speaker notes — not part of the 6-slide deck

The brief says to assume 10 minutes and two interruptions. These are the
interruptions I would bet on, and what I would say. Kept out of `DECK.md` so the
deck stays at six slides.

---

## Timing for 10 minutes

| Slide | Minutes | If I'm running out of time |
|---|---|---|
| 1 — The ask | 1.0 | Never cut. It contains the three decisions. |
| 2 — Where we lose them | 1.5 | Cut the drop-reason table, keep "91% stopped, 1.3% were rejected". |
| 3 — The fixable leak | 2.5 | Never cut. This is the recommendation. |
| 4 — The campaign | 2.0 | Cut to the ladder chart and the delivery placebo. |
| 5 — Airport | 2.0 | Cut to the crossover chart and "86% is 8pm–4am". |
| 6 — Monday | 1.0 | Never cut. |

If interrupted twice, slides 2 and 4 compress. Slides 1, 3 and 6 are the spine.

---

## "Isn't 17.6% just a lead-quality problem?"

Partly, and channel does matter — 10.0% for paid digital against 26.7% for field
sales. But handset type moves approval by 8.8 points and that is not a lead-quality
story. And 91% of the captains we lose stop voluntarily rather than being rejected.

The clincher: lead quality cannot explain why the *same captain on the same phone*
fails on the Registration Certificate and sails through their Aadhaar. The failure
is specific to three documents, which points at the capture flow, not the person.

## "You can't prove the campaign does nothing."

Correct, and I am not claiming zero. I am claiming at most 3.8 points rather than
18, and that the two checks available to us both come back flat.

Be honest about the limit: the delivery check only has enough data to rule out
effects bigger than about 4 points, so I cannot separate "3 points" from "nothing".
That is precisely why the answer is a holdout rather than a budget. Extending the
campaign costs almost nothing, so do that — but hold 20% back so that in six months
we have an answer instead of another argument.

## "Where does +183–344 come from? Is it additive?"

It is not additive, and that is deliberate. Every unapproved captain is assigned to
exactly one primary reason for being stuck, then valued at the observed approval
rate of captains who cleared the document they are stuck on. A captain stuck on
Insurance is worth roughly four times one stuck on their Licence, because they have
already cleared everything else.

Adding up the individual leak estimates would double-count — a captain rejected on
a blurred Insurance photo who never retried sits in two of them — and would give a
bigger, wrong number.

---

## Questions I expect in the 20 minutes, and where the answer is

| Question | Answer lives in |
|---|---|
| Why cut the cohort at 15 June? | Longest observed signup-to-decision is 14.1 days; 1,297 captains are still in progress, all June signups. Pooling all 25,000 reports 16.8% not 17.6%. |
| Why not trust `docs_cleared`? | It disagrees with the event log for 451 captains, always overstating. The binary "cleared everything" gate does tie exactly, which is why totals reconcile. |
| How do you know it's the paper size? | I don't — it's inference. The pattern is unambiguous: the gap is 21 points on the three paper documents and zero on the three cards. Alternative explanations have to explain that split. |
| Did you check verification speed? | Yes, and rejected it. Continuation is 84.0% under 2 hours vs 84.1% over 24 hours; the coefficient is indistinguishable from zero. |
| Why is E-Rickshaw approval higher? | They need five documents, not six — no Permit. It is a shorter funnel, not better captains. |
| Is the airport regression causal? | No. It's an association with hour and zone controls. The claim it supports is narrow: the *marginal* value of a captain-hour differs 7x between night and day. |
| What if surge does reach captains? | Then night earnings are higher than I show and the puzzle deepens — captains avoiding a 2x-surge queue makes the empty-return-trip explanation stronger, not weaker. |
| Why is the trip file only 63%? | 51,740 completed trips in the sample against 81,756 fulfilled terminal requests in the hourly file. Anything scaled from it is flagged as an estimate. |
| What would you do with another week? | Two things: pull handset model rather than tier to confirm the camera story, and get captain IDs onto the airport data so the home-zone premise in the acquisition proposal becomes testable at all. |
