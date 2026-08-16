# AI Usage Log

Honest record of AI involvement. Short version: **AI wrote most of the code in
this repository, under a frozen specification I wrote and reviewed against, and
I reviewed every diff before it was committed.**

## Tools

| Tool | Used for |
|---|---|
| **Claude Opus (chat)** | Architecture design, before any code existed |
| **ChatGPT, Gemini** | Independent adversarial review of that design |
| **Claude Code (Opus)** | Nearly all implementation, in reviewed checkpoints |

## How it went, in order

**Session 0 — architecture, no code.** System design worked out in conversation
with Claude Opus acting as architect. ChatGPT and Gemini were used as cold
adversarial reviewers of the design; three of their corrections were adopted
(open-vocabulary detector over the COCO `book` class, `work_id` in the catalog
schema, the three-tier ambiguity ladder). The resulting matcher specification
was frozen into `CLAUDE.md` before a line was written.

**Implementation.** Claude Code implemented against that frozen spec in
checkpoints: catalog → matcher → detector → VLM layer → Django API → Expo app.
Each checkpoint was proposed as a file-by-file plan, shown to me for approval
before writing, and gated on a verification step (tests, `manage.py check`,
`tsc`, `expo-doctor`) before moving on. I reviewed every diff and stopped
several from landing.

## Corrections I made to AI output

Worth listing, because "AI wrote it" should not imply "AI got it right":

- **Systemic failures were being disguised as data.** The first VLM design
  returned N `legible:false` books when the API key was missing. I required a
  structured exception hierarchy instead — a missing key must not look like an
  unreadable shelf.
- **A metric that lied on the failure path.** `requests_made` was assigned after
  a call that could raise, so an exhausted-retry failure would have reported
  zero hosted requests. Now recorded in a `finally`.
- **Undercounted tokens.** `_usage_dict()` omitted `thoughts_token_count`; the
  real 16-crop call showed 1,626 such tokens — 8.2% of the total, and billable.
- **A catalog consistency bug.** `run_scan()` matched against an injected
  catalog but resolved display strings from the globally cached one. Fixed by
  deriving both from a single resolution point.
- **A 404 contract violation.** The confirm endpoint validated the body before
  looking up the item, so an unknown id with a bad body returned 400.
- **Orphaned uploads.** Cleanup covered the pipeline but not persistence.
- **Missing discard.** The first review UI offered confirm and correct but no
  discard, against the brief's explicit "confirms, corrects, or discards".
- **Decision state that evaporated.** Confirm/discard state lived inside
  components that unmount on a tab switch.
- **Two unverified claims in comments/docs**, corrected once measured: a
  `validate_title` comment that misdescribed DRF's whitespace trimming, and a
  cost note asserting free-tier usage that we had not actually verified for
  this key.

## What I did not use AI for

- Choosing what to cut against the deadline.
- Deciding the human-in-the-loop boundary (nothing persists without a tap).
- The catalog's trap design — the specific ambiguities were chosen by hand,
  though the row bulk was LLM-generated and then hand-checked.
- Accepting numbers. Every figure in the README was measured on this machine;
  where a number could not be measured (per-token pricing), the source is cited
  and the result is labelled an estimate.

## Where AI was most and least useful

**Most:** boilerplate with a precise spec (serializers, retry/backoff, test
scaffolding), and holding a long contract in view across a multi-day build.

**Least:** judgement about failure semantics. Left alone it consistently
preferred returning *something* over failing loudly, which is exactly wrong for
a pipeline whose whole point is that the model is sometimes wrong. Most of my
corrections above are variations on that single theme.
