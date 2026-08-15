# AI Usage Log

Honest record of AI involvement, per session.

* **Session 0 — architecture (no code):** System design worked out in conversation with Claude Opus, acting as architect. ChatGPT and Gemini used as independent adversarial reviewers of the design; three corrections adopted (open-vocabulary detector over COCO `book` class, `work_id` catalog schema, three-tier ambiguity ladder). Final matcher specification frozen in `CLAUDE.md` before any code was written.
* **Implementation approach:** Claude Code implements against the frozen spec. I review every diff before committing. GPT/Gemini re-engage only at defined checkpoints for cold adversarial review, not for redesign.
