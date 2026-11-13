---
name: plan-task
description: Turn an idea into a task file in tasks/ with goal, scope, acceptance criteria and a duplication check. Use when someone proposes new work ("let's try X", "we need Y") before any code is written.
---

# plan-task

Input: an idea in a sentence or two (from the user's message).

1. Read `AGENTS.md` (goal and rules) and `tasks/README.md`.
2. **Duplication check**, reported to the user before writing anything:
   - `uv run possession methods`: does a method already do this, or could a parameter do it?
   - `ls tasks/` and the `status:` of each: is another task already covering it?
   - `uv run possession log`: was it already tried, and what happened?
   If it overlaps, propose extending the existing task or method instead and stop.
3. **Rules check**: does the idea need hand labels on this game, frame-specific constants, the
   public GT as a tuning signal, or human data about this game? If so, say so and propose a
   compliant variant (RULES.md).
4. Pick the next free number `NN`. Copy `tasks/_TEMPLATE.md` to `tasks/NN-short-slug.md` and fill
   in every section:
   - **Goal**: one sentence, plus how it moves spellF1 (which failure mode it attacks:
     flicker, late transitions, dead-ball misses, missing ball...).
   - **Scope**: the stage and files it may touch; what is explicitly out of scope.
   - **Interface**: the method name(s) and parameters it will register, or the feature it adds.
   - **Acceptance criteria**: checkable items: tests, `make check`, a logged run, and a
     label-free or other-game metric. Never "public spellF1 goes up".
   - `status: todo`, `owner:` empty unless the user names one.
5. Show the user the file. Do not start implementing; that is `/do-task NN`.
