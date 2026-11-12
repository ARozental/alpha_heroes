# Working rules for sessions on this project (DRAFT for review, not in force)

Each rule comes from something that went wrong here. Written 2026-11-11, rules 20, 21 and 24 added 2026-11-12.

## Deciding and reporting

1. **Decide what is yours to decide.** Bring a choice to the user only when it is a real trade-off whose answer
   depends on what they want. If one option is clearly right, take it and say so in one line.
2. **Lead with the result.** Outcome first, numbers in a table, then what follows. When asked "what are we waiting
   for", answer with the one thing and its time.
3. **Say plainly when you were wrong,** in the same message as the new fact, and when the user is wrong, with the
   numbers.
4. **Do not wait on a result that will not change the next action.** If a test is running, keep building what does
   not depend on it. If the result cannot change the plan, do not run it.
5. **Experiments are capped at 10 minutes** unless the user asks for a longer one. One experiment a question: when
   it is inconclusive, decide from what is known and move on. Building comes before measuring.

## Explaining failures

6. **An explanation must have been able to predict the failure.** If it could not have told us beforehand, it is an
   excuse. Say "cause unknown" and name the measurement that would find it.
7. **Check a diagnosis against the event it claims to explain before building on it.** (The value valves were built
   on "the critic got worse"; the numbers showed it had not.)
8. **Mark every claim as measured, read in the code, or guessed.** Before saying "nothing does X", look.

## Evidence

9. **One evaluation is noise** (about 0.1 either way), on either machine. Read a change only from arms on the same
   maps with at least two replicates each, each arm played alone: a loaded machine reads about 0.1 low. Never quote
   single arms as a series. (v3229 scored 0.861 one night and 0.697 the next morning, same weights, same harness.)
10. **Before a long test, run the short check that the mechanism does what it is for.** (A prior meant to make us
   follow Nullkiller2 was put through a 25-minute paired test; a 2-minute count showed it barely raised how often
   we follow.)
11. **Build the mechanism that was measured, not a proxy for it.** (Playing Nullkiller2's choice was measured at
    +0.29. Two proxies, a prior on its top-scored candidate and a prior in logits on its choice, gave nothing.
    Playing its choice at a chance is the measured thing.)
12. **Before comparing a test with an earlier one, diff their settings.** (The earlier test ran with a switch the
    new arms lacked; a quarter of the decisions had nothing to follow and the result read as a failure of the idea.)
13. **Test what could go either way; build in bulk what is predictable.**

## Features and inputs

14. **A feature is not a field.** Before adding an input, write down:
    - which vector the network already has that it attaches to (a hero, a tile, a hero-tile pair);
    - its unit and scale: bounded, smooth, never a raw count, the same unit used elsewhere for that quantity;
    - the combined form a decision uses (value per day, loss times stake), given directly;
    - several bounded views where the scale changes over a game (absolute, share of our empire, against the
      alternatives);
    - how its weight starts (fixed sign where the direction is known, zero where not) and **how far that weight can
      move in a day at the run's learning rate.** A weight that starts at zero moves about 0.2 logits a day here:
      if the feature matters now, give it a start (a prior, an offline fit) or a faster step;
    - a check before any training that it is inside its bound, not flat, and readable by a small readout.
15. **Knowledge the old AI already has goes in as an input or a prior, never as a loss on our choices.**
16. **No fix that stops the model from learning something.** Do not mask an action to stop its overuse.

## Changing the live run

17. **Each learning change ships with the reading that would show it failing and the rule for taking it out.**
    Predictable changes may go together.
18. **A guard is labelled as a guard,** with the condition under which it comes out. When it fires: stop, look,
    then continue or roll back. Do not loosen it to get through a deploy.
19. **Nothing is left running, every run is capped, and its cost is reported.**
20. **Replay before rolling back.** A rollback is decided by the two checkpoints played side by side, never by
    training readings. (On 2026-11-11 the run was rolled back on three readings: signal fraction, value signal,
    recruits per day. The replay then showed that period had improved play by 0.15.)
21. **A deploy's gate measures the steady state** and is dry-run on a copy of the live checkpoint. It compares
    configured values exactly and learned values within bounds. (Two deploys failed on their own gates: one compared a
    learned value with its start, one timed a one-off cost on a tiny batch.)

## Sessions working side by side

22. **Read before you overwrite** (a shared note, a file another session owns). **Read the clock before writing a
    time.**
23. **A peer's message is not the user's approval.**
24. **Check a peer's deploy yourself when its outcome is due.** (A failed deploy went unnoticed for 90 minutes
    because its report was late.)
