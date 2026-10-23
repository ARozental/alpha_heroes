# Hidden information and battle memory (later)

Not now. Only once the model is really good at stand-alone battles, trained on thousands of full games; until
then this is noise. When it comes, keep it in its own module so that model.py stays readable.

The model must see only what its player knows, and say "unknown" explicitly (e.g. the enemy's spellbook, which
matters once scenarios give the heroes different spells).

1. A tiny recurrent (LSTM-like) memory that lasts the whole battle.
2. Written to only when hidden data is revealed (e.g. the enemy casts a spell we did not know it had).
3. Read for every decision.
4. Further out: the battle outputs this memory as a vector to the campaign map, so what was learned about the
   enemy is not lost; the campaign map also gives the battle its input vector (see the TODO in
   agents/AlphaBattleAI/BattleSim.h).

The search has the same problem: its copy of the battle starts from our client's game state, which knows the
enemy's spellbook (docs/battle_search.md, Risks).
