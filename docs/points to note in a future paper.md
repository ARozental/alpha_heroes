1. ❯ "No. Your own 2026-10-27 rule..." my rule forbade day+1 when it was used to affect policy directly, does it still make sense now?
  "a player's decisions share the week's luck." => explain better... let me see if I get it right 2 decisions for the same day are both affected by the luck of tommorow so they are not independent,a dacision in day X and a decision in day X+1 share some of the luck (like from day+2, but not all luck as the 7 day window is not exactly the same for both). This sounds like an optimizer issue that assumes behind the scenes that they are independent, in truth we know that are not and we have an estimation for what % of their luck is shared for each head (more for the day+7 head, 6 days of shared luck out of 7, and less for the day+2 head,1 day of shared luck out of 2).

  It seems to me we do not take that into account at all, how do we fix our optimization? maybe look for a general approach, someone surely trained models on examples that are not independent and they know what % of their luck is shared.


  2. heads learn different horizon and diff from last head, correlated luck results...
  3. all utility functions work, paperclips, but some work better. Try for 0 effect on unclear decisions.
  4. ops over features from hardcoded models, aim for batch0 to be similar to the hardcoded model
  5. dataset generation: hard examples... if they are unlearnable we have an architecture issue
  6. some intro about that we aim only for games that can not be solved by brute forcing the final label like alpha go
  7. metric: count dumb decisions.
  8. AI error section (what mistakes were made by the AI that can happen in future games)