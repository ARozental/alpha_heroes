"""Imported only by the loaders' forkserver (supply.Supply): every module a loader worker imports, imported once there,
then every object frozen, so the workers forked from it share those pages and never copy them (2026-11-02 on the PC: a
worker that imported the campaign's modules after its fork and ran gc.collect() before gc.freeze() held 77 MB of its
own; one forked from this, 5 MB; 20 workers). CUDA is not initialized by these imports (a forked CUDA context is
unusable)."""
import gc

import battle_model.supply  # noqa: F401
import battle_model.train  # noqa: F401
import campaign_model.learner  # noqa: F401
import campaign_model.maps  # noqa: F401
import campaign_model.play  # noqa: F401

gc.collect()
gc.freeze()
