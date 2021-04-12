__all__ = ["learner"]

from icevision.imports import *
from icevision.engines.fastai import *
from .callbacks import NemoCallback


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs,
):
    """Fastai `Learner` adapted for nemo Matchboxnet model.

    # Arguments
        dls: `Sequence` of `DataLoaders` passed to the `Learner`.
        The first one will be used for training and the second for validation.
        model: The model to train.
        cbs: Optional `Sequence` of callbacks.
        **learner_kwargs: Keyword arguments that will be internally passed to `Learner`.

    # Returns
        A fastai `Learner`.
    """
    cbs = [NemoCallback()] + L(cbs)

    learner = adapted_fastai_learner(
        dls=dls, model=model, loss_func=model.loss, cbs=cbs, **learner_kwargs
    )

    # HACK: patch AvgLoss (in original, find_bs gives errors)
    class PatchedAvgLoss(fastai.AvgLoss):
        def accumulate(self, learn):
            bs = len(first(learn.yb))
            self.total += fastai.to_detach(learn.loss.mean()) * bs
            self.count += bs

    recorder = [cb for cb in learner.cbs if isinstance(cb, fastai.Recorder)][0]
    recorder.loss = PatchedAvgLoss()
    return learner
