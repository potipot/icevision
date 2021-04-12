from icevision.all import *
from fastai.metrics import accuracy


def test_create_learner(audio_dataloaders, pretrained_model):
    learner = nvidia_nemo.learner(dls=audio_dataloaders, model=pretrained_model)
    assert isinstance(learner, fastai.Learner)


def test_fastai_train(audio_dataloaders, pretrained_model):
    learner = nvidia_nemo.learner(
        dls=audio_dataloaders, model=pretrained_model, metrics=[accuracy]
    )
    learner.fine_tune(1)
