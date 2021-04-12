from icevision.imports import *
import pytest


@pytest.mark.skip
def test_torch_jit_script(scratch_model):
    sig = torch.rand(1, 4096, device=torch.device("cpu"))
    sig_len = torch.tensor([4096]).type_as(sig)
    scratch_model.export(
        "test.pt", input_example=(sig, sig_len), verbose=True, try_script=True
    )
    # scripted_model = torch.jit.script(scratch_model)
    # assert scripted_model
