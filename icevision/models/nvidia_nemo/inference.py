__all__ = ["predict_single_item_onnx", "predict_torchscript"]

import tempfile
import soundfile

# import onnxruntime
from icevision.imports import *
from icevision.core import *

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.asr.modules import AudioToMFCCPreprocessor
from nemo.utils import logging
from omegaconf import DictConfig


def write_files_data(manifest_file, audio_files):
    for audio_file in audio_files:
        # TODO: change this default value to 'unknown'
        default_value = "tak"
        entry = {
            "audio_filepath": audio_file,
            "duration": 100000,
            "command": default_value,
        }
        manifest_file.write(json.dumps(entry) + "\n")
    # back to the beginning of file
    manifest_file.seek(0)


def inference_dataloader(manifest_file, batch_size) -> "torch.utils.data.DataLoader":
    dl_config = {
        "manifest_filepath": manifest_file,
        "sample_rate": 16000,
        "labels": [
            "się",
            "nie",
            "jest",
            "na",
            "że",
            "ale",
            "to",
            "jak",
            "i",
            "przez",
            "tylko",
            "do",
            "jego",
            "może",
            "tak",
        ],
        "batch_size": batch_size,
        "trim_silence": True,
        "shuffle": False,
    }
    infer_dl = EncDecClassificationModel._setup_dataloader_from_config(
        None, config=dl_config
    )
    return infer_dl


def predict(asr_model, paths2audio_files: List[str], batch_size: int = 4) -> Dict:
    """
    Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

    Args:

        paths2audio_files: (a list) of paths to audio files. \
    Recommended length per file is between 5 and 25 seconds. \
    But it is possible to pass a few hours long file if enough GPU memory is available.
        batch_size: (int) batch size to use during inference. \
    Bigger will result in better throughput performance but would use more memory.
        logprobs: (bool) pass True to get log probabilities instead of transcripts.

    Returns:

        A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
    """

    propabilities, idxs = [], []
    # Model's mode and device
    mode = asr_model.training
    device = next(asr_model.parameters()).device
    try:
        # Switch model to evaluation mode
        asr_model.eval()
        logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

        # operate on a temporary file and automatically cleanup
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as manifest_file:
            write_files_data(manifest_file, paths2audio_files)
            infer_dl = inference_dataloader(manifest_file.name, batch_size)

            for infer_batch in infer_dl:
                logits = asr_model.forward(
                    input_signal=infer_batch[0].to(device),
                    input_signal_length=infer_batch[1].to(device),
                )
                propability, idx = torch.nn.functional.softmax(logits, dim=1).max(1)
                propabilities.extend(propability.tolist())
                idxs.extend(idx.tolist())
    finally:
        # set mode back to its original value
        asr_model.train(mode=mode)
        logging.set_verbosity(logging_level)
    return {"propabilities": propabilities, "idxs": idxs}


def load_and_preprocess(audio_file):
    # read waveform
    signal, sr = soundfile.read(audio_file)
    signal_length = len(signal)

    # apply padding
    signal = np.pad(signal, (0, 8192 - signal_length))

    signal = np.expand_dims(signal, 0).astype(np.float32)
    signal_length = np.expand_dims(signal_length, 0).astype(np.int64)
    return signal, signal_length


def postprocess(raw_output, class_map: Optional[ClassMap] = None):
    logits = torch.tensor(raw_output).squeeze()
    propability, idx = torch.nn.functional.softmax(logits, dim=0).max(0)
    result = {"propability": propability, "idx": idx}
    if class_map is not None:
        result["label"] = class_map.get_by_id(idx)
    return result


def predict_single_item_onnx(
    onnx_model_path, audio_file, class_map: Optional[ClassMap] = None
):
    signal, signal_length = load_and_preprocess(audio_file)
    ort_session = onnxruntime.InferenceSession(str(onnx_model_path))

    input_feed = {"0": signal, "1": signal_length}
    raw_onnx_output = ort_session.run(output_names=["887"], input_feed=input_feed)
    result = postprocess(raw_onnx_output, class_map)
    return result


def predict_torchscript(
    torchscript_model_path, audio_file, class_map: Optional[ClassMap] = None
):
    signal, signal_length = load_and_preprocess(audio_file)
    signal = torch.from_numpy(signal)
    signal_length = torch.from_numpy(signal_length)

    model = torch.jit.load(torchscript_model_path)
    raw_torchscript_output = model(signal, signal_length)
    result = postprocess(raw_torchscript_output, class_map)
    return result


def predict_onnx_from_nemo_format(onnx_model_path, file_list, batch_size):
    propabilities, idxs = [], []
    preprocessor_cfg = {
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "n_mels": 64,
        "n_mfcc": 64,
        "n_fft": 512,
    }
    preprocessor = AudioToMFCCPreprocessor(**preprocessor_cfg)
    # preprocessor = EncDecClassificationModel.from_config_dict(preprocessor_cfg)
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    with tempfile.NamedTemporaryFile("w+", suffix=".json") as manifest_file:
        write_files_data(manifest_file, file_list)
        infer_dl = inference_dataloader(manifest_file.name, batch_size)

        for test_batch in infer_dl:
            # ort_session.get_inputs()[0].name == 'audio_signal'

            input_names = [onnx_input.name for onnx_input in ort_session.get_inputs()]
            processed_signal, processed_signal_len = preprocessor(
                input_signal=test_batch[0], length=test_batch[1]
            )
            processed_inputs = {next(iter(input_names)): processed_signal.numpy()}
            onnx_output = ort_session.run(
                output_names=["logits"], input_feed=processed_inputs
            )
            # input_feed = {'input_signal': test_batch[0].numpy(), 'audio_lengths': test_batch[1].numpy()}
            # onnx_output = ort_session.run(output_names=['logits'], input_feed=input_feed)
            logits = torch.tensor(onnx_output).squeeze()
            propability, idx = torch.nn.functional.softmax(logits, dim=1).max(1)
            propabilities.extend(propability.tolist())
            idxs.extend(idx.tolist())
    return {"propabilities": propabilities, "idxs": idxs}
