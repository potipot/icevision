from icevision.imports import *
from icevision.core import *
from icevision.models.utils import *
from nemo.collections.asr.data.audio_to_text import _speech_collate_fn


def train_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.

    # Arguments
        dataset: Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
        batch_tfms: Transforms to be applied at the batch level.
        **dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`.
        The parameter `collate_fn` is already defined internally and cannot be passed here.

    # Returns
        A Pytorch `DataLoader`.
    """
    return transform_dl(
        dataset=dataset,
        build_batch=build_train_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def valid_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.

    # Arguments
        dataset: Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
        batch_tfms: Transforms to be applied at the batch level.
        **dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`.
        The parameter `collate_fn` is already defined internally and cannot be passed here.

    # Returns
        A Pytorch `DataLoader`.
    """
    return train_dl(dataset, batch_tfms, **dataloader_kwargs)


def build_train_batch(
    records: Sequence[RecordType],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Builds a batch in the format required by the model when training.

    # Arguments
        records: A `Sequence` of records.
        batch_tfms: Transforms to be applied at the batch level.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be an updated list
        of the input records with `batch_tfms` applied.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_train_batch(records)
    outs = model(*batch)
    ```
    """

    def record_to_audiolabeldataset_output(record):
        audio_signal = record.wav.squeeze()
        assert audio_signal.ndim == 1
        audio_length = torch.tensor(record.wav.shape[-1])
        token = torch.tensor(record.classification.label_ids).squeeze()
        token_length = torch.tensor(1)
        return audio_signal, audio_length, token, token_length

    batch = [record_to_audiolabeldataset_output(record) for record in records]
    audio_signals, audio_lengths, tokens, tokens_lengths = _speech_collate_fn(
        batch=batch, pad_id=0
    )

    return (audio_signals, audio_lengths, tokens, tokens_lengths), records
