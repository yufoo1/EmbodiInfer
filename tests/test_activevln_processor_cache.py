"""Prompt caching keeps fresh pixels, grid-dependent tokens and caller isolation."""

from collections import OrderedDict

import numpy as np
import torch

from embodiinfer.policies.activevln.processor_activevln import ActiveVLNProcessor
from test_activevln_batching import _observation


class _Encoded(dict):
    __getattr__ = dict.__getitem__


class _Processor:
    def __init__(self):
        self.tokenizations = 0

    def image_processor(self, *, images, return_tensors):
        assert return_tensors == "pt"
        pixels = torch.from_numpy(np.array(images[0]).copy()).float()
        # Also exercise a processor whose grid can change for the same raw shape.
        size = 4 if pixels.max() > 128 else 2
        return _Encoded(pixel_values=pixels, image_grid_thw=torch.tensor([[1, size, size]]))

    def __call__(self, *, text, images, padding, return_tensors):
        assert len(text) == 1 and padding is False
        self.tokenizations += 1
        result = self.image_processor(images=images, return_tensors=return_tensors)
        result["input_ids"] = torch.arange(int(result.image_grid_thw.prod()))[None]
        result["attention_mask"] = torch.ones_like(result.input_ids)
        return result


def _cached_processor():
    processor = ActiveVLNProcessor.__new__(ActiveVLNProcessor)
    processor._processor = _Processor()
    processor.action_space = "r2r"
    processor.text_cache_size = 1
    processor._text_cache = OrderedDict()
    return processor


def test_cached_tokens_do_not_cache_images_or_share_mutable_tensors():
    processor = _cached_processor()
    first = _observation("walk forward")
    initial = processor.process_turn(first, initial=False)
    expected_ids = initial.input_ids.clone()
    initial.input_ids.fill_(999)
    next_frame = _observation("walk forward")
    next_frame.images.fill_(0.5)
    cached = processor.process_turn(next_frame, initial=False)
    assert processor._processor.tokenizations == 1
    torch.testing.assert_close(cached.input_ids, expected_ids, rtol=0, atol=0)
    assert not torch.equal(cached.pixel_values, initial.pixel_values)
    assert next(iter(processor._text_cache.values())).pixel_values.numel() == 0
    cached.input_ids.fill_(888)
    cached.attention_mask.zero_()
    cached.image_grid_thw.zero_()
    again = processor.process_turn(first, initial=False)
    torch.testing.assert_close(again.input_ids, expected_ids, rtol=0, atol=0)
    assert again.attention_mask.all()
    assert again.image_grid_thw.tolist() == [[1, 2, 2]]


def test_cache_checks_actual_grid_and_evicts_old_instructions():
    processor = _cached_processor()
    first = _observation("walk forward")
    processor.process_turn(first, initial=False)
    changed = _observation("walk forward")
    changed.images.fill_(1)
    actual = processor.process_turn(changed, initial=False)
    assert processor._processor.tokenizations == 2
    assert actual.input_ids.shape[1] == 16
    processor.process_turn(_observation("turn left"), initial=False)
    assert len(processor._text_cache) == 1
    processor.process_turn(first, initial=False)
    assert processor._processor.tokenizations == 4
