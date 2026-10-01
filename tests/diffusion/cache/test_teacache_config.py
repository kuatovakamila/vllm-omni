# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for TeaCache model-specific defaults (coefficients and estimator adapters)."""

import pytest

from vllm_omni.diffusion.cache.teacache.coefficient_estimator import _MODEL_ADAPTERS, ZImageAdapter
from vllm_omni.diffusion.cache.teacache.config import _MODEL_COEFFICIENTS, TeaCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_zimage_has_calibrated_coefficients():
    """Z-Image must not fall back to the Qwen-Image placeholder polynomial (#8270)."""
    zimage = _MODEL_COEFFICIENTS["ZImageTransformer2DModel"]
    assert len(zimage) == 5
    assert zimage != _MODEL_COEFFICIENTS["QwenImageTransformer2DModel"]

    config = TeaCacheConfig(transformer_type="ZImageTransformer2DModel")
    assert config.coefficients == zimage
    assert config.rel_l1_thresh == 0.2


def test_zimage_estimator_adapter_registered():
    assert _MODEL_ADAPTERS["ZImage"] is ZImageAdapter
    assert ZImageAdapter.model_class_name == "ZImagePipeline"
    assert ZImageAdapter.uses_tf_config is True
