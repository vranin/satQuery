"""Tests for agent task classification and input validation."""
from __future__ import annotations

from app.agent.router import classify_task, validate_inputs_for_task
from app.schemas.requests import TaskType


def test_router_classifies_vqa():
    assert classify_task("Is there a lake?", 1) == TaskType.VQA


def test_router_classifies_caption():
    assert classify_task("Describe this satellite image", 1) == TaskType.CAPTION


def test_router_classifies_change_detection():
    assert classify_task("What changed between these images?", 2) == TaskType.CHANGE_DETECTION


def test_router_classifies_cross_modal():
    assert classify_task("Compare the SAR and optical images", 2) == TaskType.CROSS_MODAL


def test_router_rejects_invalid_combinations():
    assert validate_inputs_for_task(TaskType.CHANGE_DETECTION, 1)
    assert validate_inputs_for_task(TaskType.CROSS_MODAL, 1)
    assert validate_inputs_for_task(TaskType.UNKNOWN, 0)