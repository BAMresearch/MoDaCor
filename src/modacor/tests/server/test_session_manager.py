# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from modacor.server.session_manager import SessionManager


def test_session_manager_run_lifecycle():
    manager = SessionManager()
    manager.create_session(session_id="s1", pipeline_yaml="name: p\nsteps: {}\n")

    run = manager.enqueue_run("s1", mode="partial", effective_mode="full")
    assert run["status"] == "queued"
    assert run["mode"] == "partial"
    assert run["effective_mode"] == "full"

    done = manager.mark_run_succeeded("s1", run["run_id"], details={"num_steps": 3})
    assert done["status"] == "succeeded"
    assert done["num_steps"] == 3

    session = manager.get_session("s1")
    assert session is not None
    assert session.state == "idle"
    assert session.active_run_id is None


def test_session_manager_full_reset_clears_processing_data():
    manager = SessionManager()
    session = manager.create_session(session_id="s2", pipeline_yaml="name: p\nsteps: {}\n")
    session.processing_data = {"dummy": 1}

    manager.reset_session("s2", mode="full")
    session = manager.get_session("s2")
    assert session is not None
    assert session.processing_data is None
