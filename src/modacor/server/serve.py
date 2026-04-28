# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from modacor.server.api import create_app
from modacor.server.session_manager import SessionManager


def main():
    return create_app(SessionManager())


if __name__ == "__main__":
    raise SystemExit(main())
