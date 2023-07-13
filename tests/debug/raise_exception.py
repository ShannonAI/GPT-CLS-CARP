#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/debug/raise_exception.py
@time: 2022/12/06 20:03
@desc:
"""
import time

import openai


def raise_service_exception():
    raise openai.error.ServiceUnavailableError


def raise_rate_exception():
    raise openai.error.RateLimitError


def test_handle_exception_last_return():
    try:
        raise_service_exception()
    except:
        print("Sleep")
        time.sleep(10)
        print(f"112.003")
    return "bbba"


def test_handle_exception():
    while True:
        try:
            return raise_rate_exception()
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError):
            print("Sleep")
            time.sleep(10)
            print(f"112.003")
        except:
            raise ValueError


if __name__ == "__main__":
    returned_str = test_handle_exception_last_return()
    print(returned_str)

    print("=" * 20)
    test_handle_exception()
