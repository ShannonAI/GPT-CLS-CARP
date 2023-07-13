#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/file/collect_account_info.py
@time: 2022/12/06 20:03
@desc:
"""

import os
from typing import List


def collect_left_lst(file_path: str):
    with open(file_path, "r") as f:
        datal = [item.strip() for item in f.readlines()]
        tokenl = [item[1:-2] for item in datal]

        print("=" * 10)
        print(datal[0])
        print(tokenl[0])

    return tokenl


def load_info_text(file_path_lst: List[str]):
    info_l = []
    for file in file_path_lst:
        with open(file, "r") as f:
            datal = [item.strip() for item in f.readlines()]

            info_l.extend(datal)

    return info_l


def main():
    repo_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/token"
    save_file = os.path.join(repo_path, "left_token_info.tsv")
    file_path_lst = ["token_v1.txt", "token_v2.txt", "token_v3.txt", "token_v4.txt"]
    file_path_lst = [os.path.join(repo_path, item) for item in file_path_lst]
    info_l = load_info_text(file_path_lst)
    token_file = os.path.join(repo_path, "left_token_list.txt")
    left_tokenl = collect_left_lst(token_file)

    print(len(left_tokenl))
    token_map = {}
    for left_l in left_tokenl:
        for info in info_l:
            if left_l in info:
                token_map[left_l] = info
    print(len(token_map))
    with open(save_file, "w") as f:
        for k, v in token_map.items():
            # 邮箱账号-邮箱密码-账号密码-api
            infos = v.split("----")
            email = infos[0]
            openai_k = infos[2]
            email_k = infos[1]
            token = f'{k}'
            f.write(f"{token}\t{email}\t{openai_k}\t{email_k}\n")


if __name__ == "__main__":
    main()
