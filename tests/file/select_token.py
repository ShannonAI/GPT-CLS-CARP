#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path

repo_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/token"
with open(os.path.join(repo_path, "all_quota.txt"), "r") as f:
    datal = [item.strip() for item in f.readlines()]
    datal = [item[:-2] for item in datal]

print(datal[0])

with open(os.path.join(repo_path, "out_of_quota.txt"), "r") as f:
    newdl = [item.strip() for item in f.readlines()]
    newdl = [item[1:-2] for item in newdl]

print(newdl[0])

counter = 0
token_lst = []
for d in datal:
    if d not in newdl:
        counter += 1
        token_lst.append(d)
        # print(d)

print(counter)
for dd in token_lst:
    print(f'"{dd}",')
