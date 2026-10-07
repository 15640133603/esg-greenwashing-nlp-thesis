# -*- coding: utf-8 -*-
import re
with open('docs/论文全文.md', encoding='utf-8') as f:
    t = f.read()
t2 = re.sub(r'```.*?```', '', t, flags=re.S)
t2 = re.sub(r'!\[.*?\]\(.*?\)', '', t2)
t2 = re.sub(r'[|#*\->\[\]()`/]', '', t2)
cjk = len(re.findall(r'[\u4e00-\u9fff]', t2))
total = len(re.sub(r'\s', '', t2))
print('中文字符数:', cjk)
print('总字符数(不含空白):', total)
