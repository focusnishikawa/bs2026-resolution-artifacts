#!/usr/bin/env python3
"""IPSJ の要旨の字数・語数を数える (和文 600 字 / 英文 200 語).

⚠️ 数式は 1 語 (1 字) として数える。LaTeX の命令そのものは数えない。
usage: python3 train/count_abstract.py <tex> [<tex> ...]
"""
import re
import sys


def strip_tex(t, math_token="M"):
    t = re.sub(r"\$[^$]*\$", math_token, t)              # 数式は 1 トークン
    t = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", t)        # 強調は中身を残す
    t = re.sub(r"\\emph\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\ref\{[^{}]*\}", math_token, t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)                # 残る命令は削除
    t = t.replace("{", "").replace("}", "").replace("~", " ")
    return t.strip()


def main():
    for path in sys.argv[1:]:
        s = open(path, encoding="utf-8").read()
        for env, limit, kind in (("abstract", 600, "ja"), ("eabstract", 200, "en")):
            m = re.search(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (env, env), s, re.S)
            if not m:
                continue
            t = strip_tex(m.group(1))
            if kind == "ja":
                n = len(re.sub(r"\s", "", t))
                unit = "字"
            else:
                n = len([w for w in re.split(r"\s+", t) if w])
                unit = "語"
            flag = "  ⛔ 超過 %+d" % (n - limit) if n > limit else "  OK (残り %d)" % (limit - n)
            print("%-46s %-10s %4d %s / %d%s" % (path, env, n, unit, limit, flag))


if __name__ == "__main__":
    main()
