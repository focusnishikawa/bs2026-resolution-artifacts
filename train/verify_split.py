#!/usr/bin/env python3
"""分割前の ONNX と分割後 4 パートの連鎖が同じ出力を出すことを確かめる (経路 A の妥当性検証).

Orin で 4 分割ビルドが通っても、「分割しても分割前と同じ結果になる」ことを示さない限り
手法として使えない。ここは Orin を占有せずに済むよう HPC の onnxruntime で確認する
(グラフの等価性はランタイムに依らない。TRT/FP16 実機側の一致は別途 Orin で見る)。

usage:
    python3 verify_split.py <full.onnx> <split_dir> <tag> <input.bin> <res>
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def sess(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(path), so, providers=prov)


def main():
    full_path, split_dir, tag, bin_path, res = sys.argv[1:6]
    res = int(res)
    split_dir = Path(split_dir)

    x = np.fromfile(bin_path, dtype=np.float16).reshape(-1, 3, res, res)
    print("[input] %s shape=%s" % (bin_path, x.shape), flush=True)

    s_full = sess(full_path)
    in_name = s_full.get_inputs()[0].name
    print("[full ] %s  in=%s%s" % (Path(full_path).name, in_name, s_full.get_inputs()[0].shape), flush=True)
    ref = s_full.run(None, {in_name: x[:1]})[0]
    for i in range(1, len(x)):
        ref = np.concatenate([ref, s_full.run(None, {in_name: x[i:i+1]})[0]], axis=0)
    del s_full

    parts = sorted(split_dir.glob("%s_p*.onnx" % tag))
    print("[split] %d パート: %s" % (len(parts), [p.name for p in parts]), flush=True)
    sessions = [sess(p) for p in parts]
    for k, s in enumerate(sessions):
        i0, o0 = s.get_inputs()[0], s.get_outputs()[0]
        print("   p%d in=%-42s %s -> out=%-42s %s" % (k, i0.name[:42], i0.shape, o0.name[:42], o0.shape), flush=True)

    outs = []
    for i in range(len(x)):
        h = x[i:i+1]
        for s in sessions:
            h = s.run(None, {s.get_inputs()[0].name: h})[0]
        outs.append(h)
    got = np.concatenate(outs, axis=0)

    print("\n[ref  ] shape=%s" % (ref.shape,))
    print("[split] shape=%s" % (got.shape,))
    ref32, got32 = ref.astype(np.float64), got.astype(np.float64)
    d = np.abs(ref32 - got32)
    denom = np.maximum(np.abs(ref32), 1e-6)
    print("\n=== 一致の検査 (%d 枚 x %d クラス) ===" % ref.shape)
    print("  最大絶対差      : %.6g" % d.max())
    print("  平均絶対差      : %.6g" % d.mean())
    print("  最大相対差      : %.6g" % (d / denom).max())
    print("  argmax 一致     : %d / %d" % (int((ref.argmax(1) == got.argmax(1)).sum()), len(ref)))
    print("  完全ビット一致  : %s" % bool(np.array_equal(ref, got)))
    print("\n  [参考] 先頭 1 枚の logits")
    print("    分割前: %s" % np.array2string(ref32[0], precision=4, max_line_width=200))
    print("    分割後: %s" % np.array2string(got32[0], precision=4, max_line_width=200))

    ok = np.array_equal(ref, got)
    print("\n[RESULT] %s" % ("完全一致 (分割は出力を変えない)" if ok
                             else "差あり -> 最大絶対差 %.6g で判断が必要" % d.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
