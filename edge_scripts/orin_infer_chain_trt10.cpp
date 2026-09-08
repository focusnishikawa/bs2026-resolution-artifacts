// 分割した ViT-L の TRT エンジンを順に実行して logits を得る (経路 A の等価性検証用).
//
// ViT-L は Orin (TRT 8.5.2) では Transformer 全体が単一 MYELIN ForeignNode に融合され、
// その一括コンパイルが OOM を起こす。ブロック境界で 4 分割するとビルドが通るが、
// 「分割しても分割前と同じ出力になる」ことを確かめない限り手法として使えない。
// そのためのプログラム。
//
// パート間で受け渡すテンソルは 1 本だけ (各 block 出口がグラフのくびれ) なので、
// 前段の出力デバイスバッファをそのまま次段の入力バッファに渡せばよく、
// ホストへのコピーは一切不要。実運用のコストもこの形と同じになる。
//
// build:
//   g++ orin_infer_chain.cpp -O2 -std=c++14 \
//     -I/usr/include/aarch64-linux-gnu -I/usr/local/cuda/include \
//     -L/usr/lib/aarch64-linux-gnu -L/usr/local/cuda/lib64 \
//     -lnvinfer -lcudart -o orin_infer_chain
// run:
//   ./orin_infer_chain <out.csv> <n_imgs> <input.bin> <engine0> [engine1 ...]
//
// TensorRT 8 と 10 の両方でビルドできる。JetPack 6.2 で TRT が 8.5.2 -> 10.3.0 に上がり、
// **binding 番号ベースの API が全て削除された**ため、下の互換層で吸収している。
//   getNbBindings      -> getNbIOTensors
//   bindingIsInput     -> getTensorIOMode
//   getBindingName     -> getIOTensorName
//   getBindingDimensions/DataType/FormatDesc -> getTensorShape/DataType/FormatDesc
//   enqueueV2(bindings) -> setTensorAddress x N + enqueueV3
// 互換層より下の本体は版によらず同じコードである。
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>

using namespace nvinfer1;

#if NV_TENSORRT_MAJOR >= 10
  #define TRT10_API 1
#else
  #define TRT10_API 0
#endif

class Logger : public ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kERROR) fprintf(stderr, "[TRT] %s\n", msg);
  }
} gLogger;

static size_t elemSize(DataType t) {
  switch (t) {
    case DataType::kFLOAT: return 4;
    case DataType::kHALF:  return 2;
    case DataType::kINT32: return 4;
    case DataType::kINT8:  return 1;
#if TRT10_API
    // TRT 10 では INT64 が既定の整数型になった経路がある。取り違えると
    // バイト数計算が半分になり、境界チェックが誤って通ってしまう
    case DataType::kINT64: return 8;
#endif
    default: return 4;
  }
}

// ---- TRT 8 / 10 の入出力アクセス互換層 ----
// TRT10 は名前ベース、TRT8 は番号ベース。識別子の型ごと切り替える
#if TRT10_API
typedef const char* TensorId;
static inline int   ioCount(ICudaEngine* e)             { return e->getNbIOTensors(); }
static inline TensorId ioId(ICudaEngine* e, int i)      { return e->getIOTensorName(i); }
static inline bool  ioIsInput(ICudaEngine* e, TensorId t) {
  return e->getTensorIOMode(t) == TensorIOMode::kINPUT;
}
static inline const char* ioName(ICudaEngine*, TensorId t) { return t; }
static inline Dims     ioShape(ICudaEngine* e, TensorId t) { return e->getTensorShape(t); }
static inline DataType ioType(ICudaEngine* e, TensorId t)  { return e->getTensorDataType(t); }
static inline const char* ioFmt(ICudaEngine* e, TensorId t) { return e->getTensorFormatDesc(t); }
#else
typedef int TensorId;
static inline int   ioCount(ICudaEngine* e)             { return e->getNbBindings(); }
static inline TensorId ioId(ICudaEngine*, int i)        { return i; }
static inline bool  ioIsInput(ICudaEngine* e, TensorId t) { return e->bindingIsInput(t); }
static inline const char* ioName(ICudaEngine* e, TensorId t) { return e->getBindingName(t); }
static inline Dims     ioShape(ICudaEngine* e, TensorId t) { return e->getBindingDimensions(t); }
static inline DataType ioType(ICudaEngine* e, TensorId t)  { return e->getBindingDataType(t); }
static inline const char* ioFmt(ICudaEngine* e, TensorId t) { return e->getBindingFormatDesc(t); }
#endif

static const char* dtypeName(DataType t) {
  switch (t) {
    case DataType::kFLOAT: return "fp32";
    case DataType::kHALF:  return "fp16";
    case DataType::kINT32: return "int32";
    case DataType::kINT8:  return "int8";
    default: return "?";
  }
}

// fp16 (IEEE half) -> float. 出力 logits を人が読める形で保存するために使う
static float halfToFloat(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) & 1;
  uint32_t exp  = (uint32_t)(h >> 10) & 0x1f;
  uint32_t man  = (uint32_t)h & 0x3ff;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) { f = sign << 31; }
    else {  // 非正規化数
      exp = 127 - 15 + 1;
      while ((man & 0x400) == 0) { man <<= 1; exp--; }
      man &= 0x3ff;
      f = (sign << 31) | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7f800000 | (man << 13);
  } else {
    f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out; memcpy(&out, &f, 4); return out;
}

struct Stage {
  ICudaEngine* engine;
  IExecutionContext* ctx;
  TensorId inIdx, outIdx;
  int nb;
  bool haveIn, haveOut;
  size_t inBytes, outBytes;
  size_t outElems;
  DataType inType, outType;
};

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <out.csv> <n_imgs> <input.bin> <engine0> [engine1 ...]\n", argv[0]);
    return 1;
  }
  const char* outPath = argv[1];
  int n = atoi(argv[2]);
  const char* binPath = argv[3];
  int nStage = argc - 4;

  IRuntime* runtime = createInferRuntime(gLogger);
  std::vector<Stage> st(nStage);

  for (int s = 0; s < nStage; s++) {
    std::ifstream ef(argv[4 + s], std::ios::binary);
    if (!ef.good()) { fprintf(stderr, "engine open failed: %s\n", argv[4 + s]); return 1; }
    std::vector<char> buf((std::istreambuf_iterator<char>(ef)), std::istreambuf_iterator<char>());
    Stage& x = st[s];
    x.engine = runtime->deserializeCudaEngine(buf.data(), buf.size());
    if (!x.engine) { fprintf(stderr, "deserialize failed: %s\n", argv[4 + s]); return 1; }
    x.ctx = x.engine->createExecutionContext();
    x.nb = ioCount(x.engine);
    x.haveIn = x.haveOut = false;
    for (int i = 0; i < x.nb; i++) {
      TensorId t = ioId(x.engine, i);
      if (ioIsInput(x.engine, t)) { if (!x.haveIn)  { x.inIdx  = t; x.haveIn  = true; } }
      else                        { if (!x.haveOut) { x.outIdx = t; x.haveOut = true; } }
    }
    if (!x.haveIn || !x.haveOut) {
      fprintf(stderr, "[NG] stage%d の入出力テンソルが見つからない (nb=%d)\n", s, x.nb);
      return 1;
    }
    auto bytesOf = [&](TensorId t, size_t* elems) {
      Dims d = ioShape(x.engine, t);
      size_t e = 1;
      for (int i = 0; i < d.nbDims; i++) e *= (size_t)d.d[i];
      if (elems) *elems = e;
      return e * elemSize(ioType(x.engine, t));
    };
    size_t ie = 0;
    x.inBytes  = bytesOf(x.inIdx, &ie);
    x.outBytes = bytesOf(x.outIdx, &x.outElems);
    x.inType  = ioType(x.engine, x.inIdx);
    x.outType = ioType(x.engine, x.outIdx);
    // バイト数と dtype が一致していてもメモリレイアウトが違えば中身は壊れる。
    // TRT は FP16 の binding に kHWC8 等のベクトル化フォーマットを選ぶことがあるので、
    // 段をまたいで受け渡す以上ここを必ず確認する。
    fprintf(stderr, "[stage%d] in=%s(%s, %zu B, fmt='%s') out=%s(%s, %zu elems, %zu B, fmt='%s')\n",
            s, ioName(x.engine, x.inIdx), dtypeName(x.inType), x.inBytes,
            ioFmt(x.engine, x.inIdx),
            ioName(x.engine, x.outIdx), dtypeName(x.outType), x.outElems, x.outBytes,
            ioFmt(x.engine, x.outIdx));
  }

  // 段の境界でサイズが食い違っていないか (分割点がずれていれば必ずここで露見する)
  for (int s = 0; s + 1 < nStage; s++) {
    if (st[s].outBytes != st[s + 1].inBytes) {
      fprintf(stderr, "[NG] stage%d の出力 %zu B と stage%d の入力 %zu B が一致しない\n",
              s, st[s].outBytes, s + 1, st[s + 1].inBytes);
      return 1;
    }
  }

  const size_t imgBytes = st[0].inBytes;
  std::vector<char> host((size_t)n * imgBytes);
  std::ifstream bf(binPath, std::ios::binary);
  bf.read(host.data(), (std::streamsize)((size_t)n * imgBytes));
  if (!bf) { fprintf(stderr, "inputs read failed (want %d x %zu B)\n", n, imgBytes); return 1; }
  fprintf(stderr, "[chain] %d 枚 x %zu B を読み込み / %d 段\n", n, imgBytes, nStage);

  // 段ごとの出力バッファをあらかじめ確保し、次段の入力としてそのまま渡す
  std::vector<void*> dBuf(nStage + 1, nullptr);
  cudaMalloc(&dBuf[0], imgBytes);
  for (int s = 0; s < nStage; s++) cudaMalloc(&dBuf[s + 1], st[s].outBytes);
  cudaStream_t stream; cudaStreamCreate(&stream);

  const size_t nOut = st[nStage - 1].outElems;
  std::vector<char> outHost(st[nStage - 1].outBytes);

  FILE* fo = fopen(outPath, "w");
  fprintf(fo, "idx");
  for (size_t c = 0; c < nOut; c++) fprintf(fo, ",l%zu", c);
  fprintf(fo, ",argmax\n");

  for (int i = 0; i < n; i++) {
    cudaMemcpyAsync(dBuf[0], host.data() + (size_t)i * imgBytes, imgBytes,
                    cudaMemcpyHostToDevice, stream);
    for (int s = 0; s < nStage; s++) {
#if TRT10_API
      // TRT10 は enqueueV3 の前に全テンソルのアドレスを名前で登録する。
      // 1 つでも未設定だと enqueueV3 が false を返す (静かに壊れることはない)
      if (!st[s].ctx->setTensorAddress(st[s].inIdx, dBuf[s]) ||
          !st[s].ctx->setTensorAddress(st[s].outIdx, dBuf[s + 1])) {
        fprintf(stderr, "[NG] stage%d setTensorAddress 失敗 (img %d)\n", s, i);
        return 1;
      }
      if (!st[s].ctx->enqueueV3(stream)) {
        fprintf(stderr, "[NG] stage%d enqueue 失敗 (img %d)\n", s, i);
        return 1;
      }
#else
      std::vector<void*> b(st[s].nb, nullptr);
      b[st[s].inIdx]  = dBuf[s];
      b[st[s].outIdx] = dBuf[s + 1];
      if (!st[s].ctx->enqueueV2(b.data(), stream, nullptr)) {
        fprintf(stderr, "[NG] stage%d enqueue 失敗 (img %d)\n", s, i);
        return 1;
      }
#endif
    }
    cudaMemcpyAsync(outHost.data(), dBuf[nStage], st[nStage - 1].outBytes,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> v(nOut);
    if (st[nStage - 1].outType == DataType::kHALF) {
      const uint16_t* p = reinterpret_cast<const uint16_t*>(outHost.data());
      for (size_t c = 0; c < nOut; c++) v[c] = halfToFloat(p[c]);
    } else {
      const float* p = reinterpret_cast<const float*>(outHost.data());
      for (size_t c = 0; c < nOut; c++) v[c] = p[c];
    }
    size_t am = 0;
    for (size_t c = 1; c < nOut; c++) if (v[c] > v[am]) am = c;
    fprintf(fo, "%d", i);
    for (size_t c = 0; c < nOut; c++) fprintf(fo, ",%.6f", v[c]);
    fprintf(fo, ",%zu\n", am);
    if (n > 50 && (i % 200) == 0) fprintf(stderr, "  [%d/%d]\n", i, n);
  }
  fclose(fo);
  fprintf(stderr, "[chain] 完了: %d 件 -> %s\n", n, outPath);

  for (int s = 0; s <= nStage; s++) cudaFree(dBuf[s]);
  cudaStreamDestroy(stream);
  for (int s = 0; s < nStage; s++) { delete st[s].ctx; delete st[s].engine; }
  delete runtime;
  return 0;
}
