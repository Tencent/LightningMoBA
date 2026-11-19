#pragma once

extern "C" int lightning_moba_block8_launch(
    const void* q, const void* k, const void* v, const void* keep,
    void* row_max, const void* offsets, const void* nnz,
    const void* indices, const void* slots, const void* tiles,
    void* out, void* den, void* count,
    int sq, int sk, int heads, int kv_heads, int columns, int words,
    int items, int warps, int device, void* stream);

extern "C" int lightning_moba_block8_probe(void* out, void* stream, int device);
