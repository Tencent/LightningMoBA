"""CSC block8 routing and numerically stable attention.

Selection masks are regrouped into execution columns without changing which
8-token blocks each query selects. Attention first reduces the true maximum
over selected, causally visible token scores across all columns. A second QK
pass uses that maximum for bounded softmax exponentials and atomic FP32 PV /
denominator accumulation. This adds a QK pass but requires only one FP32
maximum per query/head, rather than a full matrix or per-edge D-vector buffer.
The pooled maximum in the pattern belongs to selection, not normalization.
"""
import torch
import triton
import triton.language as tl

from .selector import (BLOCKS, CAP_BINS, HEAD_DIMS, MIN_DOT, SEGMENT, WORD,
                       check_bitmask_indexing, mean_pool_keys, select_blocks)


@triton.jit
def _count_kernel(
    keep_ptr, counts_ptr, keep_row_stride, keep_head_stride, counts_head_stride,
    seqlen_q, seqlen_k, n_cols, n_exec,
    BLOCK: tl.constexpr, WIDTH: tl.constexpr, ROWS: tl.constexpr, WORD_BITS: tl.constexpr,
):
    tl.static_assert(BLOCK == 8 and WIDTH == 256 and WORD_BITS == 32)
    rows = tl.program_id(0) * 256 + tl.arange(0, 256)
    head = tl.program_id(1)
    cols = tl.program_id(2) * 8 + tl.arange(0, 8)
    valid = (rows[:, None] < seqlen_q) & (cols[None, :] < n_exec)
    offset = head.to(tl.int64) * keep_head_stride + rows[:, None].to(tl.int64) * keep_row_stride + cols[None, :]
    words = tl.load(keep_ptr + offset, valid, other=0)
    count = tl.sum((words != 0).to(tl.int32), 0)
    tl.atomic_add(counts_ptr + head * counts_head_stride + cols, count,
                  mask=(cols < n_exec) & (count > 0), sem="relaxed")


@triton.jit
def _scatter_kernel(
    keep_ptr, offsets_ptr, cursor_ptr, indices_ptr,
    keep_row_stride, keep_head_stride, counts_head_stride,
    seqlen_q, seqlen_k, n_cols, n_exec,
    BLOCK: tl.constexpr, WIDTH: tl.constexpr, ROWS: tl.constexpr, WORD_BITS: tl.constexpr,
):
    tl.static_assert(BLOCK == 8 and WIDTH == 256 and WORD_BITS == 32)
    rows = tl.program_id(0) * 256 + tl.arange(0, 256)
    head = tl.program_id(1)
    cols = tl.program_id(2) * 8 + tl.arange(0, 8)
    valid = (rows[:, None] < seqlen_q) & (cols[None, :] < n_exec)
    offset = head.to(tl.int64) * keep_head_stride + rows[:, None].to(tl.int64) * keep_row_stride + cols[None, :]
    wanted = (tl.load(keep_ptr + offset, valid, other=0) != 0) & valid
    total = tl.sum(wanted.to(tl.int32), 0)
    slot = tl.atomic_add(cursor_ptr + head * counts_head_stride + cols, total,
                         mask=(cols < n_exec) & (total > 0), sem="relaxed")
    base = tl.load(offsets_ptr + head * counts_head_stride + cols, cols < n_exec, other=0)
    ranks = tl.cumsum(wanted.to(tl.int32), 0) - 1
    tl.store(indices_ptr + base[None, :] + slot[None, :] + ranks, rows[:, None], wanted)


@triton.jit
def _attend_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, denom_ptr, count_ptr, shift_ptr, keep_ptr,
    offsets_ptr, nnz_ptr, indices_ptr, work_slot_ptr, work_tile_ptr,
    q_row_stride, q_head_stride, k_row_stride, k_head_stride,
    v_row_stride, v_head_stride, out_row_stride, out_head_stride,
    keep_row_stride, keep_head_stride, counts_head_stride,
    seqlen_q, seqlen_k, n_cols, n_exec, groups, scale,
    BLOCK: tl.constexpr, WIDTH: tl.constexpr, ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr, WORD_BITS: tl.constexpr, GROUP: tl.constexpr,
    FIND_MAX: tl.constexpr = False,
):
    """One column group against a tile of the rows that selected it.

    The grid is a flat work list, not (tile, column, head). Anchors make the
    most recent columns selected by every row, so max_rows_per_column reaches
    the full sequence while the mean is far lower; sizing a rectangular grid by
    that maximum launched 839k programs to do 270k tiles of work on B128, and
    97% of them loaded nnz and returned. Worse, the empty fraction grew as the
    tile narrowed, because n_exec scales inversely with WIDTH.

    A program owns GROUP consecutive WIDTH-wide tiles and accumulates across
    them in registers, writing back once. The write-back is why: probe_attend_
    cost.py attributes 62% of the attend stage at width 128 and 80% at width 16
    to the atomic adds, and removing them took block 8 from 45.9 to 120.4
    TFLOPS. Atomic traffic is pairs * HEAD_DIM * 4 bytes, which GROUP divides
    directly, lifting the accumulate from WIDTH/2 to GROUP*WIDTH/2 FLOP/byte.

    Selection semantics are untouched: each inner tile still reads its own keep
    bits and masks per row, so a row only ever sees the `block`-sized blocks it
    chose. Tiles no live row selected are skipped outright.
    """
    item = tl.program_id(0)
    slot = tl.load(work_slot_ptr + item)
    tile = tl.load(work_tile_ptr + item)
    head = slot // n_exec
    column = slot % n_exec
    nnz = tl.load(nnz_ptr + head * counts_head_stride + column)
    slots = tile * ROWS + tl.arange(0, ROWS)
    live = slots < nnz
    start = tl.load(offsets_ptr + head * counts_head_stride + column)
    rows = tl.load(indices_ptr + start + slots, mask=live, other=0)

    dims = tl.arange(0, HEAD_DIM)
    kv_head = head // groups
    positions = rows + (seqlen_k - seqlen_q)
    if FIND_MAX:
        shift = tl.full((ROWS,), float('-inf'), tl.float32)
    else:
        shift = tl.load(shift_ptr + head * seqlen_q + rows, mask=live, other=0.0)
    q = tl.load(q_ptr + head * q_head_stride + rows[:, None] * q_row_stride + dims[None, :],
                mask=live[:, None], other=0.0)

    accumulated = tl.zeros((ROWS, HEAD_DIM), dtype=tl.float32)
    total = tl.zeros((ROWS,), dtype=tl.float32)
    attended = tl.zeros((ROWS,), dtype=tl.int32)

    for inner in range(0, GROUP):
        tokens = column * (GROUP * WIDTH) + inner * WIDTH + tl.arange(0, WIDTH)
        inside = tokens < seqlen_k
        columns = tokens // BLOCK
        words = tl.load(keep_ptr + head * keep_head_stride + rows[:, None] * keep_row_stride +
                        (columns // WORD_BITS)[None, :],
                        mask=live[:, None] & inside[None, :], other=0)
        chosen = ((words >> (columns % WORD_BITS)[None, :]) & 1) != 0
        allowed = (chosen & inside[None, :] & live[:, None] &
                   (tokens[None, :] <= positions[:, None]))
        # Nothing in this tile belongs to any row in the list: skip the loads.
        if tl.sum(allowed.to(tl.int32)) > 0:
            k = tl.load(k_ptr + kv_head * k_head_stride + tokens[:, None] * k_row_stride +
                        dims[None, :], mask=inside[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            if FIND_MAX:
                shift = tl.maximum(shift, tl.max(tl.where(allowed, scores, float('-inf')), 1))
            else:
                # The preceding pass reduced all selected, visible columns.
                # Every nonempty row has a weight of one, without clipping.
                weights = tl.where(allowed, tl.exp(scores - shift[:, None]), 0.0)
                v = tl.load(v_ptr + kv_head * v_head_stride + tokens[:, None] * v_row_stride +
                            dims[None, :], mask=inside[:, None], other=0.0)
                accumulated += tl.dot(weights.to(v.dtype), v, input_precision="ieee")
                total += tl.sum(weights, 1)
                attended += tl.sum(allowed.to(tl.int32), 1)

    if FIND_MAX:
        tl.atomic_max(shift_ptr + head * seqlen_q + rows, shift, mask=live)
    else:
        tl.atomic_add(out_ptr + head * out_head_stride + rows[:, None] * out_row_stride +
                      dims[None, :], accumulated, mask=live[:, None])
        tl.atomic_add(denom_ptr + head * seqlen_q + rows, total, mask=live)
        tl.atomic_add(count_ptr + head * seqlen_q + rows, attended, mask=live)


@triton.jit
def _finalize_kernel(
    out_ptr, denom_ptr, dest_ptr, out_row_stride, out_head_stride,
    dest_row_stride, dest_head_stride, seqlen_q,
    HEAD_DIM: tl.constexpr, ROWS: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    rows = tile * ROWS + tl.arange(0, ROWS)
    live = rows < seqlen_q
    dims = tl.arange(0, HEAD_DIM)
    accumulated = tl.load(out_ptr + head * out_head_stride + rows[:, None] * out_row_stride +
                          dims[None, :], mask=live[:, None], other=0.0)
    denominator = tl.load(denom_ptr + head * seqlen_q + rows, mask=live, other=1.0)
    safe = tl.where(denominator == 0.0, 1.0, denominator)
    tl.store(dest_ptr + head * dest_head_stride + rows[:, None] * dest_row_stride +
             dims[None, :], (accumulated / safe[:, None]).to(dest_ptr.dtype.element_ty),
             mask=live[:, None])


def csc_pattern(q, k, logalpha, block=8, width=128, group=2, km=None,
                select_block_m=64, rows=64, warps=4, stages=1,
                cap=None, cap_bins=CAP_BINS, cap_strict=False,
                statistics=False, selection=None):
    """Select, then build the CSC pattern and the attend work list.

    Columns are `width * group` tokens wide, because that is the unit a single
    attend program owns. count/scatter need no notion of the inner tile: they
    only ask which rows touch a column, so passing them the group width is
    enough.

    Split out from csc_threshold_attention so a profiler can reuse the real
    pattern instead of approximating it: probe_attend_cost.py times variants of
    the attend stage, and any discrepancy between its pattern and this one
    would land in the attribution rather than in the kernel.

    Returns a dict of everything stage 4 consumes.
    """
    if q.dim() != 3 or k.dim() != 3:
        raise ValueError("expected [heads, seqlen, head_dim] q and matching k/v")
    heads, seqlen_q, dim = q.shape
    kv_heads, seqlen_k, kv_dim = k.shape
    if dim != kv_dim or heads % kv_heads or seqlen_q > seqlen_k:
        raise ValueError("head_dim must match, query heads must be a multiple of kv heads")
    if selection is None and (logalpha.shape != (heads,) or not bool((logalpha <= 0).all())):
        raise ValueError("logalpha must hold one value per query head, each at most zero")
    if dim not in HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {HEAD_DIMS}")
    if block not in BLOCKS or SEGMENT % block:
        raise ValueError(f"block must be one of {BLOCKS}")
    if seqlen_k % SEGMENT or seqlen_k % (width * group):
        raise ValueError(f"key length must be a multiple of {SEGMENT} and of the "
                         f"column width {width * group}")
    if width < MIN_DOT or width % block or width & (width - 1):
        raise ValueError(f"width must be a power of two, at least {MIN_DOT}, "
                         f"and a multiple of block {block}")
    if group < 1 or group & (group - 1):
        raise ValueError("group must be a power of two")
    check_bitmask_indexing(heads, seqlen_q, seqlen_k, block)
    if select_block_m < MIN_DOT or select_block_m & (select_block_m - 1):
        raise ValueError(f"select_block_m must be a power of two, at least {MIN_DOT}: "
                         "selection scores Q against pooled K with tl.dot")
    if rows < MIN_DOT or rows & (rows - 1):
        raise ValueError(f"rows must be a power of two, at least {MIN_DOT}")
    if warps not in (1, 2, 4, 8) or stages < 1:
        raise ValueError("warps must be one of 1,2,4,8 and stages positive")
    if not (q.is_contiguous() and k.is_contiguous()):
        raise ValueError("q and k must be contiguous")
    if km is None:
        km = mean_pool_keys(k, block)
    elif km.shape != (kv_heads, seqlen_k // block, dim):
        raise ValueError(f"pooled keys must be {(kv_heads, seqlen_k // block, dim)}")
    km = km.contiguous()

    if block != 8 or width * group != 256:
        raise ValueError("optimized CSC requires block8 and width*group=256")
    n_cols = seqlen_k // block
    column_width = width * group
    n_exec = seqlen_k // column_width
    groups = heads // kv_heads
    scale = dim**-0.5
    alpha = logalpha.to(torch.float32).contiguous() if selection is None else None

    # ---- stage 1: selection, reusing the validated row-centric selector ----
    # select_blocks owns the per-row threshold and the optional block cap, so
    # the CSC path and the row-centric path cannot drift apart on selection.
    # `statistics` makes it count blocks per row even without a cap, which an
    # uncapped control arm needs in order to be comparable; it costs an extra
    # kernel, so it stays off the timed path.
    if selection is None:
        selection = select_blocks(q, km, alpha, block, seqlen_k, block_m=select_block_m,
                                  warps=warps, stages=stages, cap=cap, cap_bins=cap_bins,
                                  cap_strict=cap_strict, statistics=statistics)
    keep, shift = selection["keep"], selection["rowmax"]
    cap_report = selection.get("report")

    # ---- stages 2 and 3: build the CSC pattern ----
    nnz = torch.zeros((heads, n_exec), device=q.device, dtype=torch.int32)
    count_grid = (triton.cdiv(seqlen_q, 256), heads, triton.cdiv(n_exec, 8))
    _count_kernel[count_grid](
        keep, nnz, keep.stride(1), keep.stride(0), nnz.stride(0),
        seqlen_q, seqlen_k, n_cols, n_exec,
        BLOCK=block, WIDTH=column_width, ROWS=rows, WORD_BITS=WORD, num_warps=warps,
    )
    offsets = torch.zeros((heads, n_exec), device=q.device, dtype=torch.int64)
    flat = nnz.to(torch.int64).reshape(-1)
    offsets.reshape(-1)[1:] = flat.cumsum(0)[:-1]
    pairs = int(flat.sum())
    cursor = torch.zeros_like(nnz)
    indices = torch.empty(max(pairs, 1), device=q.device, dtype=torch.int32)
    _scatter_kernel[count_grid](
        keep, offsets, cursor, indices,
        keep.stride(1), keep.stride(0), nnz.stride(0),
        seqlen_q, seqlen_k, n_cols, n_exec,
        BLOCK=block, WIDTH=column_width, ROWS=rows, WORD_BITS=WORD, num_warps=warps,
    )

    # Flatten (column, row tile) into a work list. A rectangular grid has to be
    # sized by the deepest column, and anchors make that the whole sequence, so
    # all but a few percent of its programs would exit immediately.
    per_column = ((nnz + rows - 1) // rows).reshape(-1).to(torch.int64)
    work_items = int(per_column.sum())
    slots_flat = torch.arange(per_column.numel(), device=q.device, dtype=torch.int32)
    work_slot = torch.repeat_interleave(slots_flat, per_column)
    starts = per_column.cumsum(0) - per_column
    work_tile = (torch.arange(work_items, device=q.device, dtype=torch.int64) -
                 torch.repeat_interleave(starts, per_column)).to(torch.int32)
    return dict(keep=keep, shift=shift, nnz=nnz, offsets=offsets, indices=indices,
                work_slot=work_slot, work_tile=work_tile, work_items=work_items,
                pairs=pairs, deepest=int(nnz.max()) if n_exec else 0,
                n_cols=n_cols, n_exec=n_exec, column_width=column_width,
                groups=groups, scale=scale, km=km, cap_report=cap_report)


def csc_threshold_attention(q, k, v, logalpha, block=8, width=128, group=2, km=None,
                            select_block_m=64, rows=64, warps=4, stages=1,
                            statistics=False, cap=None, cap_bins=CAP_BINS,
                            cap_strict=False):
    """Block-sparse threshold attention over a column-centric pattern.

    q/k/v are [heads, seqlen, head_dim]; query heads must be a multiple of kv
    heads. logalpha holds one value per query head, at most zero. A `block`
    sized key block is kept for a row when its pooled score reaches that row's
    pooled maximum plus logalpha, or when it is an anchor (the first logical block and the
    current/preceding logical blocks), clipped causally.

    `cap` bounds the blocks a single row may select, by tightening that row's
    threshold rather than truncating its selection; see kernel.select_blocks.
    With `statistics` the returned dict carries the cap report, which is where
    to read how many rows the cap actually bit.

    `width` is the MMA tile and must be a multiple of `block`, at least 16
    because tl.dot rejects narrower operands. `group` sets how many such tiles
    one program owns, so a CSC column is `width * group` tokens and the atomic
    write-back is amortised over the whole group. Neither affects selection:
    tokens inside a tile that the row did not select are masked, so the result
    matches a pure `block`-granular selector exactly for any width and group.

    Returns (output, counts), plus statistics when `statistics` is set.
    counts is the number of key tokens each row attended.
    """
    if k.shape != v.shape or not v.is_contiguous():
        raise ValueError("expected matching, contiguous k and v")
    heads, seqlen_q, dim = q.shape
    pattern = csc_pattern(q, k, logalpha, block=block, width=width, group=group, km=km,
                          select_block_m=select_block_m, rows=rows, warps=warps,
                          stages=stages, cap=cap, cap_bins=cap_bins,
                          cap_strict=cap_strict, statistics=statistics)
    seqlen_k = k.shape[1]
    n_cols, n_exec = pattern["n_cols"], pattern["n_exec"]
    work_items, pairs, deepest = (pattern["work_items"], pattern["pairs"],
                                  pattern["deepest"])

    # ---- stage 4: true masked maximum, then stable exponentials/PV ----
    accumulator = torch.zeros((heads, seqlen_q, dim), device=q.device, dtype=torch.float32)
    denominator = torch.zeros((heads, seqlen_q), device=q.device, dtype=torch.float32)
    counts = torch.zeros((heads, seqlen_q), device=q.device, dtype=torch.int32)
    keep, nnz = pattern["keep"], pattern["nnz"]
    row_max = torch.full((heads, seqlen_q), float('-inf'), device=q.device, dtype=torch.float32)
    if work_items:
        for find_max in (True, False):
            _attend_kernel[(work_items,)](
                q, k, v, accumulator, denominator, counts, row_max, keep,
                pattern["offsets"], nnz, pattern["indices"],
                pattern["work_slot"], pattern["work_tile"],
                q.stride(1), q.stride(0), k.stride(1), k.stride(0),
                v.stride(1), v.stride(0), accumulator.stride(1), accumulator.stride(0),
                keep.stride(1), keep.stride(0), nnz.stride(0),
                seqlen_q, seqlen_k, n_cols, n_exec, pattern["groups"], pattern["scale"],
                BLOCK=block, WIDTH=width, ROWS=rows, HEAD_DIM=dim, WORD_BITS=WORD,
                GROUP=group, FIND_MAX=find_max, num_warps=warps, num_stages=stages,
                enable_fp_fusion=False,
            )
    output = torch.empty_like(q)
    _finalize_kernel[(triton.cdiv(seqlen_q, rows), heads)](
        accumulator, denominator, output,
        accumulator.stride(1), accumulator.stride(0),
        output.stride(1), output.stride(0), seqlen_q,
        HEAD_DIM=dim, ROWS=rows, num_warps=warps,
    )
    if not statistics:
        return output, counts
    rectangular = max(triton.cdiv(deepest, rows), 1) * n_exec * heads
    # Atomic output traffic scales with pairs, not with FLOPs: every (column,
    # row tile) program writes ROWS x HEAD_DIM floats back once for the whole
    # group. That makes the arithmetic intensity GROUP*WIDTH/2 FLOP/byte.
    return output, counts, dict(csc_pairs=pairs, max_rows_per_column=deepest,
                                mean_rows_per_column=pairs / max(n_exec * heads, 1),
                                indices_gib=pattern["indices"].numel() * 4 / 2**30,
                                atomic_gib=pairs * dim * 4 / 2**30,
                                accumulate_intensity=width * group / 2,
                                column_width=pattern["column_width"],
                                work_items=work_items,
                                rectangular_grid=rectangular,
                                empty_fraction_avoided=1 - work_items / rectangular,
                                cap_report=pattern["cap_report"])
