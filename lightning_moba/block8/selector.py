"""Triton pooled-key routing and a row-centric attention reference.

Selection and anchors use logical blocks of ``block`` tokens. The public block8
interface uses block=8, sink=1 and window=2: the first 8-token block, the query's
current block and its preceding block are forced, with a causal mask applied
when attending to individual tokens. The 128-token length-alignment constraint
in this module does not define the anchor grid.

Mean pooling produces one key per logical block. Selection computes a maximum
over eligible pooled-key scores, then emits a per-row bitmask using the
row maximum plus logalpha, including forced blocks. These internal helpers
accept per-head logalpha; the public block8 API broadcasts a scalar alpha.

The block8 CSC backend reuses pooling and selection from this module. The
row-centric reference instead walks the union of selected blocks in a query
tile and applies each row's mask during attention. Neither is the block128
SM90 pipeline exposed by lightning_moba_varlen_func.
"""
import torch
import triton
import triton.language as tl

SEGMENT = 128
WORD = 32
MIN_DOT = 16
HEAD_DIMS = (32, 64, 128, 256)
# block 1 is token-level selection: the pooled key is the key itself, so the
# same pooled-K threshold rule degenerates to scoring every token. Nothing in
# the selector special-cases it; only the keep bitmask gets 1 bit per token.
BLOCKS = (1, 8, 16, 32, 64, 128)


@triton.jit
def _anchors(positions, cols, BLOCK: tl.constexpr, SEGMENT_SIZE: tl.constexpr,
             PER_SEGMENT: tl.constexpr):
    """Forced and eligible block columns.

    forced   first logical block, current logical block, preceding block
    eligible forced, or a block whose last token is already causal. Partially
             visible blocks are reachable only through `forced`, because a
             pooled key averages tokens the row cannot legally see yet.
    """
    fine = positions[:, None] // BLOCK
    col = cols[None, :]
    # Match the original scalar-alpha CUDA selector: sink=1, window=2,
    # measured in logical selection blocks, independent of execution tiles.
    forced = (col < 1) | ((col <= fine) & (fine - col < 2))
    return forced, forced | ((col + 1) * BLOCK <= positions[:, None])


@triton.jit
def _select_kernel(
    q_ptr, km_ptr, keep_ptr, union_ptr, rowmax_ptr, logalpha_ptr,
    q_row_stride, q_head_stride, km_row_stride, km_head_stride,
    keep_row_stride, keep_head_stride, union_row_stride, union_head_stride,
    logalpha_head_stride,
    seqlen_q, seqlen_k, n_cols, n_words, groups, scale,
    BLOCK: tl.constexpr, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr, PER_SEGMENT: tl.constexpr, WORD_BITS: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    bits = tl.arange(0, WORD_BITS)
    offset = seqlen_k - seqlen_q
    positions = rows + offset
    live = rows < seqlen_q
    q = tl.load(q_ptr + head * q_head_stride + rows[:, None] * q_row_stride + dims[None, :],
                mask=live[:, None], other=0.0)
    km_base = km_ptr + (head // groups) * km_head_stride
    # Only columns a row in this block can reach are ever scored.
    limit = tl.cdiv(tl.minimum(block * BLOCK_M + BLOCK_M + offset, seqlen_k), BLOCK)

    rowmax = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    for word in range(0, tl.cdiv(limit, WORD_BITS)):
        cols = word * WORD_BITS + bits
        inside = cols < limit
        km = tl.load(km_base + cols[:, None] * km_row_stride + dims[None, :],
                     mask=inside[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(km), input_precision="ieee") * scale
        _, eligible = _anchors(positions, cols, BLOCK, SEGMENT_SIZE, PER_SEGMENT)
        ok = eligible & inside[None, :] & live[:, None]
        rowmax = tl.maximum(rowmax, tl.max(tl.where(ok, scores, float("-inf")), 1))

    # One threshold per row, not per head. Without a cap every row of a head
    # carries the same value and this is the old per-head rule exactly; with a
    # cap the rows that overflowed it carry a tightened value instead, so the
    # selection rule itself never needs to know a cap exists.
    threshold = rowmax + tl.load(logalpha_ptr + head * logalpha_head_stride + rows,
                                 mask=live, other=0.0)
    tl.store(rowmax_ptr + head * seqlen_q + rows, rowmax, mask=live)

    for word in range(0, n_words):
        cols = word * WORD_BITS + bits
        inside = cols < limit
        km = tl.load(km_base + cols[:, None] * km_row_stride + dims[None, :],
                     mask=inside[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(km), input_precision="ieee") * scale
        forced, eligible = _anchors(positions, cols, BLOCK, SEGMENT_SIZE, PER_SEGMENT)
        keep = (eligible & inside[None, :] & live[:, None] &
                (forced | (scores >= threshold[:, None])))
        # Bit 31 of an int32 word is negative, so the per-column reduction has to
        # run on the booleans. Reducing the weighted values with tl.max would
        # compare -2147483648 against 0 and silently drop the last column.
        weight = tl.where(keep, (1 << bits)[None, :].to(tl.int32), 0)
        tl.store(keep_ptr + head * keep_head_stride + rows * keep_row_stride + word,
                 tl.sum(weight, 1), mask=live)
        touched = tl.max(keep.to(tl.int32), 0) != 0
        tl.store(union_ptr + block * union_row_stride + head * union_head_stride + word,
                 tl.sum(tl.where(touched, (1 << bits).to(tl.int32), 0)))


@triton.jit
def _block_count_kernel(
    keep_ptr, counts_ptr, keep_row_stride, keep_head_stride,
    seqlen_q, n_cols, n_words,
    ROWS: tl.constexpr, WORD_BITS: tl.constexpr,
):
    """Selected *blocks* per row, straight from the keep bitmask.

    Distinct from calibrate._token_count_kernel, which weights each block by
    how much of it the row can causally see and so answers a token budget. The
    cap is stated in blocks, and a partially visible block costs a full block
    of selection work, so the cap has to count blocks.
    """
    tile = tl.program_id(0)
    head = tl.program_id(1)
    rows = tile * ROWS + tl.arange(0, ROWS)
    live = rows < seqlen_q
    bits = tl.arange(0, WORD_BITS)
    base = keep_ptr + head * keep_head_stride + rows * keep_row_stride
    total = tl.zeros((ROWS,), dtype=tl.int32)
    for word in range(0, n_words):
        packed = tl.load(base + word, mask=live, other=0)
        picked = ((packed[:, None] >> bits[None, :]) & 1) != 0
        cols = word * WORD_BITS + bits
        total += tl.sum(tl.where(picked & (cols < n_cols)[None, :], 1, 0), 1)
    tl.store(counts_ptr + head * seqlen_q + rows, total, mask=live)


@triton.jit
def _cap_histogram_kernel(
    q_ptr, km_ptr, rowmax_ptr, width_ptr, logalpha_ptr, slot_ptr, hist_ptr,
    q_row_stride, q_head_stride, km_row_stride, km_head_stride,
    seqlen_q, seqlen_k, n_cols, groups, scale,
    BLOCK: tl.constexpr, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr, PER_SEGMENT: tl.constexpr, WORD_BITS: tl.constexpr,
    BINS: tl.constexpr,
):
    """Histogram of (pooled score - row maximum) for rows that overflowed the cap.

    Only those rows, and only over their *selected, non-anchor* columns: a
    column below the head's threshold is already rejected and tightening the
    threshold can never bring it back, so the histogram range is exactly
    [logalpha[head], 0] and every column that falls inside it is one the row
    currently keeps. Anchors are excluded because they survive any threshold,
    which is why the cap is spent on them first.

    `slot` compacts the overflowing rows: -1 means the row is inside the cap
    and contributes nothing. A row tile with no overflowing row skips its whole
    column sweep, so the pass costs in proportion to how often the cap bites.
    """
    block = tl.program_id(0)
    head = tl.program_id(1)
    rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
    live = rows < seqlen_q
    slot = tl.load(slot_ptr + head * seqlen_q + rows, mask=live, other=-1)
    active = live & (slot >= 0)
    if tl.sum(active.to(tl.int32)) > 0:
        dims = tl.arange(0, HEAD_DIM)
        bits = tl.arange(0, WORD_BITS)
        offset = seqlen_k - seqlen_q
        positions = rows + offset
        q = tl.load(q_ptr + head * q_head_stride + rows[:, None] * q_row_stride +
                    dims[None, :], mask=active[:, None], other=0.0)
        km_base = km_ptr + (head // groups) * km_head_stride
        limit = tl.cdiv(tl.minimum(block * BLOCK_M + BLOCK_M + offset, seqlen_k), BLOCK)
        # Loaded with other=0.0 rather than -inf: an inactive row would turn
        # every delta into a NaN whose int32 cast is undefined even masked.
        rowmax = tl.load(rowmax_ptr + head * seqlen_q + rows, mask=active, other=0.0)
        low = tl.load(logalpha_ptr + head)
        width = tl.load(width_ptr + head)
        for word in range(0, tl.cdiv(limit, WORD_BITS)):
            cols = word * WORD_BITS + bits
            inside = cols < limit
            km = tl.load(km_base + cols[:, None] * km_row_stride + dims[None, :],
                         mask=inside[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(km), input_precision="ieee") * scale
            forced, eligible = _anchors(positions, cols, BLOCK, SEGMENT_SIZE, PER_SEGMENT)
            delta = scores - rowmax[:, None]
            ok = (active[:, None] & inside[None, :] & eligible & (forced == 0) &
                  (delta >= low))
            # delta - low >= 0 and width > 0, so truncation is a floor. Spelled
            # as a cast to keep the host reference bit-identical without having
            # to match a particular tl.floor lowering.
            index = ((delta - low) / width).to(tl.int32)
            index = tl.minimum(tl.maximum(index, 0), BINS - 1)
            tl.atomic_add(hist_ptr + slot[:, None] * BINS + index, 1, mask=ok)


@triton.jit
def _attend_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, count_ptr, visit_ptr, keep_ptr, union_ptr,
    q_row_stride, q_head_stride, k_row_stride, k_head_stride,
    v_row_stride, v_head_stride, out_row_stride, out_head_stride,
    keep_row_stride, keep_head_stride, union_row_stride, union_head_stride,
    seqlen_q, seqlen_k, n_cols, groups, scale,
    BLOCK: tl.constexpr, BLOCK_M: tl.constexpr, ATTEND_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, WORD_BITS: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    offset = seqlen_k - seqlen_q
    positions = rows + offset
    live = rows < seqlen_q
    q = tl.load(q_ptr + head * q_head_stride + rows[:, None] * q_row_stride + dims[None, :],
                mask=live[:, None], other=0.0)
    kv_head = head // groups
    k_base = k_ptr + kv_head * k_head_stride
    v_base = v_ptr + kv_head * v_head_stride
    keep_base = keep_ptr + head * keep_head_stride + rows * keep_row_stride
    union_base = union_ptr + block * union_row_stride + head * union_head_stride
    limit = tl.minimum(block * BLOCK_M + BLOCK_M + offset, seqlen_k)
    # The attend tile is independent of the selection block: tl.dot needs at
    # least 16 columns, so a block of 8 cannot be a tile on its own. A tile
    # covering several blocks is visited when any of them survives, and the
    # unselected blocks inside it are masked per row, which costs work but
    # keeps the result exact.
    PER_TILE: tl.constexpr = ATTEND_N // BLOCK

    # Online softmax over exact token scores: the pooled row maximum used for
    # selection is not an upper bound on them, so it cannot normalise here.
    running = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    denominator = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    kept = tl.zeros((BLOCK_M,), dtype=tl.int32)
    visited = 0
    for start in range(0, limit, ATTEND_N):
        tile_cols = start // BLOCK + tl.arange(0, PER_TILE)
        reachable = tile_cols < n_cols
        words = tl.load(union_base + tile_cols // WORD_BITS, mask=reachable, other=0)
        touched = reachable & (((words >> (tile_cols % WORD_BITS)) & 1) != 0)
        if tl.sum(touched.to(tl.int32)) > 0:
            visited += 1
            tokens = start + tl.arange(0, ATTEND_N)
            inside = tokens < limit
            k = tl.load(k_base + tokens[:, None] * k_row_stride + dims[None, :],
                        mask=inside[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            # Read the keep mask per token. Reading it per block and widening
            # with broadcast_to/reshape issues fewer loads but forces a layout
            # conversion that measured 1.7x slower, so the redundant loads stay.
            cols = tokens // BLOCK
            row_words = tl.load(keep_base[:, None] + (cols // WORD_BITS)[None, :],
                                mask=live[:, None] & inside[None, :], other=0)
            chosen = ((row_words >> (cols % WORD_BITS)[None, :]) & 1) != 0
            allowed = (chosen & inside[None, :] & live[:, None] &
                       (tokens[None, :] <= positions[:, None]))
            scores = tl.where(allowed, scores, float("-inf"))
            peak = tl.maximum(running, tl.max(scores, 1))
            # A row block can legally contain rows that selected nothing in this
            # tile; keep the shift finite so the exponentials stay defined.
            safe = tl.where(peak == float("-inf"), 0.0, peak)
            rescale = tl.where(running == float("-inf"), 0.0, tl.exp(running - safe))
            weights = tl.where(allowed, tl.exp(scores - safe[:, None]), 0.0)
            v = tl.load(v_base + tokens[:, None] * v_row_stride + dims[None, :],
                        mask=inside[:, None], other=0.0)
            accumulator = accumulator * rescale[:, None]
            accumulator = tl.dot(weights.to(v.dtype), v, accumulator, input_precision="ieee")
            denominator = denominator * rescale + tl.sum(weights, 1)
            kept += tl.sum(allowed.to(tl.int32), 1)
            running = peak

    output = accumulator / tl.where(denominator == 0.0, 1.0, denominator)[:, None]
    tl.store(out_ptr + head * out_head_stride + rows[:, None] * out_row_stride + dims[None, :],
             output.to(out_ptr.dtype.element_ty), mask=live[:, None])
    tl.store(count_ptr + head * seqlen_q + rows, kept, mask=live)
    tl.store(visit_ptr + block * tl.num_programs(1) + head, visited)


def mean_pool_keys(k, block):
    """Mean-pool K over key blocks, matching the reference router's pooling."""
    kv_heads, length, dim = k.shape
    if length % block:
        raise ValueError(f"sequence {length} must be a multiple of block {block}")
    return k.reshape(kv_heads, length // block, block, dim).float().mean(2).to(k.dtype)


# Triton does pointer arithmetic in int32 unless every index is widened, so an
# element offset past 2**31 - 1 wraps to a negative address. The keep bitmask is
# the first tensor to reach that: heads x seqlen_q x ceil(seqlen_k/block/32)
# int32 words, which grows quadratically once block is small.
INDEX_LIMIT = 2**31 - 1


def bitmask_elements(heads, seqlen_q, seqlen_k, block, word=WORD):
    """Element count of the keep bitmask."""
    n_words = -(-(seqlen_k // block) // word)
    return heads * seqlen_q * n_words


def bitmask_last_index(heads, seqlen_q, seqlen_k, block, word=WORD):
    """Largest element index the selector actually forms.

    The bound is on this, not on the element count, and the difference decides
    a real case: block 8 at 128K with 32 heads lands on exactly 2,147,483,647,
    the last representable int32, and runs. Comparing the count instead would
    reject it by one element.
    """
    return bitmask_elements(heads, seqlen_q, seqlen_k, block, word) - 1


def check_bitmask_indexing(heads, seqlen_q, seqlen_k, block, word=WORD):
    """Refuse configurations whose bitmask indices overflow int32.

    Measured rather than assumed: block 1 at 64K with 32 heads addresses
    4,294,903,807 elements, twice the int32 ceiling, and the selector died with
    'illegal memory access'. Block 1 at 32K reaches 1,073,710,079 and runs. A
    wrapped pointer can also hit a valid-but-wrong address, so this converts a
    potentially silent failure into a refusal that names the way under it.
    """
    last = bitmask_last_index(heads, seqlen_q, seqlen_k, block, word)
    if last > INDEX_LIMIT:
        per_row = max(bitmask_elements(heads, 1, seqlen_k, block, word), 1)
        raise ValueError(
            f"keep bitmask addresses index {last:,} for heads={heads}, "
            f"seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, block={block}, over the "
            f"int32 pointer limit of {INDEX_LIMIT:,}. Triton would wrap to a "
            "negative address and fault. Either raise the block size or split "
            f"the queries: at most {max(INDEX_LIMIT // per_row, 1):,} rows per "
            "call at this block size.")
    return last


# Bins for the per-row cap. The cap wants the loosest threshold whose block
# count still fits, which is the (cap)-th largest pooled score on that row: an
# exact per-row selection problem. Solving it exactly would need a sort or a
# second scoring pass per bisection step; a histogram of (score - row maximum)
# answers it in one pass, to within a bin, and interpolating inside the
# straddling bin recovers most of that. 64 bins over the head's own threshold
# range is 1.6% of the range per bin.
CAP_BINS = 64
# A head calibrated to alpha=1 has a zero-wide range; keep the divisor finite.
CAP_MIN_WIDTH = 1e-9


def forced_block_count(seqlen_q, seqlen_k, block, device):
    """Anchor blocks per row, in closed form.

    Materialising the [rows, columns] anchor mask to sum it would be 0.5 GB at
    64K with block 8, so this counts the union of the three anchor regions
    directly. The regions are 128-token segments: segment 0, the row's own
    segment and the one before it, so they overlap only for the first two
    segments, and the row's own segment is the only partially visible one.

    test_kernel.py checks this against reference_anchors rather than trusting
    the case analysis.
    """
    per_segment = SEGMENT // block
    positions = torch.arange(seqlen_k - seqlen_q, seqlen_k, device=device)
    fine = positions // block
    coarse = positions // SEGMENT
    own = fine - coarse * per_segment + 1
    counts = torch.where(
        coarse == 0, fine + 1,
        torch.where(coarse == 1, per_segment + own, 2 * per_segment + own))
    return counts.to(torch.int32)


def cap_bin_width(logalpha, bins=CAP_BINS):
    """Histogram bin width per head, over the range [logalpha[head], 0]."""
    return (-logalpha.to(torch.float32) / bins).clamp(min=CAP_MIN_WIDTH)


def row_logalpha_from_histogram(hist, target, low, width):
    """Per-row threshold offsets that fit `target` non-anchor blocks.

    hist[r, j] counts the row's non-anchor selected columns whose score sits
    `low + j*width` to `low + (j+1)*width` under the row maximum, so the
    suffix sum S[j] is how many survive the threshold `low + j*width`. S is
    non-increasing, and the loosest edge that already fits is

        j = min{ j : S[j] <= target }

    Returns (interpolated, edge). `edge` is that bin edge: it fits by
    construction, but undershoots by up to a bin. `interpolated` assumes the
    straddling bin is uniformly dense and lowers the threshold by
    (target - S[j]) / (S[j-1] - S[j]) of a bin width, which would admit
    exactly the missing blocks if the assumption held. It does not -- density
    rises steeply towards the row maximum -- so interpolation overshot the cap
    by 4% in testing. The caller therefore uses `interpolated` and falls back
    to `edge` on the rows that overshot, which needs the realised counts and
    so cannot be decided here.

    Shared verbatim by the Triton path and the torch reference, so the two
    cannot disagree about the arithmetic -- only about the histogram.
    """
    bins = hist.shape[1]
    suffix = hist.flip(-1).cumsum(-1, dtype=torch.int64).flip(-1)
    fits = suffix <= target[:, None]
    satisfiable = fits.any(-1)
    # argmax on a 0/1 row returns the first 1, which is the smallest fitting j.
    index = torch.where(satisfiable, fits.to(torch.uint8).argmax(-1),
                        torch.full_like(target, bins - 1))
    chosen = suffix.gather(-1, index[:, None]).squeeze(-1)
    # A row over the cap has S[0] > target, so index >= 1 and this is a real
    # previous edge rather than a clamped duplicate.
    previous = suffix.gather(-1, (index - 1).clamp(min=0)[:, None]).squeeze(-1)
    edge = low + index.to(low.dtype) * width
    population = (previous - chosen).clamp(min=1)
    fraction = (target - chosen).clamp(min=0).to(low.dtype) / population.to(low.dtype)
    interpolated = edge - width * fraction
    # Unsatisfiable means even the tightest edge keeps more than the cap, which
    # takes a mass of scores tied at the row maximum. alpha=1 is then the best
    # available and the caller reports the overflow.
    zero = torch.zeros_like(edge)
    edge = torch.where(satisfiable, edge, zero).clamp(max=0.0)
    interpolated = torch.where(satisfiable, interpolated, zero).clamp(max=0.0)
    return interpolated, edge


def select_blocks(q, km, logalpha, block, seqlen_k, block_m=64, warps=4, stages=1,
                  cap=None, cap_bins=CAP_BINS, cap_strict=False, count_rows=64,
                  buffers=None, statistics=False):
    """Run selection, optionally enforcing a per-row block cap.

    Without a cap this is one `_select_kernel` launch and every row of a head
    shares its threshold, exactly as before. With a cap it is three passes:

      select     with the head threshold, to find out who overflows
      histogram  over the overflowing rows only, tile-skipped elsewhere
      select     again, with the tightened per-row thresholds

    so the cap costs one extra scoring pass plus however much the histogram
    pass cannot skip. The cap is never enforced by dropping the lowest-scoring
    blocks of an over-cap row: it raises that row's threshold until the row
    fits, which keeps the selection rule a single comparison against a
    threshold and keeps anchors mandatory.

    Returns a dict; `blocks` and `report` are present only when a cap or
    `statistics` was asked for.
    """
    heads, seqlen_q, dim = q.shape
    n_cols = seqlen_k // block
    n_words = triton.cdiv(n_cols, WORD)
    row_blocks = triton.cdiv(seqlen_q, block_m)
    groups = heads // km.shape[0]
    alpha = logalpha.to(torch.float32).contiguous()
    if cap is not None and (cap < 1 or block_m < MIN_DOT):
        raise ValueError(f"cap must be at least 1 and the cap histogram scores "
                         f"with tl.dot, so block_m must be at least {MIN_DOT}")
    if cap is not None and cap_bins < 2:
        raise ValueError("cap_bins must be at least 2")

    if buffers is None:
        keep = torch.zeros((heads, seqlen_q, n_words), device=q.device, dtype=torch.int32)
        union = torch.zeros((row_blocks, heads, n_words), device=q.device, dtype=torch.int32)
        rowmax = torch.empty((heads, seqlen_q), device=q.device, dtype=torch.float32)
        blocks = torch.zeros((heads, seqlen_q), device=q.device, dtype=torch.int32)
    else:
        keep, union, rowmax, blocks = buffers
    row_alpha = alpha[:, None].expand(heads, seqlen_q).contiguous()

    def run_select():
        _select_kernel[(row_blocks, heads)](
            q, km, keep, union, rowmax, row_alpha,
            q.stride(1), q.stride(0), km.stride(1), km.stride(0),
            keep.stride(1), keep.stride(0), union.stride(0), union.stride(1),
            row_alpha.stride(0),
            seqlen_q, seqlen_k, n_cols, n_words, groups, dim**-0.5,
            BLOCK=block, BLOCK_M=block_m, HEAD_DIM=dim, SEGMENT_SIZE=SEGMENT,
            PER_SEGMENT=SEGMENT // block, WORD_BITS=WORD,
            num_warps=warps, num_stages=stages,
        )

    def count_blocks():
        _block_count_kernel[(triton.cdiv(seqlen_q, count_rows), heads)](
            keep, blocks, keep.stride(1), keep.stride(0),
            seqlen_q, n_cols, n_words,
            ROWS=count_rows, WORD_BITS=WORD, num_warps=warps,
        )
        return blocks

    run_select()
    result = dict(keep=keep, union=union, rowmax=rowmax, row_logalpha=row_alpha)
    if cap is None and not statistics:
        return result

    count_blocks()
    report = dict(cap=cap, mean_blocks=float(blocks.double().mean()),
                  max_blocks=int(blocks.max()))
    if cap is None:
        result.update(blocks=blocks, report=report)
        return result

    over = blocks > cap
    overflowing = int(over.sum())
    report.update(overflowing_rows=overflowing,
                  overflowing_fraction=overflowing / (heads * seqlen_q))
    if overflowing:
        flat = over.reshape(-1)
        slot = torch.full((heads * seqlen_q,), -1, device=q.device, dtype=torch.int32)
        slot[flat] = torch.arange(overflowing, device=q.device, dtype=torch.int32)
        slot = slot.view(heads, seqlen_q)
        width = cap_bin_width(alpha, cap_bins)
        hist = torch.zeros((overflowing, cap_bins), device=q.device, dtype=torch.int32)
        _cap_histogram_kernel[(row_blocks, heads)](
            q, km, rowmax, width, alpha, slot, hist,
            q.stride(1), q.stride(0), km.stride(1), km.stride(0),
            seqlen_q, seqlen_k, n_cols, groups, dim**-0.5,
            BLOCK=block, BLOCK_M=block_m, HEAD_DIM=dim, SEGMENT_SIZE=SEGMENT,
            PER_SEGMENT=SEGMENT // block, WORD_BITS=WORD, BINS=cap_bins,
            num_warps=warps, num_stages=stages,
        )
        # The cap is spent on anchors first: they survive every threshold, so
        # only what is left over can be bought with score.
        anchors = forced_block_count(seqlen_q, seqlen_k, block, q.device)
        target = (cap - anchors[None, :].expand(heads, seqlen_q)[over]).to(torch.int64)
        tightened, safe = row_logalpha_from_histogram(
            hist, target, alpha[:, None].expand(heads, seqlen_q)[over],
            width[:, None].expand(heads, seqlen_q)[over])
        row_alpha = row_alpha.clone()
        row_alpha[over] = safe if cap_strict else tightened
        report["anchor_starved_rows"] = int((target < 0).sum())
        run_select()
        count_blocks()
        report.update(capped_mean_blocks=float(blocks.double().mean()),
                      capped_max_blocks=int(blocks.max()),
                      interpolated_over_cap=int((blocks > cap).sum()))
        # Interpolation assumes a uniform bin and the bins are not uniform, so
        # some rows land over the cap anyway. Those fall back to the bin edge,
        # which fits by construction. A row inside the cap was never retightened
        # and cannot have grown, so the stragglers are a subset of `over` and
        # their slots are still valid.
        straggling = blocks > cap
        if not cap_strict and bool(straggling.any()):
            row_alpha[straggling] = safe[slot[straggling].to(torch.int64)]
            run_select()
            count_blocks()
            report.update(capped_mean_blocks=float(blocks.double().mean()),
                          capped_max_blocks=int(blocks.max()))
        report["still_over_cap"] = int((blocks > cap).sum())
        result["row_logalpha"] = row_alpha
    result.update(blocks=blocks, report=report)
    return result


def block_threshold_attention(q, k, v, logalpha, block=128, block_m=64, km=None,
                              attend_n=None, warps=4, stages=1, statistics=False,
                              cap=None, cap_bins=CAP_BINS, cap_strict=False):
    """Attention over threshold-selected key blocks.

    q/k/v are [heads, seqlen, head_dim]; query heads must be a multiple of kv
    heads. logalpha holds one value per query head, at most zero: a block is
    kept when its pooled score reaches the row's pooled maximum plus logalpha,
    or when it is an anchor. Anchors are the first logical block, the query's
    current block and the preceding block, each spanning `block` tokens;
    individual token visibility is clipped causally.

    `cap` bounds how many blocks any single row may select. A head's alpha is
    calibrated to a mean token budget and says nothing about the tail, so one
    row can select far more than the mean; the cap raises just those rows'
    thresholds until they fit. See select_blocks for what that costs and
    row_logalpha_from_histogram for how the tightened threshold is found. Use
    select_blocks directly to read back how often the cap bit.

    `attend_n` is the attention tile width and defaults to max(block, 16). It
    must be a multiple of `block`, and at least 16 because tl.dot rejects
    narrower operands; a block of 8 therefore shares its tile with neighbours
    and pays for their unselected tokens. Raising it past `block` trades exact
    skipping for MMA efficiency.

    Pass `km` to reuse pooled keys across calls, as an alpha calibration sweep
    should, since pooling is not part of the measured work.

    Returns (output, counts), plus the fraction of attention tiles visited when
    `statistics` is set. counts is the number of key *tokens* each row attended,
    directly comparable to a token budget.
    """
    if q.dim() != 3 or k.dim() != 3 or k.shape != v.shape:
        raise ValueError("expected [heads, seqlen, head_dim] q and matching k/v")
    heads, seqlen_q, dim = q.shape
    kv_heads, seqlen_k, kv_dim = k.shape
    if dim != kv_dim or heads % kv_heads or seqlen_q > seqlen_k:
        raise ValueError("head_dim must match, query heads must be a multiple of kv heads")
    if logalpha.shape != (heads,) or not bool((logalpha <= 0).all()):
        raise ValueError("logalpha must hold one value per query head, each at most zero")
    if dim not in HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {HEAD_DIMS}")
    if block not in BLOCKS or SEGMENT % block:
        raise ValueError(f"block must be one of {BLOCKS}")
    if seqlen_k % block or seqlen_k % SEGMENT:
        raise ValueError(f"key length must be a multiple of {SEGMENT} and of block {block}")
    check_bitmask_indexing(heads, seqlen_q, seqlen_k, block)
    if block_m < 1 or block_m & (block_m - 1):
        raise ValueError("block_m must be a power of two")
    if attend_n is None:
        attend_n = max(block, MIN_DOT)
    if attend_n < MIN_DOT or attend_n % block or attend_n & (attend_n - 1):
        raise ValueError(f"attend_n must be a power of two, at least {MIN_DOT}, "
                         f"and a multiple of block {block}")
    if attend_n > seqlen_k:
        raise ValueError("attend_n cannot exceed the key length")
    if warps not in (1, 2, 4, 8) or stages < 1:
        raise ValueError("warps must be one of 1,2,4,8 and stages positive")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k and v must be contiguous")
    if km is None:
        km = mean_pool_keys(k, block)
    elif km.shape != (kv_heads, seqlen_k // block, dim):
        raise ValueError(f"pooled keys must be {(kv_heads, seqlen_k // block, dim)}")
    km = km.contiguous()

    n_cols = seqlen_k // block
    row_blocks = triton.cdiv(seqlen_q, block_m)
    selection = select_blocks(q, km, logalpha, block, seqlen_k, block_m=block_m,
                              warps=warps, stages=stages, cap=cap,
                              cap_bins=cap_bins, cap_strict=cap_strict)
    keep, union = selection["keep"], selection["union"]
    grid = (row_blocks, heads)
    output = torch.empty_like(q)
    counts = torch.empty((heads, seqlen_q), device=q.device, dtype=torch.int32)
    visits = torch.zeros((row_blocks, heads), device=q.device, dtype=torch.int32)
    _attend_kernel[grid](
        q, k, v, output, counts, visits, keep, union,
        q.stride(1), q.stride(0), k.stride(1), k.stride(0),
        v.stride(1), v.stride(0), output.stride(1), output.stride(0),
        keep.stride(1), keep.stride(0), union.stride(0), union.stride(1),
        seqlen_q, seqlen_k, n_cols, heads // kv_heads, dim**-0.5,
        BLOCK=block, BLOCK_M=block_m, ATTEND_N=attend_n, HEAD_DIM=dim, WORD_BITS=WORD,
        num_warps=warps, num_stages=stages,
    )
    if not statistics:
        return output, counts
    reachable = sum(triton.cdiv(min(seqlen_k, b * block_m + block_m + seqlen_k - seqlen_q),
                                attend_n) for b in range(row_blocks))
    return output, counts, float(visits.sum()) / (reachable * heads)
