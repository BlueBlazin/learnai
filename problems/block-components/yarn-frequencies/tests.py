"""Tests for yarn-frequencies.

Numeric cases compare with the oracle; two cases compare with what
`_compute_yarn_parameters` of `transformers` returns for the YaRN parameters of
gpt-oss-20b and DeepSeek-V4 (`fixtures/shipped.npz`, written by `fixtures/make.py`).

Inverse frequencies span seven orders of magnitude, so they are compared with a
relative tolerance and no useful absolute one: with the default `atol` of 1e-6 the slow
half of every table would pass whatever it held.

Tables are compared with an absolute tolerance that grows with the largest position
(`table_tol`). The angle is a product of a position and a frequency, and a solution may
form it in float32 or in float64: the two differ by up to 6e-8 of the angle, which is
1.6e-2 radians at position 2^18.
"""

import math

import numpy as np

from learnai.judge import case, check

FREQ, TABLES = "yarn_inv_freq", "yarn_tables"
FREQ_TOL = {"rtol": 1e-4, "atol": 1e-12}


def table_tol(positions):
    """The tolerance the statement gives for tables at these positions."""
    return {"rtol": 1e-4, "atol": 1e-5 + 2e-7 * float(np.max(positions))}


# (dim, base, factor, original_max_positions, beta_fast, beta_slow) with 0 <= low < high
# and high <= dim / 2 - 1, so the slowest pair is fully rescaled.
SETS = [
    (16, 10000.0, 8.0, 512, 32.0, 1.0),
    (16, 10000.0, 4.0, 2048, 32.0, 1.0),
    (64, 150000.0, 32.0, 4096, 32.0, 1.0),
    (64, 160000.0, 16.0, 65536, 32.0, 1.0),
    (64, 10000.0, 2.0, 8192, 16.0, 2.0),
]


def original(dim, base):
    """The unscaled inverse frequencies, in float64."""
    return base ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)


def bounds(dim, base, max_positions, beta_fast, beta_slow, truncate):
    """The ramp's two ends as the statement defines them."""

    def boundary(rotations):
        return dim * math.log(max_positions / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low, high = boundary(beta_fast), boundary(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, dim - 1)


@case("the worked example: dim 16, base 10000, factor 8, trained on 512 positions", public=True)
def worked_example(ctx):
    got = ctx.check(FREQ, 16, 10000.0, 8.0, 512, **FREQ_TOL)
    ratio = [1.0, 0.78125, 0.5625, 0.34375, 0.125, 0.125, 0.125, 0.125]
    check.allclose(got, original(16, 10000.0) * ratio, **FREQ_TOL, label=FREQ)
    got = ctx.check(FREQ, 16, 10000.0, 8.0, 512, truncate=False, **FREQ_TOL)
    expected = [1.0, 0.29893628, 0.06546508, 0.01151012, 0.00125, 3.952847e-4, 1.25e-4, 3.952847e-5]
    check.allclose(got, np.array(expected), rtol=1e-4, atol=1e-9, label=f"{FREQ}, truncate=False")


# truncate=False: a solution that always floors and ceils the two bounds fails here.
@case("gpt-oss-20b: factor 32 from 4096 positions, base 150000, truncate off", public=True)
def gpt_oss(ctx):
    f = ctx.fixture("shipped.npz")
    got = ctx.check(FREQ, 64, 150000.0, 32.0, 4096, 32.0, 1.0, False, **FREQ_TOL)
    check.allclose(got, f["gpt_inv_freq"], **FREQ_TOL, label=f"{FREQ} versus transformers")


# No truncate key in this configuration: the default, True, applies. A solution that
# never truncates fails here.
@case("DeepSeek-V4: factor 16 from 65536 positions, base 160000, truncate on", public=True)
def deepseek_v4(ctx):
    f = ctx.fixture("shipped.npz")
    got = ctx.check(FREQ, 64, 160000.0, 16.0, 65536, **FREQ_TOL)
    check.allclose(got, f["ds_inv_freq"], **FREQ_TOL, label=f"{FREQ} versus transformers")


@case("the tables of gpt-oss-20b at positions up to 131071", public=True)
def gpt_oss_tables(ctx):
    f = ctx.fixture("shipped.npz")
    tol = table_tol(f["gpt_positions"])
    cos, sin = ctx.check(TABLES, f["gpt_positions"], f["gpt_inv_freq"], 32.0, **tol)
    check.allclose(cos, f["gpt_cos"], **tol, label="cos table versus transformers")
    check.allclose(sin, f["gpt_sin"], **tol, label="sin table versus transformers")


@case("factor = 1 changes nothing: the original frequencies and unscaled tables", public=True)
def factor_one(ctx):
    for dim, base, _, max_positions, beta_fast, beta_slow in SETS[:3]:
        got = ctx.check(FREQ, dim, base, 1.0, max_positions, beta_fast, beta_slow, **FREQ_TOL)
        check.allclose(got, original(dim, base), **FREQ_TOL, label=f"{FREQ} with factor 1")
    inv_freq = original(16, 10000.0).astype(np.float32)
    positions = np.array([0, 1, 5, 300, 4000])
    tol = table_tol(positions)
    cos, sin = ctx.check(TABLES, positions, inv_freq, 1.0, **tol)
    angles = positions[:, None].astype(np.float64) * inv_freq
    check.allclose(cos, np.cos(angles), **tol, label="cos table with factor 1")
    check.allclose(sin, np.sin(angles), **tol, label="sin table with factor 1")


# Rejects ramp_reversed, betas_exchanged, and multiplies_by_factor without the oracle.
@case("fast pairs keep their frequency; slow pairs are divided by the factor exactly")
def both_ends(ctx):
    for dim, base, factor, max_positions, beta_fast, beta_slow in SETS:
        for truncate in (True, False):
            low, high = bounds(dim, base, max_positions, beta_fast, beta_slow, truncate)
            got = ctx.call(FREQ, dim, base, factor, max_positions, beta_fast, beta_slow, truncate)
            check.shape(got, (dim // 2,), label=FREQ)
            index = np.arange(dim // 2)
            fast, slow = index <= low, index >= high
            plain = original(dim, base)
            check.allclose(
                got[fast], plain[fast], rtol=1e-5, atol=0.0, label=f"{FREQ}, pairs below the ramp"
            )
            check.allclose(
                got[slow],
                plain[slow] / factor,
                rtol=1e-5,
                atol=0.0,
                label=f"{FREQ}, pairs above the ramp",
            )


@case("the ratio to the original frequencies never rises along the head")
def ratio_is_monotone(ctx):
    for dim, base, factor, max_positions, beta_fast, beta_slow in SETS:
        got = ctx.call(FREQ, dim, base, factor, max_positions, beta_fast, beta_slow, False)
        ratio = got / original(dim, base)
        check.that(
            bool(np.all(np.diff(ratio) <= 1e-5)),
            f"{FREQ}: the scaled frequency divided by the original one rises somewhere",
            f"dim={dim}, factor={factor:g}\nratio = {np.round(ratio, 5).tolist()}",
        )
        check.allclose(ratio[[0, -1]], [1.0, 1.0 / factor], rtol=1e-4, label="ratio at both ends")


# Rejects never_truncates and always_truncates on parameters of the test's own.
@case("truncate on and off for the same parameters")
def truncate_both_ways(ctx):
    for dim, base, factor, max_positions, beta_fast, beta_slow in SETS[:2]:
        for truncate in (True, False):
            args = (dim, base, factor, max_positions, beta_fast, beta_slow, truncate)
            ctx.check(FREQ, *args, **FREQ_TOL)


# Rejects betas_exchanged and missing_two_pi where the defaults are not in play.
@case("other rotation counts: beta_fast 16 and beta_slow 2, then 8 and 0.5")
def other_betas(ctx):
    for truncate in (True, False):
        ctx.check(FREQ, 64, 10000.0, 2.0, 8192, 16.0, 2.0, truncate, **FREQ_TOL)
        ctx.check(FREQ, 64, 10000.0, 4.0, 8192, beta_fast=8.0, beta_slow=0.5, **FREQ_TOL)


# The lower bound is negative before clamping. Rejects no_lower_clamp.
@case("a short original context: every pair but the first few is on or past the ramp")
def low_clamped_to_zero(ctx):
    for truncate in (True, False):
        ctx.check(FREQ, 16, 10000.0, 4.0, 64, truncate=truncate, **FREQ_TOL)
        ctx.check(FREQ, 64, 150000.0, 8.0, 128, truncate=truncate, **FREQ_TOL)


# The upper bound lies past the last pair, dim / 2 - 1, and is clamped only to dim - 1,
# so the slowest pair is not fully rescaled. Rejects clamps_high_to_last_pair.
@case("a small base and a long original context: the ramp ends past the last pair")
def high_past_last_pair(ctx):
    for truncate in (True, False):
        got = ctx.check(FREQ, 16, 100.0, 8.0, 4096, truncate=truncate, **FREQ_TOL)
        ratio = got[-1] / original(16, 100.0)[-1]
        check.that(
            ratio > 0.5,
            f"{FREQ}: the slowest pair was rescaled further than the clamped ramp allows",
            f"ratio of the slowest pair to its original frequency: {ratio:.4f}",
        )
    ctx.check(FREQ, 64, 150000.0, 32.0, 2**22, truncate=False, **FREQ_TOL)


# With beta_fast = 20000 the ramp starts at pair 6.09 and its unclamped end would be
# 23.3, beyond dim - 1 = 15: the clamp changes the last pair (ramp 0.102 instead of
# 0.053 without truncation). Expected values from transformers. Rejects no_upper_clamp.
@case("a ramp whose upper end is clamped to dim - 1")
def upper_clamp_binds(ctx):
    f = ctx.fixture("shipped.npz")
    for truncate, key in ((False, "clamp_inv_freq"), (True, "clamp_inv_freq_truncated")):
        got = ctx.check(FREQ, 16, 100.0, 8.0, 4194304, 20000.0, 1.0, truncate, **FREQ_TOL)
        check.allclose(got, f[key], **FREQ_TOL, label=f"{FREQ} versus transformers")


# cos^2 + sin^2 is the square of the table factor. Rejects factor_on_cos_only and
# log10_table_factor without the oracle.
@case("tables at random positions: cos^2 + sin^2 is the square of the table factor")
def table_norm(ctx):
    inv_freq = original(16, 10000.0).astype(np.float32)
    for factor in (2.0, 8.0, 32.0):
        positions = ctx.randint(0, 5000, (5,))
        cos, sin = ctx.check(TABLES, positions, inv_freq, factor, **table_tol(positions))
        check.shape(cos, (5, 8), label="cos table")
        scale = 0.1 * math.log(factor) + 1.0
        check.allclose(
            cos**2 + sin**2,
            np.full((5, 8), scale**2),
            rtol=1e-4,
            label="cos^2 + sin^2 of the tables",
        )


@case("tables from rescaled frequencies, a single position, and repeated positions")
def tables_from_yarn(ctx):
    f = ctx.fixture("shipped.npz")
    for positions in (np.array([70000]), np.array([9, 0, 9, 2, 131071, 5])):
        ctx.check(TABLES, positions, f["ds_inv_freq"], 16.0, **table_tol(positions))
    # Small positions: the tight end of the tolerance.
    positions = np.array([0, 1, 2, 3, 17, 63])
    ctx.check(TABLES, positions, f["gpt_inv_freq"], 32.0, **table_tol(positions))
