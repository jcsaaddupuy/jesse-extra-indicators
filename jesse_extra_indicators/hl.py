# implementation based on
# https://medium.com/raposa-technologies/higher-highs-lower-lows-and-calculating-price-trends-in-python-9bc9703f46a1
import numpy as np
from scipy.signal import argrelextrema
from collections import deque

try:
    from numba import jit
except ImportError:
    jit = lambda f: f


@jit
def higher_lows(data: np.array, order: int = 5, K: int = 2):
    """
    Finds consecutive higher lows in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive lows need to be higher.
    """
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    # Ensure consecutive lows are higher than previous lows
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] < lows[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())
    return extrema


@jit
def lower_highs(data: np.array, order: int = 5, K: int = 2):
    """
    Finds consecutive lower highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be lower.
    """
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are lower than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] > highs[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema


@jit
def higher_highs(data: np.array, order: int = 5, K: int = 2):
    """
    Finds consecutive higher highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be higher.
    """
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are higher than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] < highs[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())
    return extrema


@jit
def lower_lows(data: np.array, order: int = 5, K: int = 2):
    """
    Finds consecutive lower lows in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive lows need to be lower.
    """
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    # Ensure consecutive lows are lower than previous lows
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] > lows[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())
    return extrema


from collections import namedtuple

RawHighsLows = namedtuple(
    "RawHighsLows", ["higher_highs", "lower_highs", "higher_lows", "lower_lows"]
)


def highs_lows(data: np.array, order: int = 5, K: int = 2):
    """
    hl = highs_lows(self.candles[:, 2])
    rsi_hl = highs_lows(ta.rsi(self.candles))
    window = 2
    # bullish divergence
    regular_divergence = hl.lower_lows[-window:].any() and rsi_hl.higher_lows[-window:].any()
    hidden_divergence = hl.higher_lows[-window:].any() and rsi_hl.lower_lows[-window:].any()
    # bearish divergence
    regular_divergence = hl.higher_highs[-window:].any() and rsi_hl.lower_highs[-window:].any()
    hidden_divergence = hl.lower_highs[-window:].any() and rsi_hl.higher_highs[-window:].any()
    """

    hh = higher_highs(data, order=order, K=K)
    hh_sig = np.zeros(len(data), dtype=int)
    for s in hh:
        hh_sig[s] = 1

    hl = higher_lows(data, order=order, K=K)
    hl_sig = np.zeros(len(data), dtype=int)
    for s in hl:
        hl_sig[s] = 1

    lh = lower_highs(data, order=order, K=K)
    lh_sig = np.zeros(len(data), dtype=int)
    for s in lh:
        lh_sig[s] = 1

    ll = lower_lows(data, order=order, K=K)
    ll_sig = np.zeros(len(data), dtype=int)
    for s in ll:
        ll_sig[s] = 1

    return RawHighsLows(hh_sig, lh_sig, hl_sig, ll_sig)


class HighLow:
    def __init__(self, hl: RawHighsLows):
        self.hl = hl

    @jit
    def has_lower_lows(self, window: int) -> bool:
        return self.hl.lower_lows[-window:].any()

    @jit
    def has_lower_highs(self, window: int) -> bool:
        return self.hl.lower_highs[-window:].any()

    @jit
    def has_higher_lows(self, window: int) -> bool:
        return self.hl.higher_lows[-window:].any()

    @jit
    def has_higher_highs(self, window: int) -> bool:
        return self.hl.higher_highs[-window:].any()

    def strict_has_lower_lows(self, window: int) -> bool:
        return self.has_lower_lows(window) and not (
            self.has_lower_highs(window)
            or self.has_higher_lows(window)
            or self.has_higher_highs(window)
        )

    def strict_has_lower_highs(self, window: int) -> bool:
        return self.has_lower_highs(window) and not (
            self.has_lower_lows(window)
            or self.has_higher_lows(window)
            or self.has_higher_highs(window)
        )

    def strict_has_higher_lows(self, window: int) -> bool:
        return self.has_higher_lows(window) and not (
            self.has_lower_lows(window)
            or self.has_lower_highs(window)
            or self.has_higher_highs(window)
        )

    def strict_has_higher_highs(self, window: int) -> bool:
        return self.has_higher_highs(window) and not (
            self.has_lower_lows(window)
            or self.has_lower_highs(window)
            or self.has_higher_lows(window)
        )


def highs_lows_divergence(data, order: int = 5, K: int = 2) -> HighLow:
    return HighLow(highs_lows(data, order, K))


class IndicatorDivergence:
    def __init__(self, hld1: HighLow, hld2: HighLow):
        self.hld1 = hld1
        self.hld2 = hld2

    def bullish_regular_divergence(self, window: int, strict: bool = True) -> bool:
        if strict:
            return self.hld1.strict_has_lower_lows(
                window
            ) and self.hld2.strict_has_higher_lows(window)
        return self.hld1.has_lower_lows(window) and self.hld2.has_higher_lows(window)

    def bearish_regular_divergence(self, window: int, strict: bool = True) -> bool:
        if strict:
            return self.hld1.strict_has_higher_highs(
                window
            ) and self.hld2.strict_has_lower_highs(window)
        return self.hld1.has_higher_highs(window) and self.hld2.has_lower_highs(window)

    def bullish_hidden_divergence(self, window: int, strict: bool = True) -> bool:
        if strict:
            return self.hld1.strict_has_higher_lows(
                window
            ) and self.hld2.strict_has_lower_lows(window)
        return self.hld1.has_higher_lows(window) and self.hld2.has_lower_lows(window)

    def bearish_hidden_divergence(self, window: int, strict: bool = True) -> bool:
        if strict:
            return self.hld1.strict_has_lower_highs(
                window
            ) and self.hld2.strict_has_higher_highs(window)
        return self.hld1.has_lower_highs(window) and self.hld2.has_higher_highs(window)

    def regular_divergence(self, window: int, mode, strict: bool = True) -> bool:
        if mode == "bullish":
            return self.bullish_regular_divergence(window, strict)
        if mode == "bearish":
            return self.bearish_regular_divergence(window, strict)
        raise ValueError(f"Unknown mode '{mode}'")

    def hidden_divergence(self, window: int, mode, strict: bool = True) -> bool:
        if mode == "bullish":
            return self.bullish_hidden_divergence(window, strict)
        if mode == "bearish":
            return self.bearish_hidden_divergence(window, strict)
        raise ValueError(f"Unknown mode '{mode}'")

    def higher_highs_confirmation(self, window: int):
        return self.hld1.has_higher_highs(window) and self.hld2.has_higher_highs(window)

    def higher_lows_confirmation(self, window: int):
        return self.hld1.has_higher_lows(window) and self.hld2.has_higher_lows(window)

    def lower_highs_confirmation(self, window: int):
        return self.hld1.has_lower_highs(window) and self.hld2.has_lower_highs(window)

    def lower_lows_confirmation(self, window: int):
        return self.hld1.has_lower_lows(window) and self.hld2.has_lower_lows(window)

    def bullish_confirmation(self, window: int, strict: bool = True) -> bool:
        if strict:
            return self.higher_highs_confirmation(
                window
            ) and self.higher_lows_confirmation(window)
        return self.higher_highs_confirmation(window) or self.higher_lows_confirmation(
            window
        )

    def bearish_confirmation(self, window: int, strict: bool = True) -> bool:
        if strict:
            return self.lower_highs_confirmation(
                window
            ) and self.lower_lows_confirmation(window)
        else:
            return self.lower_highs_confirmation(
                window
            ) or self.lower_lows_confirmation(window)

    def confirmation(self, window: int, mode, strict: bool = True) -> bool:
        if mode == "bullish":
            return self.bullish_confirmation(window, strict)
        if mode == "bearish":
            return self.bearish_confirmation(window, strict)
        raise ValueError(f"Unknown mode '{mode}'")


def indicators_divergence(
    data1: np.array, data2: np.array, order: int = 5, K: int = 2
) -> IndicatorDivergence:
    return IndicatorDivergence(
        HighLow(highs_lows(data1, order, K)),
        HighLow(highs_lows(data2, order, K)),
    )
