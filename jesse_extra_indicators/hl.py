# implementation based on
# https://medium.com/raposa-technologies/higher-highs-lower-lows-and-calculating-price-trends-in-python-9bc9703f46a1
import typing as t
import numpy as np
from scipy.signal import argrelextrema
from collections import deque

try:
    from numba import jit
except ImportError:
    jit = lambda f: f


@jit
def higher_lows(data: np.array, order: int = 5, K: int = 2) -> t.List[deque]:
    """Finds consecutive higher lows in time series like list of values.

    Args:
        data (np.array): np.array of data points values.
        order (int, optional): argrelextrema order. Defaults to 5.
        K (int, optional): How many consecutive lows need to be higher. Defaults to 2.

    Returns:
        t.List[deque]: List of deque containing lower lows indexes.
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
def lower_highs(data: np.array, order: int = 5, K: int = 2) -> t.List[deque]:
    """Finds consecutive lower highs in time series like list of values.

    Args:
        data (np.array): np.array of data points values.
        order (int, optional): argrelextrema order. Defaults to 5.
        K (int, optional): How many consecutive highs need to be lower. Defaults to 2.

    Returns:
        t.List[deque]: List of deque containing lower lows indexes.
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
def higher_highs(data: np.array, order: int = 5, K: int = 2) -> t.List[deque]:
    """Finds consecutive higher highs in time series like list of values.

    Args:
        data (np.array): np.array of data points values.
        order (int, optional): argrelextrema order. Defaults to 5.
        K (int, optional): How many consecutive highs need to be higher. Defaults to 2.

    Returns:
        t.List[deque]: List of deque containing lower lows indexes.
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
def lower_lows(data: np.array, order: int = 5, K: int = 2) -> t.List[deque]:
    """Finds consecutive lower lows in time series like list of values.

    Args:
        data (np.array): np.array of data points values.
        order (int, optional): argrelextrema order. Defaults to 5.
        K (int, optional): How many consecutive low need to be lower. Defaults to 2.

    Returns:
        t.List[deque]: List of deque containing lower lows indexes.
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


def highs_lows(data: np.array, order: int = 5, K: int = 2) -> RawHighsLows:
    """Finds consecutive higher highs / higher lows / lower highs / lower lows in time series like list of values.

    Args:
        data (np.array): np.array of data points values.
        order (int, optional): argrelextrema order. Defaults to 5.
        K (int, optional): How many consecutive highs need to be higher. Defaults to 2.

    Returns:
        RawHighsLows: Named tuple containing hh/hl/ll/lh indexes in higher_highs, lower_highs, higher_lows, lower_lows attributes.
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
    def __init__(self, data: np.array, order=5, K=2):
        """Initialize a new HighLow object.

        Args:
            data (np.array): np.array of data points values.
            order (int, optional): argrelextrema order. Defaults to 5.
            K (int, optional): How many consecutive highs need to be higher. Defaults to 2.
        """
        self.hl = highs_lows(data, order, K)

    @jit
    def has_lower_lows(self, window: int) -> bool:
        """Detect lower lows.

        Args:
            window (int): Loopback window.

        Returns:
            bool: True if data has lower lows in the given window, False otherwise.
        """
        return self.hl.lower_lows[-window:].any()

    @jit
    def has_lower_highs(self, window: int) -> bool:
        """Detect lower highs.

        Args:
            window (int): Loopback window.

        Returns:
            bool: True if data has lower highs in the given window, False otherwise.
        """
        return self.hl.lower_highs[-window:].any()

    @jit
    def has_higher_lows(self, window: int) -> bool:
        """Detect higher lows.

        Args:
            window (int): Loopback window.

        Returns:
            bool: True if data has higher lows in the given window, False otherwise.
        """
        return self.hl.higher_lows[-window:].any()

    @jit
    def has_higher_highs(self, window: int) -> bool:
        """Detect higher highs.

        Args:
            window (int): Loopback window.

        Returns:
            bool: True if data has higher highs in the given window, False otherwise.
        """
        return self.hl.higher_highs[-window:].any()

    def strict_has_lower_lows(self, window: int) -> bool:
        """Detect lower lows in strict mode.

        Args:
            window (int): Loopback window.

        Returns:
            bool: True if data has *only* lower lows in the given window, False otherwise.
        """
        return self.has_lower_lows(window) and not (
            self.has_lower_highs(window)
            or self.has_higher_lows(window)
            or self.has_higher_highs(window)
        )

    def strict_has_lower_highs(self, window: int) -> bool:
        """Detect lower highs in strict mode.

        Args:
            window (int): Loopback window.

        Returns:
            bool: True if data has *only* lower highs in the given window, False otherwise.
        """
        return self.has_lower_highs(window) and not (
            self.has_lower_lows(window)
            or self.has_higher_lows(window)
            or self.has_higher_highs(window)
        )

    def strict_has_higher_lows(self, window: int) -> bool:
        """Detect higher lows in strict mode.

        Args:
            window (int): Loopback window.

        Returns:
            bool: True if data has *only* higher lows in the given window, False otherwise.
        """
        return self.has_higher_lows(window) and not (
            self.has_lower_lows(window)
            or self.has_lower_highs(window)
            or self.has_higher_highs(window)
        )

    def strict_has_higher_highs(self, window: int) -> bool:
        """Detect higher highs in strict mode.

        Args:
            window (int): Loopback window.

        Returns:
            bool: True if data has *only* higher highs in the given window, False otherwise.
        """
        return self.has_higher_highs(window) and not (
            self.has_lower_lows(window)
            or self.has_lower_highs(window)
            or self.has_higher_lows(window)
        )


class IndicatorDivergence:
    def __init__(self, hld1: HighLow, hld2: HighLow):
        """Initialize a new IndicatorDivergence. This objects helps to find divergences or confirmation between two indicators.

        Args:
            hld1 (HighLow): indicator 1 HighLow object
            hld2 (HighLow): indicator 2 HighLow object
        """
        self.hld1 = hld1
        self.hld2 = hld2

    def bullish_regular_divergence(self, window: int, strict: bool = True) -> bool:
        """Detect bullish regular divergence.

        Args:
            window (int): Loopback window.
            strict (bool, optional): If True, will use the strict version on higher highs/lower low methods. Defaults to True.

        Returns:
            bool: True if indicator 1 has lower low and indicator 2 has higher lows, False otherwise.
        """
        if strict:
            return self.hld1.strict_has_lower_lows(
                window
            ) and self.hld2.strict_has_higher_lows(window)
        return self.hld1.has_lower_lows(window) and self.hld2.has_higher_lows(window)

    def bearish_regular_divergence(self, window: int, strict: bool = True) -> bool:
        """Detect bearish regular divergence.

        Args:
            window (int): Loopback window.
            strict (bool, optional): If True, will use the strict version on higher highs/lower low methods. Defaults to True.

        Returns:
            bool: True if indicator 1 has higher highs and indicator 2 has lower highs, False otherwise.
        """
        if strict:
            return self.hld1.strict_has_higher_highs(
                window
            ) and self.hld2.strict_has_lower_highs(window)
        return self.hld1.has_higher_highs(window) and self.hld2.has_lower_highs(window)

    def bullish_hidden_divergence(self, window: int, strict: bool = True) -> bool:
        """Detect bullish hidden divergence.

        Args:
            window (int): Loopback window.
            strict (bool, optional): If True, will use the strict version on higher highs/lower low methods. Defaults to True.

        Returns:
            bool: True if indicator 1 has higher lows and indicator 2 has lower lows, False otherwise.
        """
        if strict:
            return self.hld1.strict_has_higher_lows(
                window
            ) and self.hld2.strict_has_lower_lows(window)
        return self.hld1.has_higher_lows(window) and self.hld2.has_lower_lows(window)

    def bearish_hidden_divergence(self, window: int, strict: bool = True) -> bool:
        """Detect bearish hidden divergence.

        Args:
            window (int): Loopback window.
            strict (bool, optional): If True, will use the strict version on higher highs/lower low methods. Defaults to True.

        Returns:
            bool: True if indicator 1 has lower highs and indicator 2 has higher highs, False otherwise.
        """
        if strict:
            return self.hld1.strict_has_lower_highs(
                window
            ) and self.hld2.strict_has_higher_highs(window)
        return self.hld1.has_lower_highs(window) and self.hld2.has_higher_highs(window)

    def regular_divergence(self, window: int, mode, strict: bool = True) -> bool:
        """Detect a regular divergence.

        Args:
            window (int): Loopback window.
            mode ([type]): "bullish"/"bearish"
            strict (bool, optional): If True, will use the strict version on higher highs/lower low methods. Defaults to True.

        Raises:
            ValueError: An unknown value have been given to the parameter mode.

        Returns:
            bool: True if a regular divergence have been detected, False otherwise.
        """
        if mode == "bullish":
            return self.bullish_regular_divergence(window, strict)
        if mode == "bearish":
            return self.bearish_regular_divergence(window, strict)
        raise ValueError(f"Unknown mode '{mode}'")

    def hidden_divergence(self, window: int, mode, strict: bool = True) -> bool:
        """Detect a hidden divergence.

        Args:
            window (int): Loopback window.
            mode ([type]): "bullish"/"bearish"
            strict (bool, optional): If True, will use the strict version on higher highs/lower low methods. Defaults to True.

        Raises:
            ValueError: An unknown value have been given to the parameter mode.

        Returns:
            bool: True if a hidden divergence have been detected, False otherwise.
        """
        if mode == "bullish":
            return self.bullish_hidden_divergence(window, strict)
        if mode == "bearish":
            return self.bearish_hidden_divergence(window, strict)
        raise ValueError(f"Unknown mode '{mode}'")

    def higher_highs_confirmation(self, window: int) -> True:
        """Detect higher highs confirmations (both indicators agree).

        Args:
            window (int): Loopback window.

        Returns:
            True: True if a higher highs confirmation have been detected, False otherwise.
        """
        return self.hld1.has_higher_highs(window) and self.hld2.has_higher_highs(window)

    def higher_lows_confirmation(self, window: int) -> bool:
        """Detect higher lows confirmations (both indicators agree).

        Args:
            window (int): Loopback window.

        Returns:
            True: True if a higher lows confirmation have been detected, False otherwise.
        """
        return self.hld1.has_higher_lows(window) and self.hld2.has_higher_lows(window)

    def lower_highs_confirmation(self, window: int) -> bool:
        """Detect lower highs confirmations (both indicators agree).

        Args:
            window (int): Loopback window.

        Returns:
            True: True if a lower highs confirmation have been detected, False otherwise.
        """
        return self.hld1.has_lower_highs(window) and self.hld2.has_lower_highs(window)

    def lower_lows_confirmation(self, window: int):
        """Detect lower lows confirmations (both indicators agree).

        Args:
            window (int): Loopback window.

        Returns:
            True: True if a lower lows confirmation have been detected, False otherwise.
        """
        return self.hld1.has_lower_lows(window) and self.hld2.has_lower_lows(window)

    def bullish_confirmation(self, window: int, strict: bool = True) -> bool:
        """Detect a bullish confirmation.

        Args:
            window (int): Loopback window.
            strict (bool, optional): If True, will check that indicators have both higher highs and higher lows confirmation.
            If False, this function will check only one of these conditions.

        Returns:
            bool: True if bullish confirmation have been detected.
        """
        if strict:
            return self.higher_highs_confirmation(
                window
            ) and self.higher_lows_confirmation(window)
        return self.higher_highs_confirmation(window) or self.higher_lows_confirmation(
            window
        )

    def bearish_confirmation(self, window: int, strict: bool = True) -> bool:
        """Detect a bearish confirmation.

        Args:
            window (int): Loopback window.
            strict (bool, optional): If True, will check that indicators have both lower highs and lower lows confirmation.
            If False, this function will check only one of these conditions.

        Returns:
            bool: True if bullish confirmation have been detected.
        """
        if strict:
            return self.lower_highs_confirmation(
                window
            ) and self.lower_lows_confirmation(window)
        else:
            return self.lower_highs_confirmation(
                window
            ) or self.lower_lows_confirmation(window)

    def confirmation(self, window: int, mode, strict: bool = True) -> bool:
        """Detect indicators confirmation.

        Args:
            window (int): Loopback window.
            mode ([type]): "bullish"/"bearish"
            strict (bool, optional): If True, will detect in strict mode.

        Raises:
            ValueError: An unknown value have been given to the parameter mode.

        Returns:
            bool: True if a bullish/bearish confirmation have been detected, False otherwise.
        """
        if mode == "bullish":
            return self.bullish_confirmation(window, strict)
        if mode == "bearish":
            return self.bearish_confirmation(window, strict)
        raise ValueError(f"Unknown mode '{mode}'")


def indicators_divergence(
    data1: np.array, data2: np.array, order: int = 5, K: int = 2
) -> IndicatorDivergence:
    """Helper method to instanciate IndicatorDivergence from raw numpy arrays.

    Args:
        data1 (np.array): Data for indicator 1.
        data2 (np.array): Data for indicator 2.
        order (int, optional): argrelextrema order. Defaults to 5.
        K (int, optional): How many consecutive highs need to be higher. Defaults to 2.

    Returns:
        IndicatorDivergence: Indicator Divergence object.
    """
    return IndicatorDivergence(
        HighLow(data1, order, K),
        HighLow(data2, order, K),
    )
