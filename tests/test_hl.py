import jesse_extra_indicators as xta

import numpy as np
from collections import deque


def test_higher_highs():
    data = np.array([0, 10, 0, 20, 0])
    res = xta.hl.higher_highs(data, order=1, K=2)
    # 2 consecutives hh detected between index 1 and 3
    assert res == [deque([1, 3])]

    data = np.array([0, 10, 0, 20, 0, 30, 0])
    res = xta.hl.higher_highs(data, order=1, K=2)

    # 2 consecutives hh detected between index 1 and 3 and index 3 and 5
    assert res == [deque([1, 3]), deque([3, 5])]

    # K > 2
    data = np.array([0, 10, 0, 20, 0, 30, 0])
    res = xta.hl.higher_highs(data, order=1, K=3)
    # 3 consecutive hh detected at index 1, 3 and 5
    assert res == [deque([1, 3, 5])]


def test_higher_lows():
    data = np.array([100, 1, 100, 2, 100])
    res = xta.hl.higher_lows(data, order=1, K=2)
    # 2 consecutives hl detected between index 2 and 4
    assert res == [deque([1, 3])]

    data = np.array([100, 1, 100, 2, 100, 3, 100])
    res = xta.hl.higher_lows(data, order=1, K=2)

    # 2 consecutives hh detected between index 2 and 4 and index 4 and 6
    assert res == [deque([1, 3]), deque([3, 5])]

    # K > 2
    data = np.array([100, 1, 100, 2, 100, 3, 100])
    res = xta.hl.higher_lows(data, order=1, K=3)
    # 3 consecutive hh detected at index 1, 3 and 5
    assert res == [deque([1, 3, 5])]


def test_lower_highs():
    data = np.array([0, 10, 0, 9, 0])
    res = xta.hl.lower_highs(data, order=1, K=2)
    # 2 consecutives hh detected between index 1 and 3
    assert res == [deque([1, 3])]

    data = np.array([0, 10, 0, 9, 0, 8, 0])
    res = xta.hl.lower_highs(data, order=1, K=2)

    # 2 consecutives hh detected between index 1 and 3 and index 3 and 5
    assert res == [deque([1, 3]), deque([3, 5])]

    # K > 2
    data = np.array([0, 10, 0, 9, 0, 8, 0])
    res = xta.hl.lower_highs(data, order=1, K=3)
    # 3 consecutive hh detected at index 1, 3 and 5
    assert res == [deque([1, 3, 5])]


def test_lower_lows():
    data = np.array([100, 10, 100, 9, 100])
    res = xta.hl.lower_lows(data, order=1, K=2)
    # 2 consecutives hl detected between index 2 and 4
    assert res == [deque([1, 3])]

    data = np.array([100, 10, 100, 9, 100, 8, 100])
    res = xta.hl.lower_lows(data, order=1, K=2)

    # 2 consecutives hh detected between index 2 and 4 and index 4 and 6
    assert res == [deque([1, 3]), deque([3, 5])]

    # K > 2
    data = np.array([100, 10, 100, 9, 100, 8, 100])
    res = xta.hl.lower_lows(data, order=1, K=3)
    # 3 consecutive hh detected at index 1, 3 and 5
    assert res == [deque([1, 3, 5])]


def test_highs_lows():
    # higher high signals
    data = np.array([0, 10, 0, 20, 0])
    hl = xta.hl.highs_lows(data, order=1)

    assert (hl.higher_highs == np.array([0, 1, 0, 1, 0])).all()
    # everything else is 0
    assert (hl.higher_lows == np.array([0, 0, 0, 0, 0])).all()
    assert (hl.lower_highs == np.array([0, 0, 0, 0, 0])).all()
    assert (hl.lower_lows == np.array([0, 0, 0, 0, 0])).all()

    # higher lows signals
    data = np.array([100, 1, 100, 2, 100])
    hl = xta.hl.highs_lows(data, order=1)

    assert (hl.higher_lows == np.array([0, 1, 0, 1, 0])).all()
    # everything else is 0
    assert (hl.higher_highs == np.array([0, 0, 0, 0, 0])).all()
    assert (hl.lower_highs == np.array([0, 0, 0, 0, 0])).all()
    assert (hl.lower_lows == np.array([0, 0, 0, 0, 0])).all()

    # higher lows signals
    data = np.array([0, 10, 0, 9, 0])
    hl = xta.hl.highs_lows(data, order=1)

    assert (hl.lower_highs == np.array([0, 1, 0, 1, 0])).all()
    # everything else is 0
    assert (hl.higher_highs == np.array([0, 0, 0, 0, 0])).all()
    assert (hl.higher_lows == np.array([0, 0, 0, 0, 0])).all()
    assert (hl.lower_lows == np.array([0, 0, 0, 0, 0])).all()

    # Lower highs signals
    data = np.array([100, 10, 100, 9, 100])
    hl = xta.hl.highs_lows(data, order=1)

    # lower lows signals : 1 means hl have been detected at this index
    assert (hl.lower_lows == np.array([0, 1, 0, 1, 0])).all()
    # everything else is 0
    assert (hl.higher_highs == np.array([0, 0, 0, 0, 0])).all()
    assert (hl.higher_lows == np.array([0, 0, 0, 0, 0])).all()
    assert (hl.lower_highs == np.array([0, 0, 0, 0, 0])).all()
