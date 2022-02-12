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


def test_HighLow():
    w = 2
    # higher high signals
    data = np.array([0, 10, 0, 20, 0])
    hl = xta.hl.HighLow(data, order=1)

    assert hl.has_higher_highs(w)
    assert hl.strict_has_higher_highs(w)

    # everything else is False
    assert not hl.has_higher_lows(w)
    assert not hl.strict_has_higher_lows(w)
    assert not hl.has_lower_highs(w)
    assert not hl.strict_has_lower_highs(w)
    assert not hl.has_lower_lows(w)
    assert not hl.strict_has_lower_lows(w)

    # higher lows signals
    data = np.array([100, 1, 100, 2, 100])
    hl = xta.hl.HighLow(data, order=1)

    assert hl.has_higher_lows(w)
    assert hl.strict_has_higher_lows(w)
    # everything else is False
    assert not hl.has_higher_highs(w)
    assert not hl.strict_has_higher_highs(w)
    assert not hl.has_lower_highs(w)
    assert not hl.strict_has_lower_highs(w)
    assert not hl.has_lower_lows(w)
    assert not hl.strict_has_lower_lows(w)

    # higher lows signals
    data = np.array([0, 10, 0, 9, 0])
    hl = xta.hl.HighLow(data, order=1)

    assert hl.has_lower_highs(w)
    assert hl.strict_has_lower_highs(w)
    # everything else is False
    assert not hl.has_higher_highs(w)
    assert not hl.strict_has_higher_highs(w)
    assert not hl.has_higher_lows(w)
    assert not hl.strict_has_higher_lows(w)
    assert not hl.has_lower_lows(w)
    assert not hl.strict_has_lower_lows(w)

    # Lower highs signals
    data = np.array([100, 10, 100, 9, 100])
    hl = xta.hl.HighLow(data, order=1)

    # lower lows signals : 1 means hl have been detected at this index
    assert hl.has_lower_lows(w)
    assert hl.strict_has_lower_lows(w)
    # everything else is False
    assert not hl.has_higher_highs(w)
    assert not hl.strict_has_higher_highs(w)
    assert not hl.has_higher_lows(w)
    assert not hl.strict_has_higher_lows(w)
    assert not hl.has_lower_highs(w)
    assert not hl.strict_has_lower_highs(w)


def test_IndicatorDivergence_bullish_regular_divergence():
    w = 2
    # higher high signals

    # this indicators has lower lows
    ind1 = np.array([100, 10, 100, 9, 100])
    ind1_hl = xta.hl.HighLow(ind1, order=1)

    # this indicators has higher lows
    ind2 = np.array([100, 1, 100, 2, 100])
    ind2_hl = xta.hl.HighLow(ind2, order=1)

    # build indicator divergence object
    ind_div = xta.hl.IndicatorDivergence(ind1_hl, ind2_hl)
    # check bullish regular divergence is detected
    assert ind_div.bullish_regular_divergence(w)
    assert ind_div.bullish_regular_divergence(w, strict=False)
    # also check top level method
    assert ind_div.regular_divergence(w, "bullish")
    assert ind_div.regular_divergence(w, "bullish", strict=False)


def test_IndicatorDivergence_bearish_regular_divergence():
    w = 2
    # higher high signals

    # this indicators has higher highs
    ind1 = np.array([0, 10, 0, 11, 0])
    ind1_hl = xta.hl.HighLow(ind1, order=1)

    # this indicators has lower highs
    ind2 = np.array([0, 10, 0, 9, 0])
    ind2_hl = xta.hl.HighLow(ind2, order=1)

    # build indicator divergence object
    ind_div = xta.hl.IndicatorDivergence(ind1_hl, ind2_hl)
    # check bearish regulat divergence is detected
    assert ind_div.bearish_regular_divergence(w)
    assert ind_div.bearish_regular_divergence(w, strict=False)

    # also check top level method
    assert ind_div.regular_divergence(w, "bearish")
    assert ind_div.regular_divergence(w, "bearish", strict=False)


def test_IndicatorDivergence_bullish_hidden_divergence():
    w = 2
    # higher high signals

    # this indicators has higher lows
    ind1 = np.array([100, 1, 100, 2, 100])
    ind1_hl = xta.hl.HighLow(ind1, order=1)

    # this indicators has lower lows
    ind2 = np.array([100, 10, 100, 9, 100])
    ind2_hl = xta.hl.HighLow(ind2, order=1)

    # build indicator divergence object
    ind_div = xta.hl.IndicatorDivergence(ind1_hl, ind2_hl)
    # check bearish regulat divergence is detected
    assert ind_div.bullish_hidden_divergence(w)
    assert ind_div.bullish_hidden_divergence(w, strict=False)
    # also check top level method
    assert ind_div.hidden_divergence(w, "bullish")
    assert ind_div.hidden_divergence(w, "bullish", strict=False)


def test_IndicatorDivergence_bearish_hidden_divergence():
    w = 2
    # higher high signals

    # this indicators has lower highs
    ind1 = np.array([0, 10, 0, 9, 0])
    ind1_hl = xta.hl.HighLow(ind1, order=1)

    # this indicators has higher highs
    ind2 = np.array([0, 10, 0, 11, 0])
    ind2_hl = xta.hl.HighLow(ind2, order=1)

    # build indicator divergence object
    ind_div = xta.hl.IndicatorDivergence(ind1_hl, ind2_hl)
    # check bearish regulat divergence is detected
    assert ind_div.bearish_hidden_divergence(w)
    assert ind_div.bearish_hidden_divergence(w, strict=False)

    # also check top level method
    assert ind_div.hidden_divergence(w, "bearish")
    assert ind_div.hidden_divergence(w, "bearish", strict=False)


def test_IndicatorDivergence_bullish_confirmation():
    w = 8
    # higher high signals

    # this indicators has higher high and higher lows
    ind1 = np.array([0, 10, 0, 11, 0, 1, 2, 0, 3])
    ind1_hl = xta.hl.HighLow(ind1, order=1)
    assert ind1_hl.has_higher_highs(w)
    assert ind1_hl.has_higher_lows(w)

    # this indicators has higher high and higher lows
    ind2 = np.array([0, 10, 0, 11, 0, 5, 6, 0, 7])
    ind2_hl = xta.hl.HighLow(ind2, order=1)
    assert ind2_hl.has_higher_highs(w)
    assert ind2_hl.has_higher_lows(w)

    # build indicator divergence object
    ind_div = xta.hl.IndicatorDivergence(ind1_hl, ind2_hl)
    # check bullish regular divergence is detected
    assert ind_div.higher_highs_confirmation(w)
    assert ind_div.higher_lows_confirmation(w)

    # also check top level method
    assert ind_div.bullish_confirmation(w)
    assert ind_div.bullish_confirmation(w, strict=False)
    assert ind_div.confirmation(w, "bullish")


def test_IndicatorDivergence_bearish_confirmation():
    w = 8
    # higher high signals

    # this indicators has lower high and lower lows
    ind1 = np.array([100, 90, 80, 100, 0, 4, 3, 0, 2])
    ind1_hl = xta.hl.HighLow(ind1, order=1)
    assert ind1_hl.has_lower_highs(w)
    assert ind1_hl.has_lower_lows(w)

    # this indicators has lower high and lower lows
    ind2 = np.array([100, 90, 80, 100, 0, 4, 3, 0, 2])
    ind2_hl = xta.hl.HighLow(ind2, order=1)
    assert ind2_hl.has_lower_highs(w)
    assert ind2_hl.has_lower_lows(w)

    # build indicator divergence object
    ind_div = xta.hl.IndicatorDivergence(ind1_hl, ind2_hl)
    # check bullish regular divergence is detected
    assert ind_div.lower_highs_confirmation(w)
    assert ind_div.lower_lows_confirmation(w)

    # also check top level method
    assert ind_div.bearish_confirmation(w)
    assert ind_div.bearish_confirmation(w, strict=False)
    assert ind_div.confirmation(w, "bearish")
