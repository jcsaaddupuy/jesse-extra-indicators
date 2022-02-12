# Indicator divergence library

This module aims to help to find bullish/bearish divergences (regular or hidden) between two indicators using `argrelextrema` from `scipy.signal`.

Code based on [higher-highs-lower-lows-and-calculating-price-trends-in-python](https://medium.com/raposa-technologies/higher-highs-lower-lows-and-calculating-price-trends-in-python-9bc9703f46a1) article on medium.

This package is mean to use in conjunction with [jesse ai](https://github.com/jesse-ai/jesse.git) but is generic enough to be used on its own.

# Install

```bash
poetry build
pip install dist/jesse_extra_indicators-0.1.0-py3-none-any.whl
```

# Example usage

```python
import jesse_extra_indicators as xta

# Loopback window
w = 2

# this indicators has lower highs
ind1 = np.array([0, 10, 0, 9, 0]) # ex df.close
ind1_hl = xta.hl.HighLow(ind1, order=1)

# this indicators has higher highs
ind2 = np.array([0, 10, 0, 11, 0]) # ex ta.rsi(df.close)
ind2_hl = xta.hl.HighLow(ind2, order=1)

# build indicator divergence object
ind_div = xta.hl.IndicatorDivergence(ind1_hl, ind2_hl)

# Check regular or hidden bearish divergence
assert ind_div.regular_divergence(w, "bearish")
assert ind_div.hidden_divergence(w, "bearish")

# Check bearish confirmation 
assert ind_div.confirmation(w, "bearish")


# Check regular or hidden bullish divergence
assert ind_div.regular_divergence(w, "bullish")
assert ind_div.hidden_divergence(w, "bullish")

# Check bearish confirmation 
assert ind_div.confirmation(w, "bullish")
```

# Example jesse strategy

```python
from jesse.strategies import Strategy, cached
import jesse.indicators as ta
import jesse_extra_indicators as xta

class Example(Strategy):

    @property
    @cached
    def close_rsi_div(self):
        # no need to have the full data, this will speed up processing
        w = 100 

        close_hl = xta.hl.HighLow(self.candles[-w:, 2])
        rsi_hl = xta.hl.HighLow(self.rsi)

        # build indicator divergence object
        return xta.hl.IndicatorDivergence(close_hl, close_hl)


    def should_long(self) -> bool:
        w = 2 # you may wan to tweak the loopback window
        return (
            self.close_rsi_div.regular_divergence(w, "bullish")
            or self.close_rsi_div.hidden_divergence(w, "bullish")
        )


    def should_short(self) -> bool:
        w = 2 # you may wan to tweak the loopback window
        return (
            self.close_rsi_div.regular_divergence(w, "bearish")
            or self.close_rsi_div.hidden_divergence(w, "bearish")
        )

    def should_cancel(self) -> bool:
        ...

    def go_long(self):
        ...

    def go_short(self):
        ...
```


Find this usefull and want to buy me a coffee ?

send tips to 

33w68oGuotfpJy59fPP5fUDk2fT3EzGkmS (btc)

D8zxqb2Fzm7Kqkn7QcKrcQjBPwbEBmMbRE (dogecoin)
