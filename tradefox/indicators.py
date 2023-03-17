import pandas as pd
import numpy as np
from typing import Tuple


def simple_moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA).

    Args:
        data (pd.Series): Price data.
        window (int): The window size for the moving average.

    Returns:
        pd.Series: The simple moving average.
    """
    return data.rolling(window=window).mean()


def exponential_moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA).

    Args:
        data (pd.Series): Price data.
        window (int): The window size for the moving average.

    Returns:
        pd.Series: The exponential moving average.
    """
    return data.ewm(span=window).mean()


def relative_strength_index(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).

    Args:
        data (pd.Series): Price data.
        window (int): The window size for the RSI.

    Returns:
        pd.Series: The relative strength index.
    """
    delta = data.diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = -loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def moving_average_convergence_divergence(
    data: pd.Series, short_window: int, long_window: int, signal_window: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) line, signal line, and histogram.

    Args:
        data (pd.Series): Price data.
        short_window (int): The short window size for EMA calculation.
        long_window (int): The long window size for EMA calculation.
        signal_window (int): The window size for the signal line.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: The MACD line, signal line, and histogram.
    """
    ema_short = exponential_moving_average(data, short_window)
    ema_long = exponential_moving_average(data, long_window)
    macd_line = ema_short - ema_long
    signal_line = exponential_moving_average(macd_line, signal_window)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic_oscillator(
    data: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    window: int,
    smooth_window: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate the Stochastic Oscillator (%K and %D lines).

    Args:
        data (pd.DataFrame): OHLC price data.
        high_col (str): Column name for the high prices.
        low_col (str): Column name for the low prices.
        close_col (str): Column name for the closing prices.
        window (int): The window size for the %K calculation.
        smooth_window (int): The window size for the %D calculation (smoothing of %K).

    Returns:
        Tuple[pd.Series, pd.Series]: The %K line and %D line.
    """
    highest_high = data[high_col].rolling(window=window).max()
    lowest_low = data[low_col].rolling(window=window).min()
    percent_k = ((data[close_col] - lowest_low) / (highest_high - lowest_low)) * 100
    percent_d = percent_k.rolling(window=smooth_window).mean()
    return percent_k, percent_d


def average_true_range(
    data: pd.DataFrame, high_col: str, low_col: str, close_col: str, window: int
) -> pd.Series:
    """
    Calculate the Average True Range (ATR).

    Args:
        data (pd.DataFrame): OHLC price data.
        high_col (str): Column name for the high prices.
        low_col (str): Column name for the low prices.
        close_col (str): Column name for the closing prices.
        window (int): The window size for the ATR calculation.

    Returns:
        pd.Series: The average true range.
    """
    prev_close = data[close_col].shift(1)
    high_low = data[high_col] - data[low_col]
    high_close = (data[high_col] - prev_close).abs()
    low_close = (data[low_col] - prev_close).abs()
    true_range = high_low.combine(high_close, max).combine(low_close, max)
    atr = true_range.rolling(window=window).mean()
    return atr


def on_balance_volume(data: pd.DataFrame, close_col: str, volume_col: str) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV).

    Args:
        data (pd.DataFrame): OHLCV price data.
        close_col (str): Column name for the closing prices.
        volume_col (str): Column name for the trading volume.

    Returns:
        pd.Series: The on-balance volume.
    """
    close_change = data[close_col].diff()
    volume_multiplier = pd.Series(
        np.where(close_change > 0, 1, np.where(close_change < 0, -1, 0)),
        index=data.index,
    )
    obv = (data[volume_col] * volume_multiplier).cumsum()
    return obv


def parabolic_sar(
    data: pd.DataFrame,
    high_col: str,
    low_col: str,
    initial_af: float = 0.02,
    max_af: float = 0.2,
) -> pd.Series:
    """
    Calculate the Parabolic SAR (PSAR).

    Args:
        data (pd.DataFrame): OHLC price data.
        high_col (str): Column name for the high prices.
        low_col (str): Column name for the low prices.
        initial_af (float): The initial acceleration factor.
        max_af (float): The maximum acceleration factor.

    Returns:
        pd.Series: The parabolic SAR.
    """
    psar = data[low_col].copy()
    is_long = True
    af = initial_af
    extreme_point = data[high_col][0]

    for i in range(1, len(data)):
        psar[i] = psar[i - 1] + af * (extreme_point - psar[i - 1])

        if is_long:
            if data[low_col][i] < psar[i]:
                is_long = False
                psar[i] = extreme_point
                extreme_point = data[low_col][i]
                af = initial_af
            else:
                if data[high_col][i] > extreme_point:
                    extreme_point = data[high_col][i]
                    af = min(af + initial_af, max_af)
        else:
            if data[high_col][i] > psar[i]:
                is_long = True
                psar[i] = extreme_point
                extreme_point = data[high_col][i]
                af = initial_af
            else:
                if data[low_col][i] < extreme_point:
                    extreme_point = data[low_col][i]
                    af = min(af + initial_af, max_af)

    return pd.Series(psar, index=data.index)


def rate_of_change(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Rate of Change (ROC).

    Args:
        data (pd.Series): Price data.
        window (int): The window size for the ROC calculation.

    Returns:
        pd.Series: The rate of change.
    """
    return ((data / data.shift(window)) - 1) * 100


def commodity_channel_index(
    data: pd.DataFrame, high_col: str, low_col: str, close_col: str, window: int
) -> pd.Series:
    """
    Calculate the Commodity Channel Index (CCI).

    Args:
        data (pd.DataFrame): OHLC price data.
        high_col (str): Column name for the high prices.
        low_col (str): Column name for the low prices.
        close_col (str): Column name for the closing prices.
        window (int): The window size for the CCI calculation.

    Returns:
        pd.Series: The commodity channel index.
    """
    typical_price = (data[high_col] + data[low_col] + data[close_col]) / 3
    mean_typical_price = typical_price.rolling(window=window).mean()
    mean_deviation = typical_price.rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    cci = (typical_price - mean_typical_price) / (0.015 * mean_deviation)
    return cci


def chaikin_money_flow(
    data: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
    window: int,
) -> pd.Series:
    """
    Calculate the Chaikin Money Flow (CMF).

    Args:
        data (pd.DataFrame): OHLCV price data.
        high_col (str): Column name for the high prices.
        low_col (str): Column name for the low prices.
        close_col (str): Column name for the closing prices.
        volume_col (str): Column name for the trading volume.
        window (int): The window size for the CMF calculation.

    Returns:
        pd.Series: The chaikin money flow.
    """
    money_flow_multiplier = (
        (data[close_col] - data[low_col]) - (data[high_col] - data[close_col])
    ) / (data[high_col] - data[low_col])
    money_flow_volume = money_flow_multiplier * data[volume_col]
    cmf = (
        money_flow_volume.rolling(window=window).sum()
        / data[volume_col].rolling(window=window).sum()
    )
    return cmf


def bollinger_bands(
    data: pd.Series, window: int, num_std: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Bollinger Bands (Upper Band, Middle Band, and Lower Band).

    Args:
        data (pd.Series): Price data.
        window (int): The window size for the moving average calculation.
        num_std (int): The number of standard deviations to use for the bands.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: The upper band, middle band (moving average), and lower band.
    """
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + num_std * std_dev
    lower_band = middle_band - num_std * std_dev
    return upper_band, middle_band, lower_band


def money_flow_index(
    data: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
    window: int,
) -> pd.Series:
    """
    Calculate the Money Flow Index (MFI).

    Args:
        data (pd.DataFrame): OHLCV price data.
        high_col (str): Column name for the high prices.
        low_col (str): Column name for the low prices.
        close_col (str): Column name for the closing prices.
        volume_col (str): Column name for the trading volume.
        window (int): The window size for the MFI calculation.

    Returns:
        pd.Series: The money flow index.
    """
    typical_price = (data[high_col] + data[low_col] + data[close_col]) / 3
    money_flow = typical_price * data[volume_col]
    positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_money_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    money_flow_ratio = (
        positive_money_flow.rolling(window=window).sum()
        / negative_money_flow.rolling(window=window).sum()
    )
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi


def accumulation_distribution_line(
    data: pd.DataFrame, high_col: str, low_col: str, close_col: str, volume_col: str
) -> pd.Series:
    """
    Calculate the Accumulation/Distribution Line (ADL).

    Args:
        data (pd.DataFrame): OHLCV price data.
        high_col (str): Column name for the high prices.
        low_col (str): Column name for the low prices.
        close_col (str): Column name for the closing prices.
        volume_col (str): Column name for the trading volume.

    Returns:
        pd.Series: The accumulation/distribution line.
    """
    money_flow_multiplier = (
        (data[close_col] - data[low_col]) - (data[high_col] - data[close_col])
    ) / (data[high_col] - data[low_col])
    money_flow_volume = money_flow_multiplier * data[volume_col]
    adl = money_flow_volume.cumsum()
    return adl


def average_directional_index(
    data: pd.DataFrame, high_col: str, low_col: str, close_col: str, window: int
) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX).

    Args:
        data (pd.DataFrame): OHLC price data.
        high_col (str): Column name for the high prices.
        low_col (str): Column name for the low prices.
        close_col (str): Column name for the closing prices.
        window (int): The window size for the ADX calculation.

    Returns:
        pd.Series: The average directional index.
    """
    # Calculate True Range
    prev_close = data[close_col].shift(1)
    tr = pd.DataFrame(
        {
            "TR1": data[high_col] - data[low_col],
            "TR2": (data[high_col] - prev_close).abs(),
            "TR3": (data[low_col] - prev_close).abs(),
        }
    )
    true_range = tr.max(axis=1)

    # Calculate directional movement
    up_move = data[high_col] - data[high_col].shift(1)
    down_move = data[low_col].shift(1) - data[low_col]
    pos_dm = up_move.where(up_move > down_move, 0).where(up_move > 0, 0)
    neg_dm = down_move.where(down_move > up_move, 0).where(down_move > 0, 0)

    # Calculate smoothed True Range and Directional Movement
    smoothed_true_range = true_range.rolling(window=window).sum()
    smoothed_pos_dm = pos_dm.rolling(window=window).sum()
    smoothed_neg_dm = neg_dm.rolling(window=window).sum()

    # Calculate Directional Indicators
    pos_di = 100 * smoothed_pos_dm / smoothed_true_range
    neg_di = 100 * smoothed_neg_dm / smoothed_true_range

    # Calculate ADX
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
    adx = dx.rolling(window=window).mean()

    return adx


def stochastic_oscillator(
    data: pd.DataFrame, high_col: str, low_col: str, close_col: str, window: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate the Stochastic Oscillator (%K and %D lines).

    Args:
        data (pd.DataFrame): OHLC price data.
        high_col (str): Column name for the high prices.
        low_col (str): Column name for the low prices.
        close_col (str): Column name for the closing prices.
        window (int): The window size for the Stochastic Oscillator calculation.

    Returns:
        Tuple[pd.Series, pd.Series]: The %K line and %D line.
    """
    highest_high = data[high_col].rolling(window=window).max()
    lowest_low = data[low_col].rolling(window=window).min()
    percent_k = 100 * (data[close_col] - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(window=window).mean()
    return percent_k, percent_d


def on_balance_volume(data: pd.DataFrame, close_col: str, volume_col: str) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV).

    Args:
        data (pd.DataFrame): OHLCV price data.
        close_col (str): Column name for the closing prices.
        volume_col (str): Column name for the trading volume.

    Returns:
        pd.Series: The on-balance volume.
    """
    delta_close = data[close_col].diff()
    obv = (
        data[volume_col]
        .where(delta_close > 0, -data[volume_col].where(delta_close < 0, 0))
        .cumsum()
    )
    return obv


def on_balance_volume(data: pd.DataFrame, close_col: str, volume_col: str) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV).

    Args:
        data (pd.DataFrame): OHLCV price data.
        close_col (str): Column name for the closing prices.
        volume_col (str): Column name for the trading volume.

    Returns:
        pd.Series: The on-balance volume.
    """
    obv = (
        data[volume_col]
        .where(data[close_col] > data[close_col].shift(1), -data[volume_col])
        .where(data[close_col] != data[close_col].shift(1), 0)
        .cumsum()
    )
    return obv


def aroon_indicator(
    data: pd.DataFrame, high_col: str, low_col: str, window: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate the Aroon Indicator (Aroon Up and Aroon Down).

    Args:
        data (pd.DataFrame): OHLC price data.
        high_col (str): Column name for the high prices.
        low_col (str): Column name for the low prices.
        window (int): The window size for the Aroon Indicator calculation.

    Returns:
        Tuple[pd.Series, pd.Series]: The Aroon Up and Aroon Down.
    """
    aroon_up = (
        100
        * (window - (data[high_col].rolling(window=window).apply(lambda x: x.argmax())))
        / window
    )
    aroon_down = (
        100
        * (window - (data[low_col].rolling(window=window).apply(lambda x: x.argmin())))
        / window
    )
    return aroon_up, aroon_down


indicators = [
    simple_moving_average,
    exponential_moving_average,
    bollinger_bands,
    relative_strength_index,
    average_true_range,
    on_balance_volume,
    parabolic_sar,
    rate_of_change,
    commodity_channel_index,
    chaikin_money_flow,
    money_flow_index,
    accumulation_distribution_line,
    average_directional_index,
    stochastic_oscillator,
    on_balance_volume,
    aroon_indicator,
]
