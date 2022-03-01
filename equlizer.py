#! /bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def calc_third_octave_band_levels(
    m: np.ndarray, sr: np.intc, order: np.intc = 4
) -> np.ndarray:
    third_octave_values = np.zeros_like(third_octave_names, dtype=np.double)
    third_octave_values[0] = np.sum(
        np.square(high_cut_butter(m, sr, third_octave_cuts[0], order))
    )
    third_octave_values[len(third_octave_values) - 1] = np.sum(
        np.square(
            low_cut_butter(m, sr, third_octave_cuts[len(third_octave_cuts) - 1], order)
        )
    )
    for i in range(0, len(third_octave_cuts) - 1):
        third_octave_values[i + 1] = np.sum(
            np.square(
                band_pass_butter(
                    m, sr, third_octave_cuts[i], third_octave_cuts[i + 1], order
                )
            )
        )

    return 10 * np.log10(third_octave_values)


def display_thrid_octave_bands(m: np.ndarray, sr: np.intc):
    third_octave_values = np.clip(calc_third_octave_band_levels(m, sr) + 30, 0.0, None)
    plt.bar(third_octave_names, third_octave_values)
    plt.xlabel("Freq [Hz]")
    plt.ylabel("Amplitude")
    plt.show()


def calc_xth_octave_band_cuts(
    x: np.double, fmin: np.double, fmax: np.double
) -> np.ndarray:
    factor = np.power(2.0, 1.0 / x)
    i = 0
    cuts = [fmin]
    while cuts[i] * factor < fmax:
        cuts.append(cuts[i] * factor)
        i = i + 1
    return np.array(cuts)


def calc_xth_octave_band_levels(
    m: np.ndarray, sr: np.intc, cuts: np.ndarray, order: np.intc = 4
) -> np.ndarray:
    xth_octave_values = np.zeros(cuts.size + 1, dtype=np.double)
    xth_octave_values[0] = np.sum(np.square(high_cut_butter(m, sr, cuts[0], order)))
    xth_octave_values[len(xth_octave_values) - 1] = np.sum(
        np.square(low_cut_butter(m, sr, cuts[len(cuts) - 1], order))
    )
    for i in range(0, len(cuts) - 1):
        xth_octave_values[i + 1] = np.sum(
            np.square(band_pass_butter(m, sr, cuts[i], cuts[i + 1], order))
        )

    return 10 * np.log10(xth_octave_values)


def display_xth_octave_bands(m: np.ndarray, sr: np.intc, x: np.intc):
    cuts = calc_xth_octave_band_cuts(x, 10, 24000)
    xth_octave_values = np.clip(
        calc_xth_octave_band_levels(m, sr, cuts) + 30, 0.0, None
    )
    plt.bar(np.arange(len(cuts) + 1), xth_octave_values)
    plt.xlabel("Freq [Hz]")
    plt.ylabel("Amplitude")
    plt.show()


def low_shelf_1order(
    m: np.ndarray, sr: np.intc, gain_db: np.double, freq: np.double
) -> np.ndarray:
    G = np.power(10.0, 0.1 * gain_db)
    sqrt_G = np.sqrt(G)
    tan_w_c_half = np.tan(np.pi * freq / sr)
    norm = 1.0 / (tan_w_c_half + sqrt_G)
    coff = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    coff[0][0] = norm * (G * tan_w_c_half + sqrt_G)
    coff[0][1] = norm * (G * tan_w_c_half - sqrt_G)
    coff[0][3] = 1.0
    coff[0][4] = norm * (tan_w_c_half - sqrt_G)
    return signal.sosfilt(coff, m)


def high_shelf_1order(
    m: np.ndarray, sr: np.intc, gain_db: np.double, freq: np.double
) -> np.ndarray:
    G = np.power(10.0, 0.1 * gain_db)
    sqrt_G = np.sqrt(G)
    tan_w_c_half = np.tan(np.pi * freq / sr)
    norm = 1.0 / (sqrt_G * tan_w_c_half + 1.0)
    coff = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    coff[0][0] = norm * (sqrt_G * tan_w_c_half + G)
    coff[0][1] = norm * (sqrt_G * tan_w_c_half - G)
    coff[0][3] = 1.0
    coff[0][4] = norm * (sqrt_G * tan_w_c_half - 1.0)
    return signal.sosfilt(coff, m)


def peak_filter_2order_1(
    m: np.ndarray, sr: np.intc, gain_db: np.double, freq: np.double, Q: np.double
) -> np.ndarray:
    w_c = 2.0 * np.pi * freq / sr
    G = np.power(10.0, 0.1 * gain_db)
    sqrt_G = np.sqrt(G)
    tan_B_half = np.tan(w_c / Q / 2.0)
    cos_w_c = np.cos(w_c)
    norm = 1.0 / (tan_B_half + sqrt_G)
    coff = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    coff[0][0] = norm * (sqrt_G + G * tan_B_half)
    coff[0][1] = norm * (-2.0 * sqrt_G * cos_w_c)
    coff[0][2] = norm * (sqrt_G - G * tan_B_half)
    coff[0][3] = 1.0
    coff[0][4] = norm * (-2.0 * sqrt_G * cos_w_c)
    coff[0][5] = norm * (sqrt_G - tan_B_half)
    return signal.sosfilt(coff, m)


def high_shelf_2order(
    m: np.ndarray, sr: np.intc, gain_db: np.double, freq: np.double
) -> np.ndarray:
    O = np.tan(np.pi * freq / sr)
    G = np.power(10.0, 0.1 * gain_db)
    sqrt_G = np.sqrt(G)
    rt4_G = np.sqrt(sqrt_G)
    sqrt_2 = np.sqrt(2.0)
    norm = 1.0 / (sqrt_G * O * O + sqrt_2 * O * rt4_G + 1.0)
    coff = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    coff[0][0] = norm * sqrt_G * (sqrt_G + sqrt_2 * O * rt4_G + O * O)
    coff[0][1] = norm * sqrt_G * -2.0 * (sqrt_G - O * O)
    coff[0][2] = norm * sqrt_G * (sqrt_G - sqrt_2 * O * rt4_G + O * O)
    coff[0][3] = 1.0
    coff[0][4] = norm * 2.0 * (sqrt_G * O * O - 1.0)
    coff[0][5] = norm * (sqrt_G * O * O - sqrt_2 * O * rt4_G + 1.0)
    return signal.sosfilt(coff, m)


def low_shelf_2order(
    m: np.ndarray, sr: np.intc, gain_db: np.double, freq: np.double
) -> np.ndarray:
    O = np.tan(np.pi * freq / sr)
    G = np.power(10.0, 0.1 * gain_db)
    sqrt_G = np.sqrt(G)
    rt4_G = np.sqrt(sqrt_G)
    sqrt_2 = np.sqrt(2.0)
    norm = 1.0 / (sqrt_G + sqrt_2 * O * rt4_G + O * O)
    coff = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    coff[0][0] = norm * sqrt_G * (sqrt_G * O * O + sqrt_2 * O * rt4_G + 1.0)
    coff[0][1] = norm * sqrt_G * 2.0 * (sqrt_G * O * O - 1.0)
    coff[0][2] = norm * sqrt_G * (sqrt_G * O * O - sqrt_2 * O * rt4_G + 1.0)
    coff[0][3] = 1.0
    coff[0][4] = norm * 2.0 * (O * O - sqrt_G)
    coff[0][5] = norm * (sqrt_G - sqrt_2 * O * rt4_G + O * O)
    return signal.sosfilt(coff, m)


def low_cut_butter(
    m: np.ndarray, sr: np.intc, freq: np.double, order: np.intc = 4
) -> np.ndarray:
    coff = signal.butter(order, freq, btype="highpass", output="sos", fs=sr)
    return signal.sosfilt(coff, m)


def high_cut_butter(
    m: np.ndarray, sr: np.intc, freq: np.double, order: np.intc = 4
) -> np.ndarray:
    coff = signal.butter(order, freq, btype="lowpass", output="sos", fs=sr)
    return signal.sosfilt(coff, m)


def band_pass_butter(
    m: np.ndarray,
    sr: np.intc,
    freq_low: np.double,
    freq_high: np.double,
    order: np.intc = 4,
) -> np.ndarray:
    coff = signal.butter(
        order, (freq_low, freq_high), btype="bandpass", output="sos", fs=sr
    )
    return signal.sosfilt(coff, m)


def four_band_eq_bank(
    m: np.ndarray, sr: np.intc, params: np.ndarray((6, 3), dtype=np.double)
) -> np.ndarray:
    if params[0][0] != 0.0:
        m = low_cut_butter(m, sr, params[0][1])
    m = peak_filter_2order_1(m, sr, params[1][0], params[1][1], params[1][2])
    m = peak_filter_2order_1(m, sr, params[2][0], params[2][1], params[2][2])
    m = peak_filter_2order_1(m, sr, params[3][0], params[3][1], params[3][2])
    m = peak_filter_2order_1(m, sr, params[4][0], params[4][1], params[4][2])
    if params[5][0] != 0.0:
        m = high_cut_butter(m, sr, params[5][1])
    return m


def main():
    print("You shouldn't run this as main")


if __name__ == "__main__":
    main()
