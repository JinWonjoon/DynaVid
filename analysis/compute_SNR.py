import math



if __name__ == "__main__":
    signal_power = 2.5899988787174224  # from get_variance_from_data.py result

    sigmas = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.9, 1.0, 1.43]  # latent scale 기준
    snr_linear = []
    snr_db = []

    for sigma in sigmas:
        noise_power = sigma ** 2
        snr = signal_power / noise_power
        snr_linear.append(snr)
        snr_db.append(10 * math.log10(snr))

    for s, l, d in zip(sigmas, snr_linear, snr_db):
        print(f"sigma={s:.3f} -> SNR={l:.2f} ({d:.2f} dB)")
        
    # sigma=0.010 -> SNR=25899.99 (44.13 dB)
    # sigma=0.020 -> SNR=6475.00 (38.11 dB)
    # sigma=0.050 -> SNR=1036.00 (30.15 dB)
    # sigma=0.100 -> SNR=259.00 (24.13 dB)
    # sigma=0.150 -> SNR=115.11 (20.61 dB)
    # sigma=0.200 -> SNR=64.75 (18.11 dB)
    # sigma=0.250 -> SNR=41.44 (16.17 dB)
    # sigma=0.300 -> SNR=28.78 (14.59 dB)
    # sigma=0.350 -> SNR=21.14 (13.25 dB)
    # sigma=0.400 -> SNR=16.19 (12.09 dB)
    # sigma=0.450 -> SNR=12.79 (11.07 dB)
    # sigma=0.500 -> SNR=10.36 (10.15 dB)
    # sigma=0.900 -> SNR=3.20 (5.05 dB)
    # sigma=1.000 -> SNR=2.59 (4.13 dB)
    # sigma=1.430 -> SNR=1.27 (1.03 dB)   