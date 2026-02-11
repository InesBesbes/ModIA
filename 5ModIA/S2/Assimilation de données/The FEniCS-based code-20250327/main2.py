import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import generate_case
import main

def test_sensitivity_frequency():
    # Paramètres fixes
    href = 10.0
    amp = href/5
    slopes = [1e-6, 1e-3, 1e-1, 2e-1, 1]
    n_wave = 3
    
    case_id = 0
    L = 100e3
    npts = 1001
    x = np.linspace(0., L, npts)
    x_km = x / 1000

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Surface libre H(x)")
    plt.xlabel("x (km)")
    plt.ylabel("H (m)")

    plt.subplot(1, 2, 2)
    plt.title("Bathymétrie b(x)")
    plt.xlabel("x (km)")
    plt.ylabel("z (m)")

    for slope in slopes:
        case_id += 1
        path = f"./data_case_{case_id}/"

        omega = 2 * np.pi / L

        # Génération de b(x)
        b_t = slope * (L - x)
        b_t += amp * np.cos(n_wave * omega * x)
        b_t += amp/2. * np.cos(2 * n_wave * omega * x)
        b_t += amp/2. * np.cos(3 * n_wave * omega * x)
        b_t += amp/3. * np.random.uniform(-1., 1., len(b_t))  # bruit

        b_b = np.linspace(b_t[0], b_t[-1], npts) - href
        H_in = L * slope + href
        H_out = href

        d = {"L": [L, 0.], "NP": [npts, 0], "href": href, "BC": [H_in, H_out]}
        dfr = pd.DataFrame(data=d)
        os.makedirs(path, exist_ok=True)
        dfr.to_csv(path + 'case.csv')
        np.save(path + 'bathy_t.npy', b_t)
        np.save(path + 'background.npy', b_b)
        Href = np.linspace(H_in, H_out, npts)
        np.save(path + "Href.npy", Href)

        # Exécution du modèle direct
        main.run_direct(path)
        H = np.load(path + "H_t.npy")

        # Affichage
        plt.subplot(1, 2, 1)
        plt.plot(x_km, H, label=f"slope = {slope}")
        plt.subplot(1, 2, 2)
        plt.plot(x_km, b_t, label=f"slope = {slope}")

    plt.subplot(1, 2, 1)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.legend()
    plt.grid(True)

    plt.suptitle("Influence de la pente globale de la rivière")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("influence_pente_bathy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    test_sensitivity_frequency()
