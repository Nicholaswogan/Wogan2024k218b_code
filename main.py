import habitable_climate
import habitable
import habitable_plot
import neptune_climate
import neptune
import neptune_plot
import make_spectra
import spectra_plot

def main():
    habitable_climate.main()
    habitable.main()
    habitable_plot.main()
    neptune_climate.main()
    neptune.main()
    neptune_plot.main()
    make_spectra.main()
    spectra_plot.main()

if __name__ == '__main__':
    main()
