"""Basic installation tests for GWFIT."""

print("Testing GWFIT installation...\n")

import gwfit
import gwfit.physics
import gwfit.waveforms
import gwfit.nr
import gwfit.neutron_star

import gwfit.nr.nr_data_utils as nrutils


def main():
    # Package
    print(f"GWFIT version: {gwfit.__version__}")
    print(f"GWFIT location: {gwfit.__file__}")

    # Subpackages
    print("\nTesting subpackages:")
    print("  physics       ✓")
    print("  waveforms     ✓")
    print("  nr            ✓")
    print("  neutron_star  ✓")

    # NR utilities
    print("\nTesting NR utilities:")
    print(f"  SXSSimulation: {nrutils.SXSSimulation}")

    print("\nGWFIT installation test passed!")


if __name__ == "__main__":
    main()
