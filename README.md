pyIBCC
======

IBCC implementation for use with cit sci projects. Documentation provided at **[this link](https://github.com/CitizenScienceInAstronomyWorkshop/proceedings/wiki/How-to-run-IBCC)**.

The directory "python" contains the current working implementation of IBCC. The directory "attic" contains older code by Kieran Finn.

The required sample datasets and config files for some Galaxy Zoo CANDELS tests are in the repo.

To run the main Galaxy Zoo CANDELS test do:

	cd .../pyIBCC/python
    python ibcc.py config/galaxy-zoo-candels-test-full-500.py

This example uses 731998 Q1 (smooth/features/other) classifications
from the 500 most active classifiers. It takes less than a minute and
seems to produce meaningful results, although I have only glanced over
the outputs (in `.../pyIBCC/outputs/galaxy-zoo-candels-test`).

The results need to be looked at properly, the priors (alpha0 and nu0)
undoubtedly need much more attention, other questions need to be
tried, etc.

The input data is from the
`2013-11-17_galaxy_zoo_classifications_CANDELSonly.csv` file supplied
by Brooke.  There is an iPython notebook giving the code I used to
convert this data into the required format for pyIBCC:
`.../pyIBCC/data/galaxy-zoo-candels-test/prepare_data.ipynb`.

