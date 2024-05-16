#!/bin/bash
pip install -e libs/dpart/
pip install -e libs/DataSynthesizer/
pip install -e libs/DataSynthesizer_v014/
pip install -e libs/smartnoise-sdk/sql/
pip install -e libs/smartnoise-sdk/synth/
pip uninstall opendp
pip install -e libs/opendp/
pip install git+https://github.com/ryan112358/private-pgm.git
pip install -e libs/optimized_qbs/
pip install opacus==1.4.0
pip install scikit-learn==1.0.2