# Basic tests for hardware-counter Model

import os
import pytest
import dspo


dir_path = os.path.dirname(os.path.realpath(__file__))
hwc_path = '%s/../data/hwc_chem/folding_V16_R1_T1/gas_phase_chemdr.csv'%dir_path

@pytest.fixture
def datapath():
    assert os.path.exists(hwc_path)
    return hwc_path

def test_BasicDNN(datapath):
    network = dspo.hwcmodel.BasicDNN(datapath)
    network.train_network()
