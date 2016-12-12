"""General tests that simply indicate if a target is invocable or not"""

def test_pytest():
    """Reporting of this test indicates that pytest is being used as expected."""

    assert 1 == 1

def test_pybrain():
    from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

    # create a network
    n = FeedForwardNetwork()

    # create layers
    inLayer = LinearLayer(2)
    hiddenLayer = SigmoidLayer(3)
    outLayer = LinearLayer(1)

    # add layers
    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)

    # create connections
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    # add connections
    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)

    # config a network
    n.sortModules()

    # generate test output
    out = n.activate([1, 2])

    assert isinstance(out[0], float)
