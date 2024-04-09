import os
import pytest


def pytest_addoption(parser):
    parser.addoption("--base_url", action="store", default=os.path.dirname(__file__) + "/src/data/")


@pytest.fixture
def base_url(request):
    return request.config.getoption("--base_url")
