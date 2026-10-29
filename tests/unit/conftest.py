def pytest_configure(config):
    config.addinivalue_line("markers", "requirement(requirement_id): little-requirements requirement ID")
