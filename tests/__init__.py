#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define the set of tests for cube-builder."""

import pytest

if __name__ == "__main__":
    import test_timeline

    pytest.main(["--color=auto", "--no-cov", "-v"])
