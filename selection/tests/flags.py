import os

SMALL_SAMPLES = False
SET_SEED = False

if "USE_SMALL_SAMPLES" in os.environ:
    SMALL_SAMPLES = True

if "USE_TEST_SEED" in os.environ:
    SET_SEED = True