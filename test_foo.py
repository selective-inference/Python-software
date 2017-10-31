from selection.quadratic_program import foo
import numpy as np

A = np.arange(10) * 2.
B = A.copy()
print(B, foo(A))
