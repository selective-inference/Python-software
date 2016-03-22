import numpy as np
from regreg.atoms.seminorms import seminorm

class selective_penalty(seminorm):

    def setup_sampling(self, 
                       gradient, 
                       soln, 
                       linear_randomization,
                       quadratic_coef):

        """
        Should store quadratic_coef.
        Its return value is the chosen parametrization
        of the selection event.

        In other API methods, this return value is 
        referred to as `opt_vars`
        """

        raise NotImplementedError("abstract method")

    def form_subgradient(self, opt_vars):
        """
        Given the chosen parametrization
        of the selection event, this should form
        `z`, an element the subgradient of the penalty
        at `beta`.
        """
        raise NotImplementedError("abstract method")

    def form_parameters(self, opt_vars):
        """
        Given the chosen parametrization
        of the selection event, this should form
        `beta`.
        """
        raise NotImplementedError("abstract method")

    def form_optimization_vector(self, opt_vars):
        """
        Given the chosen parametrization
        of the selection event, this should form
        `(beta, z, epsilon * beta + z)`.
        """
        raise NotImplementedError("abstract method")

    def log_jacobian(self, hessian):
        """
        Given the Hessian of the loss at `beta`,
        compute the appropriate Jacobian which is the 
        determinant of this matrix plus the Jacobian
        of the map $\epsilon \beta + z$
        """
        raise NotImplementedError("abstract method")

    def step_variables(self, state, randomization, logpdf, gradient):
        """
        State is a tuple (data, opt_vars).
        This method should take a Metropolis-Hastings
        step for `opt_vars`.

        The logpdf, is the callable that computes
        the density of the randomization, 
        as well as the jacobian of the parameterization.

        randomization should be a callable that samples
        from the original randomization density.
        """
        raise NotImplementedError("abstract method")

