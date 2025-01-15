from typing import Any, Type


class DistributionFactory:
    """
    A factory class to dynamically create trainable probability distributions by
    combining a distribution family (e.g., Gaussian) with a role (e.g., Prior or RandomParameter).
    """

    @staticmethod
    def create(family: Type, role: Type, **kwargs: Any) -> Any:
        """
        Dynamically creates a class by combining a distribution family and a role, and instantiates it.

        Args:
            family (Type): The base distribution family (e.g., Gaussian).
            role (Type): The role to apply to the distribution (e.g., Prior or RandomParameter).
            **kwargs (Any): Additional arguments to initialize the combined class.

        Returns:
            Any: An instance of the dynamically created class.
        """
        class_name = f"{family.__name__}{role.__name__}"
        ConcreteDistribution = type(class_name, (family, role), {})

        # Define the combined __init__ method
        def combined_init(self, *args: Any, **kwargs: Any) -> None:
            role_kwargs = {}
            family_kwargs = {}

            # Separate kwargs for role and family initialization
            if "prior" in kwargs:
                role_kwargs["prior"] = kwargs.pop("prior")

            family_kwargs = kwargs
            if "shape" not in family_kwargs:
                family_kwargs["shape"] = role_kwargs["prior"].shape

            # Call base class initializers
            role.__init__(self, **role_kwargs)
            family.__init__(self, **family_kwargs)

        # Assign the custom __init__ method
        ConcreteDistribution.__init__ = combined_init

        return ConcreteDistribution(**kwargs)
