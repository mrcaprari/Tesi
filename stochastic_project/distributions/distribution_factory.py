import inspect
import torch
from abc import abstractmethod

class DistributionFactory:
    @staticmethod
    def create(family, role, **kwargs):
        # Dynamically create the name for the combined class
        class_name = f"{family.__name__}{role.__name__}"

        # Dynamically create the combined class
        ConcreteDistribution = type(
            class_name,
            (family, role),  # Use the family as a mixin
            {}
        )

        # Use a clean, modular __init__ definition
        def combined_init(self, *args, **kwargs):
            # Separate the kwargs for the role and family initializers
            role_kwargs = {}
            family_kwargs = {}
            # Extract 'prior' specifically for the role initializer
            if 'prior' in kwargs:
                role_kwargs['prior'] = kwargs.pop('prior')
            
            family_kwargs = kwargs
            if 'shape' not in family_kwargs:
                family_kwargs['shape'] = role_kwargs['prior'].shape
                
            # Remaining kwargs go to the family initializer
            # Call the __init__ methods of the base classes
            role.__init__(self, **role_kwargs)
            family.__init__(self, **family_kwargs)

        # Assign the custom __init__ to the dynamically created class
        ConcreteDistribution.__init__ = combined_init

        # Return the new combined class
        return ConcreteDistribution(**kwargs)


