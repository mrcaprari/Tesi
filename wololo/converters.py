import math
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.fx
from torch.fx import GraphModule

from .adapters import Adapter
from .algorithms import (
    ParticleParameter,
    all_particles,
    bbvi_initialize,
    bbvi_kl_divergence,
    compute_kernel_matrix,
    perturb_gradients,
    svgd_initialize,
)
from .tracers import PreparatoryTracer
from .transformers import VmapTransformer


class Converter:
    """
    Manages the transformation of a deterministic model into a stochastic model by:
    - Tracing and preparing the computation graph of the deterministic model.
    - Adapting parameters and nodes stochasticly, accounting for multiple realizations.
    - Transforming the forward logic to support batched parameter dimensions.

    Attributes:
        tracer (torch.fx.Tracer): Traces the computation graph. Defaults to `PreparatoryTracer`,
            which adds "transform" metadata to nodes for stochastic adaptation; user can change this
            to allow for tracing dynamic architectures.
        adapter (Adapter): Adapts the graph and substitutes parameters with stochastic modules.
            Defaults to `Adapter`.
        transformer (torch.fx.Transformer): Transforms the forward method. Defaults to `VmapTransformer`,
            well suited for dense architectures; users can change this for more optimized forward() transformations.
        toplevel_methods (Dict): A dictionary of methods to be added to the transformed module,
            useful for implementing posterior approximation algorithms like BBVI or SVGD.
    """

    def __init__(
        self,
        tracer: torch.fx.Tracer = None,
        adapter: Adapter = None,
        transformer: torch.fx.Transformer = None,
        toplevel_methods: Dict = None,
    ) -> None:
        """
        Initializes the Converter with configurable components.

        Args:
            tracer (torch.fx.Tracer, optional): A tracer for graph preparation.
                Defaults to `PreparatoryTracer`.
            adapter (Adapter, optional): Handles stochastic adaptation of parameters and nodes.
                Defaults to `Adapter`.
            transformer (torch.fx.Transformer, optional): Handles forward method transformation.
                Defaults to `VmapTransformer`.
            toplevel_methods (Dict, optional): Methods to add at the top level of the transformed module.
        """
        self.tracer = tracer or PreparatoryTracer
        self.adapter = adapter or Adapter
        self.transformer = transformer or VmapTransformer
        self.toplevel_methods = toplevel_methods or {}

    def convert(
        self,
        module: torch.nn.Module,
        stochastic_parameter: type[torch.nn.Module],
        parameter_list: Optional[list] = None,
        **kwargs,
    ) -> GraphModule:
        """
        Converts a module by applying the three-part transformation: tracing, stochastic adaptation,
        and forward method transformation. Adds methods specified in `toplevel_methods` to the final module.

        Args:
            module (torch.nn.Module): The module to be transformed.
            stochastic_parameter (type[torch.nn.Module]): A class representing stochastic parameters.
                The `forward()` method of this class must return realizations of the parameter
                and accept `n_samples` as input to generate multiple realizations.
            parameter_list (Optional[List[str]], optional): List of parameter names to replace stochastically.
                Defaults to all parameters if not specified.

        Returns:
            GraphModule: The transformed module with stochastic parameters and an updated computation graph.

        Notes:
            - For dense architectures, specify the `stochastic_parameter` class, and Converter
              handles stochastic adaptation automatically.
            - For dynamic architectures, customize the `tracer` to accommodate for
              different graph preparation or transformation approaches.
            - For convolutional architectures, customize the `transformer` to accommodate for
              different transformation approaches.
        """
        # Default parameter list to be empty if not specified
        # This will transform every parameter in the module
        if parameter_list is None:
            parameter_list = []

        # Trace and prepare the computation graph
        original_graph = self.tracer(parameter_list).trace(module)

        # Substitute parameters and adapt the graph accordingly
        new_module, new_graph = self.adapter(original_graph).adapt_module(
            module, stochastic_parameter, **kwargs
        )

        # Transforming the forward method to account for additional parameter dimensions
        # This is necessary to handle batched computations for stochastic parameters
        transformed_module = GraphModule(new_module, new_graph)
        final_module = self.transformer(transformed_module).transform()

        # Finalize transformation adding top-level methods
        self.add_methods(final_module, stochastic_parameter)
        return final_module

    def add_methods(self, module: GraphModule, stochastic_parameter) -> None:
        """
        Adds user-defined methods to the transformed module.

        Args:
            module (GraphModule): The module to which methods will be added.
        """

        for method_name, method_function in self.toplevel_methods.items():
            setattr(module, method_name, method_function.__get__(module, type(module)))


class BBVIConverter(Converter):
    """
    Specialized Converter for Black-Box Variational Inference (BBVI).

    Extends the base Converter by ensuring that the provided stochastic parameter
    includes a `kl_divergence` method. It also adds a top-level `kl_divergence`
    method to the transformed module and initializes BBVI-specific attributes
    via the `bbvi_initialize` function.

    Attributes:
        (Inherited from Converter)
    """

    def __init__(
        self,
        tracer: torch.fx.Tracer = None,
        adapter: Adapter = None,
        transformer: torch.fx.Transformer = None,
        toplevel_methods: Dict = None,
    ) -> None:
        """
        Initializes the BBVIConverter with optional components and automatically
        adds the `kl_divergence` method to the top-level methods.

        Args:
            tracer (torch.fx.Tracer, optional): Tracer for graph preparation.
                Defaults to `PreparatoryTracer`.
            adapter (Adapter, optional): Handles stochastic adaptation of parameters
                and nodes. Defaults to `Adapter`.
            transformer (torch.fx.Transformer, optional): Transforms the forward method.
                Defaults to `VmapTransformer`.
            toplevel_methods (Dict, optional): Additional methods to add to the transformed module.
        """

        toplevel_methods = toplevel_methods or {}

        # Add KL divergence method to toplevel module methods
        toplevel_methods.update({"kl_divergence": bbvi_kl_divergence})

        # Call parent initializer
        super().__init__(tracer, adapter, transformer, toplevel_methods)

    def convert(
        self,
        module: torch.nn.Module,
        stochastic_parameter: type[torch.nn.Module],
        parameter_list: Optional[list] = None,
        **kwargs,
    ) -> GraphModule:
        """
        Converts a deterministic module into a stochastic module suited for BBVI.

        The conversion process includes:
          - Tracing the module's computation graph.
          - Replacing parameters with stochastic ones (which must implement a `kl_divergence` method).
          - Transforming the forward method to handle batched parameter dimensions.
          - Initializing BBVI-specific attributes on the resulting module.

        Args:
            module (torch.nn.Module): The original deterministic module.
            stochastic_parameter (type[torch.nn.Module]): The class used for stochastic parameters.
                It must provide a `kl_divergence` method.
            parameter_list (Optional[List[str]], optional): Names of parameters to replace stochastically.
                Defaults to an empty list (i.e., transform all parameters).
            **kwargs: Additional keyword arguments for the conversion process.

        Returns:
            GraphModule: The transformed module with BBVI-specific enhancements.

        Raises:
            ValueError: If the provided stochastic_parameter does not implement `kl_divergence`.
        """

        if not hasattr(stochastic_parameter, "kl_divergence"):
            raise ValueError(
                "stochastic_parameter must have a kl_divergence method for BBVI"
            )

        # Call parent convert method
        BBVI_model = super().convert(
            module, stochastic_parameter, parameter_list, **kwargs
        )

        # Initialize BBVI-specific attributes
        bbvi_initialize(BBVI_model, stochastic_parameter)

        return BBVI_model


class SVGDConverter(Converter):
    """
    Specialized Converter for Stein Variational Gradient Descent (SVGD).

    Extends the base Converter by automatically adding SVGD-specific methods
    as top-level methods. These include:
      - `all_particles`: Concatenates particles across all stochastic parameters.
      - `compute_kernel_matrix`: Computes the RBF kernel matrix.
      - `perturb_gradients`: Perturbs particle gradients using the computed kernel matrix.

    After conversion, the module is further initialized with SVGD-specific attributes via
    the `svgd_initialize` function.

    Attributes:
        (Inherited from Converter)
    """

    def __init__(
        self,
        tracer: torch.fx.Tracer = None,
        adapter: Adapter = None,
        transformer: torch.fx.Transformer = None,
        toplevel_methods: Dict = None,
    ) -> None:
        """
        Initializes the SVGDConverter with optional components and automatically
        adds SVGD-specific methods (all_particles, compute_kernel_matrix, perturb_gradients)
        to the top-level methods.

        Args:
            tracer (torch.fx.Tracer, optional): Tracer for graph preparation.
                Defaults to `PreparatoryTracer`.
            adapter (Adapter, optional): Handles stochastic adaptation of parameters.
                Defaults to `Adapter`.
            transformer (torch.fx.Transformer, optional): Transforms the forward method.
                Defaults to `VmapTransformer`.
            toplevel_methods (Dict, optional): Additional methods to add to the transformed module.
        """

        toplevel_methods = toplevel_methods or {}
        toplevel_methods.update(
            {
                "all_particles": all_particles,
                "compute_kernel_matrix": compute_kernel_matrix,
                "perturb_gradients": perturb_gradients,
            }
        )
        super().__init__(tracer, adapter, transformer, toplevel_methods)

    def convert(
        self,
        module: torch.nn.Module,
        stochastic_parameter: type[torch.nn.Module] = ParticleParameter,
        parameter_list: Optional[list] = None,
        **kwargs,
    ) -> GraphModule:
        """
        Converts a deterministic module into a stochastic module tailored for SVGD.

        The conversion process includes:
          - Tracing the module's computation graph.
          - Replacing parameters with stochastic ones (default is ParticleParameter).
          - Transforming the forward method to support batched parameter dimensions.
          - Initializing SVGD-specific attributes (e.g., registering particle submodules).

        Args:
            module (torch.nn.Module): The original deterministic module.
            stochastic_parameter (type[torch.nn.Module], optional): The class representing stochastic parameters.
                Defaults to ParticleParameter.
            parameter_list (Optional[List[str]], optional): List of parameter names to be transformed.
            **kwargs: Additional keyword arguments for the conversion process.

        Returns:
            GraphModule: The transformed module with SVGD-specific enhancements.
        """
        # Call parent convert method
        transformed_module = super().convert(
            module, stochastic_parameter, parameter_list, **kwargs
        )

        # Initialize SVGD-specific attributes
        svgd_initialize(transformed_module)

        return transformed_module
