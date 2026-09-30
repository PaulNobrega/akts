# AKTS Documentation

Complete documentation for the AKTS Python library for kinetic analysis.

## Documentation Index

### Getting Started
- **[Quick Start Guide](getting_started.md)** - Installation and first steps
- **[Example Scripts](examples.md)** - How to run and understand the examples

### For Non-Expert Users
- **[Automated Analysis Guide](automated_analysis.md)** - Using `auto_model_isothermal_data()`
- **[JSON Input/Output Specification](json_io_specification.md)** - Web API integration

### For Expert Users
- **[API Reference](api_reference.md)** - Complete function reference
- **[Advanced Usage](advanced_usage.md)** - Custom workflows and fine control

### Scientific Background
- **[Kinetic Models Guide](kinetic_models.md)** - Model equations, mechanisms, and applications
- **[Model Selection Guide](model_selection_guide.md)** - When to use which model
- **[Model Selector Guide](model_selector.md)** - IDE autocomplete for model discovery
- **[Experimental Design](experimental_design.md)** - How to design isothermal, DSC, and TGA experiments

### Technical Details
- **[Multistart Fitting Explained](bayesian_optimization.md)** - How ODE models are fit reliably
- **[Numerical Solvers and Performance](advanced_usage.md#numerical-solvers-and-performance)** - Choosing primary/fallback ODE solvers, closed-form fast path, reproducible bootstraps
- **[Temperature Unit Conversion](temperature_units.md)** - Working with K, °C, and °F
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## Quick Navigation

- [Analyze stability data](automated_analysis.md)
- [Discover available models with IDE autocomplete](model_selector.md)
- [Integrate with a web API](json_io_specification.md)
- [Review kinetic models](kinetic_models.md)
- [Design experiments](experimental_design.md)
- [Use the API directly](api_reference.md)
- [Choose a model](model_selection_guide.md)
- [Troubleshoot an issue](troubleshooting.md)

## Getting Help

- **Installation issues**: [Quick Start Guide](getting_started.md)
- **Error messages**: [Troubleshooting](troubleshooting.md)
- **Scientific questions**: [Kinetic Models Guide](kinetic_models.md)
- **API questions**: [API Reference](api_reference.md)

## Contributing

Report documentation issues or suggest changes through a GitHub issue or pull request.

---

