# Temperature Units

The library works internally in Kelvin but accepts and outputs temperatures in K, °C, or °F.

## Parameters

```python
auto_model_isothermal_data(
    input_temperature_units='K',   # 'K', 'C', or 'F' for input data
    output_temperature_units='K',  # 'K', 'C', or 'F' for results
)
```

## Usage

**Celsius (most common)**:
```python
results = auto_model_isothermal_data(
    data_files=['25C.csv', '40C.csv'],
    predict=(2, 'year', 25),
    input_temperature_units='C',
    output_temperature_units='C'
)
```

**Fahrenheit**:
```python
results = auto_model_isothermal_data(
    data_files=['77F.csv', '104F.csv'],
    predict=(2, 'year', 77),
    input_temperature_units='F',
    output_temperature_units='F'
)
```

**Mixed** (input in C, output in K):
```python
results = auto_model_isothermal_data(
    data_files=['25C.csv'],
    predict=(2, 'year', 25),
    input_temperature_units='C',
    output_temperature_units='K'  # Results in Kelvin
)
```

## Conversion Formulas

- **°C → K**: K = C + 273.15
- **°F → K**: K = (F - 32) × 5/9 + 273.15
- **K → °C**: C = K - 273.15
- **K → °F**: F = (K - 273.15) × 9/5 + 32

## What Gets Converted

**Input**: Data file temperatures, `predict` tuple, `simulate` profile
**Output**: All result temperatures, HTML report labels

All kinetic calculations happen in Kelvin internally.
