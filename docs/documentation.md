# Inventory Management Environment Documentation

## Overview

The Inventory Management Environment (`InvPricingManagementMasterEnv`) is a custom OpenAI Gym environment designed for simulating multi-period inventory management scenarios with stochastic demand influenced by various economic factors.

## Environment Details

### State Space

The environment's state is represented by a 5-dimensional vector:

1. **Inventory Level**: Current inventory on hand.
2. **Censoring Indicator**: Whether the demand was censored in the previous period.
3. **Previous Sales**: Sales in the previous period or average demand if not available.
4. **State of Economy**: Binary indicator representing the economic state.
5. **Interest Rate**: Current interest rate affecting the economy.

### Action Space

The action space is a combination of:

- **Ordering Quantity**: Discrete values from 0 to 15 units.
- **Pricing Level**: Three pricing options mapped to actual prices (e.g., {0: \$2, 1: \$4, 2: \$6}).

### Reward Function

The reward is calculated based on the profit, which considers:

- Revenue from sales.
- Cost of ordering inventory.
- Cost of unfulfilled demand.
- Holding cost for excess inventory.

## Usage

Refer to the `examples/run_example.py` script for a basic usage example.

## Mathematical Formulation

[Include detailed mathematical equations and explanations here.]

## Assumptions

- Demand is influenced by price elasticity, economic state, and interest rate and one step before demand level.
- Censoring occurs when demand exceeds current inventory.



