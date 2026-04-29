# =============================================
# src/__init__.py
# =============================================
# This file makes the 'src' folder a Python PACKAGE.
# Without this file, Python won't recognize 'src' as a module
# and you won't be able to do: from src.data_loader import ...
#
# WHY PACKAGES MATTER:
# Instead of writing ALL code in one giant notebook,
# we split it into MODULES (separate .py files).
# This is how real-world projects are organized.
# Your notebook then imports from these modules:
#   from src.data_loader import fetch_stock_data
#   from src.model import train_model
# =============================================
