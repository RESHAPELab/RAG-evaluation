"""
Data ingestion module for loading evaluation data from various formats.
"""

from .jabref_loader import JabrefLoader
from .datatable_loader import DataTableLoader

__all__ = ['JabrefLoader', 'DataTableLoader']
