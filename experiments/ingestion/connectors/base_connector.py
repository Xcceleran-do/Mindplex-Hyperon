# ingestion/connectors/base_connector.py

from abc import ABC, abstractmethod
from typing import Any


class BaseConnector:

    def __init__(self, config):
        self.config = config

    def fetch(self):
        raise NotImplementedError("Connector must implement fetch()")