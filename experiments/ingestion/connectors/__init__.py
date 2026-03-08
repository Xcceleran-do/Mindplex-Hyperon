from .csv_connector import CsvConnector
from .pdf_connector import PdfConnector
from .json_api_connector import JsonApiConnector


def get_connector(source_type, config):

    if source_type == "csv":
        return CsvConnector(config)

    if source_type == "pdf":
        return PdfConnector(config)

    if source_type == "json_api":
        return JsonApiConnector(config)

    raise ValueError(f"Unknown source type {source_type}")