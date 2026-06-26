"""Custom exceptions for the sports science search engine."""


class PDFExtractionError(Exception):
    """Raised when PDF cannot be read or parsed."""

    pass


class SectionDetectionError(Exception):
    """Raised when sections cannot be detected and no manual mapping exists."""

    pass


class VectorUploadError(Exception):
    """Raised when Qdrant upload fails."""

    pass


class PDFProcessingError(Exception):
    """General processing error with context."""

    pass
