"""
Content extractors for different file types.

This module provides extractors that parse and chunk different types of content
(Python code, markdown documentation, git history) into indexed chunks.
"""

from kaia.extractors.code_extractor import CodeExtractor
from kaia.extractors.doc_extractor import DocExtractor
from kaia.extractors.git_extractor import GitExtractor

__all__ = ["CodeExtractor", "DocExtractor", "GitExtractor"]
