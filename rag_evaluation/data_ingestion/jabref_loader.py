"""
Jabref Data Loader

Loads evaluation data from Jabref (BibTeX) format files.
Jabref is a bibliography reference manager that uses BibTeX format.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class JabrefLoader:
    """
    Loader for Jabref (BibTeX) format data.
    
    This loader can parse BibTeX files exported from Jabref and extract
    relevant information for RAG evaluation, including abstracts, notes,
    and other fields that can serve as context or queries.
    """
    
    def __init__(self):
        """Initialize the Jabref loader."""
        pass
    
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from a Jabref BibTeX file.
        
        Args:
            file_path: Path to the BibTeX file
            
        Returns:
            List of dictionaries containing parsed entries
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if it's a BibTeX or JSON format
        if path.suffix.lower() == '.json':
            return self._load_json(file_path)
        else:
            return self._load_bibtex(file_path)
    
    def _load_bibtex(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse BibTeX format file.
        
        Args:
            file_path: Path to BibTeX file
            
        Returns:
            List of parsed entries
        """
        entries = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple BibTeX parser (can be enhanced with bibtexparser library)
        import re
        
        # Find all entries
        entry_pattern = r'@(\w+)\{([^,]+),\s*(.*?)\n\}'
        matches = re.finditer(entry_pattern, content, re.DOTALL)
        
        for match in matches:
            entry_type = match.group(1)
            entry_key = match.group(2)
            fields_str = match.group(3)
            
            # Parse fields
            fields = {}
            field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}|(\w+)\s*=\s*"([^"]*)"'
            for field_match in re.finditer(field_pattern, fields_str):
                if field_match.group(1):
                    field_name = field_match.group(1)
                    field_value = field_match.group(2)
                else:
                    field_name = field_match.group(3)
                    field_value = field_match.group(4)
                
                fields[field_name.lower()] = field_value.strip()
            
            entry = {
                'type': entry_type,
                'key': entry_key,
                'fields': fields,
                'query': fields.get('title', ''),
                'context': fields.get('abstract', '') or fields.get('note', ''),
                'ground_truth': fields.get('abstract', '')
            }
            
            entries.append(entry)
        
        return entries
    
    def _load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load Jabref data from JSON format.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of entries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single entry and list of entries
        if isinstance(data, dict):
            data = [data]
        
        entries = []
        for item in data:
            entry = {
                'query': item.get('query', item.get('title', '')),
                'context': item.get('context', item.get('abstract', '')),
                'answer': item.get('answer', ''),
                'ground_truth': item.get('ground_truth', item.get('abstract', '')),
                'metadata': item.get('metadata', {})
            }
            entries.append(entry)
        
        return entries
    
    def load_for_evaluation(self, file_path: str) -> Dict[str, List[str]]:
        """
        Load data in a format ready for batch evaluation.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Dictionary with lists of queries, contexts, answers, and ground_truths
        """
        entries = self.load(file_path)
        
        return {
            'queries': [e.get('query', '') for e in entries],
            'contexts': [e.get('context', '') for e in entries],
            'answers': [e.get('answer', '') for e in entries],
            'ground_truths': [e.get('ground_truth', '') for e in entries]
        }
