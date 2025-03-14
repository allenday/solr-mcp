#!/usr/bin/env python3
"""
Script to process markdown files, splitting them by section headings
and preparing them for indexing in Solr with vector embeddings.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import frontmatter


def extract_sections(markdown_content: str) -> List[Tuple[str, str]]:
    """
    Extract sections from a markdown document based on headings.
    
    Args:
        markdown_content: The content of the markdown file
        
    Returns:
        List of tuples (section_title, section_content)
    """
    # Split by headers (# Header)
    header_pattern = r'^(#{1,6})\s+(.+?)$'
    lines = markdown_content.split('\n')
    
    sections = []
    current_title = "Introduction"
    current_content = []
    
    for line in lines:
        header_match = re.match(header_pattern, line, re.MULTILINE)
        
        if header_match:
            # Save previous section
            if current_content:
                sections.append((current_title, '\n'.join(current_content).strip()))
                current_content = []
            
            # Start new section
            current_title = header_match.group(2).strip()
        else:
            current_content.append(line)
    
    # Add the last section
    if current_content:
        sections.append((current_title, '\n'.join(current_content).strip()))
    
    return sections


def convert_to_solr_docs(sections: List[Tuple[str, str]], filename: str, metadata: Dict) -> List[Dict]:
    """
    Convert markdown sections to Solr documents.
    
    Args:
        sections: List of (title, content) tuples
        filename: Original filename
        metadata: Metadata from frontmatter
        
    Returns:
        List of documents ready for Solr indexing
    """
    documents = []
    
    for i, (title, content) in enumerate(sections):
        # Skip empty sections
        if not content.strip():
            continue
            
        doc = {
            "id": f"{os.path.basename(filename)}_section_{i}",
            "title": title,
            "text": content,
            "source": filename,
            "section_number": i,
            "date_indexed": datetime.now().isoformat(),
            "tags": metadata.get("tags", []),
            "category": metadata.get("categories", [])
        }
        
        # Add any additional metadata
        for key, value in metadata.items():
            if key not in ["tags", "categories"] and key not in doc:
                doc[key] = value
        
        documents.append(doc)
    
    return documents


def process_markdown_file(file_path: str, output_file: str = None):
    """
    Process a markdown file, splitting it into sections and converting to Solr documents.
    
    Args:
        file_path: Path to the markdown file
        output_file: Path to save the JSON output (if None, prints to stdout)
    """
    # Read and parse markdown with frontmatter
    with open(file_path, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)
    
    # Extract frontmatter metadata and content
    metadata = dict(post.metadata)
    content = post.content
    
    # Extract sections
    sections = extract_sections(content)
    
    # Convert to Solr documents
    documents = convert_to_solr_docs(sections, file_path, metadata)
    
    # Output
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2)
        print(f"Processed {file_path} into {len(documents)} sections and saved to {output_file}")
    else:
        print(json.dumps(documents, indent=2))
        print(f"Processed {file_path} into {len(documents)} sections", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process markdown files for Solr indexing")
    parser.add_argument("file", help="Path to the markdown file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    
    args = parser.parse_args()
    
    process_markdown_file(args.file, args.output)