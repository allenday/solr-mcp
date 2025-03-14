#!/usr/bin/env python3
"""
Script to prepare data for indexing in Solr with dynamic field naming conventions.
"""

import argparse
import json
import sys
import os
from datetime import datetime

def prepare_data_for_solr(input_file, output_file):
    """
    Modify field names to use Solr dynamic field naming conventions.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file
    """
    # Load the input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Transform the data
    transformed_data = []
    for doc in data:
        transformed_doc = {}
        
        # Map fields to appropriate dynamic field suffixes
        for key, value in doc.items():
            if key == 'id' or key == 'title' or key == 'text' or key == 'source':
                # Keep standard fields as they are
                transformed_doc[key] = value
            elif key == 'section_number':
                # Integer fields get _i suffix
                transformed_doc['section_number_i'] = value
            elif key == 'date_indexed':
                # Date fields get _dt suffix and need proper Solr format
                # Convert to Solr format YYYY-MM-DDThh:mm:ssZ
                # If already a string, ensure it's in the right format
                if isinstance(value, str):
                    # Truncate microseconds if present 
                    if '.' in value:
                        parts = value.split('.')
                        value = parts[0] + 'Z'
                    elif not value.endswith('Z'):
                        value = value + 'Z'
                transformed_doc[f'{key}_dt'] = value
            elif key == 'date':
                # Ensure date has proper format
                if isinstance(value, str):
                    # If just a date (YYYY-MM-DD), add time
                    if len(value) == 10 and value.count('-') == 2:
                        value = value + 'T00:00:00Z'
                    # If it has time but no Z, add Z
                    elif 'T' in value and not value.endswith('Z'):
                        value = value + 'Z'
                transformed_doc[f'{key}_dt'] = value
            elif key == 'tags' or key == 'category':
                # Multi-valued string fields get _ss suffix
                transformed_doc[f'{key}_ss'] = value
            elif key == 'author':
                # String fields get _s suffix
                transformed_doc[f'{key}_s'] = value
            else:
                # Default: keep as is
                transformed_doc[key] = value
        
        transformed_data.append(transformed_doc)
    
    # Write the transformed data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2)
    
    print(f"Prepared {len(transformed_data)} documents for Solr indexing")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Solr indexing")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("--output", "-o", default=None, help="Path to the output JSON file")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        input_name = os.path.basename(args.input_file)
        name, ext = os.path.splitext(input_name)
        args.output = f"data/processed/{name}_solr{ext}"
    
    prepare_data_for_solr(args.input_file, args.output)