import re
from typing import Dict, List, Tuple, Optional, Set

from tskg import KPIGraphBuilder, KPINode

class KPIExtractor:
    """Extracts KPIs from table data and builds semantic understanding"""
    
    def __init__(self):
        self.kpi_patterns = {
            'sales_revenue': ['sales revenue', 'revenue', 'turnover'],
            'vehicle_sales': ['vehicle sales', 'unit sales', 'sales'],
            'operating_result': ['operating result', 'operating profit', 'profit'],
            'production': ['production', 'produced', 'manufactured', 'units'],
            'deliveries': ['deliveries', 'delivered', 'thousand units'],
            'return_on_sales': ['return on sales', 'operating return', 'margin'],
            'key_figures': ['key figures', 'key metrics'],
            'general_metric': ['metric', 'figure', 'data'],
            # Specific automotive KPIs
            'vehicle_production': ['vehicle production', 'car production', 'auto production'],
            'brand_sales': ['brand sales', 'model sales'],
            'financial_performance': ['financial performance', 'financial results']
        }
        builder = KPIGraphBuilder()
        self.graph = builder.graph

    def extract_year_from_header(self, header: str) -> Optional[int]:
        """Extract year from merged header like 'SALES REVENUE 2022'"""
        year_match = re.search(r'\b(20\d{2})\b', header)
        return int(year_match.group(1)) if year_match else None
    
    def extract_kpi_name(self, header: str) -> str:
        """Extract KPI name from header, removing year"""
        # Remove year pattern
        kpi_name = re.sub(r'\s*(20\d{2})\s*$', '', header)
        return kpi_name.strip().lower() if kpi_name else None
    
    def normalize_kpi_name(self, kpi_name: str) -> str:
        """Normalize KPI names for consistency"""
        kpi_name_lower = kpi_name.lower()
        for normalized, patterns in self.kpi_patterns.items():
            for pattern in patterns:
                if pattern in kpi_name_lower:
                    return normalized
        return kpi_name_lower
    
    def extract_units(self, header: str) -> str:
        """Extract measurement units from header"""
        unit_match = re.search(r'\((.*?)\)', header)
        return unit_match.group(1).strip() if unit_match else None
    
    def detect_table_format(self, merged_headers: List[str]) -> str:
        """Detect the table format type based on the three patterns"""

        # Type 1: Headers contain "KPI_NAME YEAR" (e.g., "SALES REVENUE 2022")
        # First header contains measurement units (thousands, millions, etc.), subsequent headers have KPI + year
        has_kpi_year_headers = all(
            self.extract_kpi_name(header) != None and self.extract_year_from_header(header) != None and
            len(header.split()) > 1 
            for header in merged_headers[1:] if header
        )
        
        # Type 2: Headers are just years, KPI in title, measurement units in first header
        # merged_headers = ["Units", "2023", "2022"] or ["Thousands", "2023", "2022"]
        has_year_only_with_units_header = (
            not self.is_string_number(merged_headers[0]) and self.is_word_string(merged_headers[0])
        )

        # Type 3: Headers are just years, KPI in first column of each row, measurement units embedded in row data
        # merged_headers = ["", "", "2022", "2021", "%"]
        # Units can be in row text itself or in second column
        #print(f"Detecting table format for title: {title}, merged_headers: {merged_headers[0].strip() == ''}, merged_headers: {merged_headers[1].strip() == ''}")

        has_empty_first_header_with_years = (
            ((merged_headers[0].strip() == "" and merged_headers[1].strip() == ""))
        )
        
      # Type 4: Empty first header, years in subsequent headers, KPI context in title, individual KPIs in rows with units in parentheses
        # merged_headers = ["", "2023", "2022", "%"] - Empty first header, then years
        # title contains main KPI context like "VOLKSWAGEN COMMERCIAL VEHICLES – KEY FIGURES"
        # rows like ["Deliveries (thousand units)", "409", "329", "24.6"]
        type_4_condition = (
            merged_headers[0].strip() == "" and self.is_string_number(merged_headers[1]) 
        )


        
        if has_kpi_year_headers and not has_empty_first_header_with_years and not has_year_only_with_units_header and not type_4_condition:
            return "type_1"  # Measurement units (thousands/millions) in first header + KPI & Year in subsequent headers
        elif has_year_only_with_units_header and not has_empty_first_header_with_years and not has_kpi_year_headers and not type_4_condition:
            return "type_2"  # Years only in headers, KPI in title, measurement units in first header
        elif has_empty_first_header_with_years and not has_kpi_year_headers and not has_year_only_with_units_header and not type_4_condition:
            return "type_3"  # Years only in headers, KPI in first column of rows, units embedded in row text
        elif type_4_condition:
            return "type_4"  # Years in headers, KPI context in title, individual KPIs in rows with units in parentheses
        else:
            return "unknown_format"
        
    def extract_kpis_from_table(self, table_data: Dict) -> List[KPINode]:
        """Extract all KPIs from a single table with format detection"""
        nodes = []
        merged_headers = table_data.get('merged_headers', [])
        # Detect table format
        table_format = self.detect_table_format(merged_headers)

        if table_format == "type_1":
            # Type 1: Measurement units (thousands/millions) in first header + KPI & Year in subsequent headers
            nodes.extend(self._extract_type_1_format(table_data))
        elif table_format == "type_2":
            # Type 2: Years only in headers, KPI in title, measurement units in first header
            nodes.extend(self._extract_type_2_format(table_data))
        elif table_format == "type_3":
            # Type 3: Years only in headers, KPI in first column of rows, units embedded in row text
            nodes.extend(self._extract_type_3_format(table_data))
        elif table_format == "type_4":
            # Type 4: Years in headers, KPI context in title, individual KPIs in rows with units in parentheses
            nodes.extend(self._extract_type_4_format(table_data))
        else:
            # Fallback to type 1 method
            nodes.extend(self._extract_type_1_format(table_data))

        return nodes

    def _extract_type_1_format(self, table_data: Dict) -> List[KPINode]:
        """Extract KPIs from Type 1: Measurement units (thousands/millions) in first header + KPI & Year in subsequent headers
        Example: ['Thousand vehicles/€ million', 'VEHICLE SALES 2023', 'VEHICLE SALES 2022', ...]
        """
        nodes = []
        merged_headers = table_data.get('merged_headers', [])
        rows = table_data.get('rows', [])
        stub_col = table_data.get('stub_col', [])
        
        # Skip first header (units/description)
        data_headers = merged_headers[1:] if len(merged_headers) > 1 else []
        
        # Extract measurement units (thousands, millions, etc.) from first header - can be multiple units
        units_list = self.extract_units_from_header(merged_headers)
        for row_idx, row in enumerate(rows):
            if row_idx >= len(stub_col):
                continue
                
            key = stub_col[row_idx].strip()
            if not key or key == "": 
                continue
            
            # Skip data values (first column is usually units)
            data_values = row[1:] if len(row) > 1 else []
            prevYear = None
            year_col = 0
            last_kpi = None
            for col_idx, value in enumerate(data_values):
                if col_idx >= len(data_headers):
                    continue
                
                header = data_headers[col_idx]
                if not self.is_valid_value(value):
                    continue
                
                year = self.extract_year_from_header(header)
                if not year:
                    continue
                
                kpi_name = self.extract_kpi_name(header)
            
                cleaned_value = self.clean_value(value)
                
                # Calculate actual column position (add 1 for the skipped first column)
                actual_col_idx = col_idx + 1    

                if(prevYear and prevYear != year and kpi_name == last_kpi):
                    column_units = self.get_units_for_column(year_col, units_list)
                    year_col += 1
                else:
                    column_units = self.get_units_for_column(col_idx, units_list)
                last_kpi = kpi_name    
                prevYear = year                   

                nodes.append(self._create_kpi_node(
                    kpi_name, key, cleaned_value, year, table_data,
                    row_idx=row_idx, col_idx=actual_col_idx, header=header, row_data=row,
                    table_units=column_units
                ))
        
        return nodes
    
    def _extract_type_2_format(self, table_data: Dict) -> List[KPINode]:
        """Extract KPIs from Type 2: Years only in headers, KPI in title, measurement units in first header
        Example: merged_headers = ['Units', '2023', '2022'], title = 'PRODUCTION'
        """
        nodes = []
        merged_headers = table_data.get('merged_headers', [])
        rows = table_data.get('rows', [])
        stub_col = table_data.get('stub_col', [])
        title = table_data.get('title', '')
        
        # Extract KPI from title
        base_kpi = self.infer_kpi_from_title(title)
        
        # Extract measurement units (thousands, millions, etc.) from first header
        table_units = merged_headers[0] if len(merged_headers) > 0 else 'N/A'
        
        # Skip first header (units) and get year columns
        data_headers = merged_headers[1:] if len(merged_headers) > 1 else []
        
        # Filter out percentage columns and get year columns
        year_columns = []
        for idx, header in enumerate(data_headers):
            if header.strip() == '%' or 'percent' in header.lower():
                continue  # Skip percentage columns
            year = self.extract_year_from_header(header)
            if year:
                year_columns.append((idx, year))
        
        for row_idx, row in enumerate(rows):
            if row_idx >= len(stub_col):
                continue
                
            key = stub_col[row_idx].strip()
            if not key or key == "":  # Skip empty keys and section headers
                continue
            
            # Skip data values (first column is usually units/description)
            data_values = row[1:] if len(row) > 1 else []
            
            for col_idx, year in year_columns:
                if col_idx >= len(data_values):
                    continue
                
                value = data_values[col_idx]
                if not self.is_valid_value(value):
                    continue
                
                cleaned_value = self.clean_value(value)
                
                # Get the original header text for this column
                header = data_headers[col_idx] if col_idx < len(data_headers) else str(year)
                # Calculate actual column position (add 1 for the skipped first column)
                actual_col_idx = col_idx + 1
                
                nodes.append(self._create_kpi_node(
                    base_kpi, key, cleaned_value, year, table_data,
                    row_idx=row_idx, col_idx=actual_col_idx, header=header, row_data=row,
                    table_units=table_units
                ))
        
        return nodes
    
    def _extract_type_3_format(self, table_data: Dict) -> List[KPINode]:
        """Extract KPIs from Type 3: Years only in headers, KPI in first column of rows, measurement units embedded in row text
        Examples: 
        - merged_headers = ['', '2023', '2022', '%'], rows = [['Deliveries (thousand units)', '4867', '4563', '6.7'], ...]
        - merged_headers = ['', '', '2022', '2021', '%'], rows = [['Number of contracts', 'thousands', '2197', '2203', '-0.3'], ...]
        """
        nodes = []
        merged_headers = table_data.get('merged_headers', [])
        rows = table_data.get('rows', [])
        stub_col = table_data.get('stub_col', [])
        
        # Skip first header (empty) and get year columns
        data_headers = merged_headers[1:] if len(merged_headers) > 1 else []
        
        # Filter out percentage columns and get year columns
        year_columns = []
        for idx, header in enumerate(data_headers):
            if header.strip() == '%' or 'percent' in header.lower():
                continue  # Skip percentage columns
            year = self.extract_year_from_header(header)
            if year:
                year_columns.append((idx, year))
        
        for row_idx, row in enumerate(rows):
            if row_idx >= len(stub_col):
                continue
                
            key = stub_col[row_idx].strip()
            if not key or key == "":  # Skip empty keys and section headers
                continue
            
            # Extract KPI name and measurement units from the key (first column)
            kpi_info = self.extract_kpi_and_units_from_key(key)
            base_kpi = kpi_info['kpi']
            row_units = kpi_info['units']
            
            # Check if units are in second column (when first column doesn't have units in parentheses)
            if row_units == 'N/A' and len(row) > 1 and row[1].strip():
                second_col = row[1].strip()
                # If second column looks like units (not a number), use it as units
                if not self.is_valid_value(second_col) and second_col not in ["", "–", "-"]:
                    row_units = second_col
                    # Skip both first and second column for data values
                    data_values = row[2:] if len(row) > 2 else []
                    year_offset = 2  # Start looking for years from 3rd column
                else:
                    # Skip only first column for data values
                    data_values = row[1:] if len(row) > 1 else []
                    year_offset = 1  # Start looking for years from 2nd column
            else:
                # Skip only first column for data values
                data_values = row[1:] if len(row) > 1 else []
                year_offset = 1  # Start looking for years from 2nd column

            for col_idx, year in year_columns:
                # Adjust column index based on whether we skipped units column
                adjusted_col_idx = col_idx - (year_offset - 1)
                
                if adjusted_col_idx < 0 or adjusted_col_idx >= len(data_values):
                    continue
                
                value = data_values[adjusted_col_idx]
                if not self.is_valid_value(value):
                    continue
                
                cleaned_value = self.clean_value(value)
                
                # Get the original header text for this column
                header = data_headers[col_idx] if col_idx < len(data_headers) else str(year)
                # Calculate actual column position in original row
                actual_col_idx = col_idx + 1

                nodes.append(self._create_kpi_node(
                    base_kpi, key, cleaned_value, year, table_data,
                    row_idx=row_idx, col_idx=actual_col_idx, header=header, row_data=row,
                    table_units=row_units
                ))
        
        return nodes
    
    def _extract_type_4_format(self, table_data: Dict) -> List[KPINode]:
        """Extract KPIs from Type 4: Years in headers, KPI context in title, individual KPIs in rows with units in parentheses
        Example: merged_headers = ['', '2023', '2022', '%'], title = 'VOLKSWAGEN COMMERCIAL VEHICLES – KEY FIGURES'
        rows = [['Deliveries (thousand units)', '409', '329', '24.6'], ['Vehicle sales', '423', '340', '24.5'], ...]
        """
        nodes = []
        merged_headers = table_data.get('merged_headers', [])
        rows = table_data.get('rows', [])
        stub_col = table_data.get('stub_col', [])
        title = table_data.get('title', '')
        
        
        # Extract main KPI context from title (similar to Type 2 but more general)
        key = self.infer_kpi_context_from_title(title)
        
        # Skip first header (empty) and get year columns
        data_headers = merged_headers[1:] if len(merged_headers) > 1 else []
        
        # Filter out percentage columns and get year columns
        year_columns = []
        for idx, header in enumerate(data_headers):
            if header.strip() == '%' or 'percent' in header.lower():
                continue  # Skip percentage columns
            year = self.extract_year_from_header(header)
            if year:
                year_columns.append((idx, year))
        
        for row_idx, row in enumerate(rows):
            if row_idx >= len(stub_col):
                continue
                
            kpi = stub_col[row_idx].strip()
            if not kpi or kpi == "":  # Skip empty keys and section headers
                continue
            
            # Extract KPI name and measurement units from the key (first column)
            kpi_info = self.extract_kpi_and_units_from_key(kpi)
            kpi = kpi_info['kpi']  # This is the specific KPI like "deliveries", "sales_revenue"
            row_units = kpi_info['units']
            
            # Skip first column (contains KPI name) for data values
            data_values = row[1:] if len(row) > 1 else []
            
            for col_idx, year in year_columns:
                if col_idx >= len(data_values):
                    continue
                
                value = data_values[col_idx]
                if not self.is_valid_value(value):
                    continue
                
                cleaned_value = self.clean_value(value)
                
                # Get the original header text for this column
                header = data_headers[col_idx] if col_idx < len(data_headers) else str(year)
                # Calculate actual column position in original row
                actual_col_idx = col_idx + 1
                
                nodes.append(self._create_kpi_node(
                    kpi, key, cleaned_value, year, table_data,
                    row_idx=row_idx, col_idx=actual_col_idx, header=header, row_data=row,
                    table_units=row_units
                ))
        
        return nodes
    
    def infer_kpi_context_from_title(self, title: str) -> str:
        """Infer KPI context from table title for Type 4 format - extract part before '-' and convert spaces to underscores"""
        if not title:
            return 'general_metric'
        
        # Extract everything before the first "-" or "–"
        if '–' in title:
            kpi_part = title.split('–')[0].strip()
        elif '-' in title:
            kpi_part = title.split('-')[0].strip()
        else:
            kpi_part = title.strip()
        
        # Convert to lowercase and replace spaces with underscores
        kpi_context = kpi_part.lower().replace(' ', '_')
        
        # Remove any non-alphanumeric characters except underscores
        kpi_context = re.sub(r'[^a-zA-Z0-9_]', '', kpi_context)
        
        return kpi_context if kpi_context else 'general_metric'
    
    def infer_kpi_from_title(self, title: str) -> str:
        """Infer KPI name from table title for Type 2 format"""
        title_lower = title.lower()

        # Extract the main noun from title
        words = title_lower.replace('–', ' ').replace('-', ' ').split()
        # Filter out common words
        meaningful_words = [w for w in words if w not in ['the', 'a', 'an', 'of', 'by', 'in', 'for', 'with']]
        if meaningful_words:
            return '_'.join(meaningful_words[:2])  # Take first two meaningful words
        return 'general_metric'
    
    def extract_kpi_and_units_from_key(self, key: str) -> Dict[str, str]:
        """Extract KPI name and measurement units from key for Type 3 format
        Example: 'Deliveries (thousand units)' -> {'kpi': 'deliveries', 'units': 'thousand units'}
        """
        # Extract measurement units from parentheses (thousands, millions, etc.)
        units_match = re.search(r'\((.*?)\)', key)
        units = units_match.group(1).strip() if units_match else 'N/A'  
        # Remove units from key to get KPI name
        kpi_text = re.sub(r'\s*\(.*?\)\s*', '', key).strip()
        
        return {
            'kpi': kpi_text,
            'units': units
        }
    
    def extract_units_from_header(self, merged_headers: List[str]) -> List[str]:
        """Extract measurement units from first header and split for multiple columns if needed"""
        if not merged_headers:
            return ['N/A']
        
        first_header = merged_headers[0]
        
        # Try to extract from parentheses first
        units_match = re.search(r'\((.*?)\)', first_header)
        if units_match:
            units_text = units_match.group(1).strip()
        else:
            units_text = first_header.strip()
        
        # Handle mixed units separated by '/' like "Thousand vehicles/€ million" or "Thousand vehicles/€ million/%"
        if '/' in units_text:
            units_parts = [part.strip() for part in units_text.split('/')]
            return units_parts
        
        # Check for common measurement unit words
        header_lower = units_text.lower()
        if any(word in header_lower for word in ['unit', 'thousand', 'million', 'billion', '€', '$', '%']):
            return [units_text]
        
        return ['N/A']
    
    def get_units_for_column(self, col_idx: int, units_list: List[str]) -> str:
        """Determine which units to use for a specific column based on KPI name and column position"""  
        # Fallback: use column position to determine units
        if col_idx < len(units_list):
            return units_list[col_idx]
        else:
            return units_list[-1]  # Use last unit if column index exceeds units list
    
    def infer_kpi_from_context(self, table_data: Dict) -> str:
        """Infer KPI from table context (legacy method, replaced by infer_kpi_from_title)"""
        return self.infer_kpi_from_title(table_data.get('title', ''))
    
    def is_valid_value(self, value: str) -> bool:
        """Check if a value is valid for KPI extraction"""
        if not value or value.strip() == "":
            return False
        
        value = value.strip()
        
        # Skip common non-data values
        if value in ["–", "-", "N/A", "n/a", "x", ""]:
            return False
        
        # Check if it looks like a number (possibly with footnotes)
        # Remove footnote markers like ^1, ^2, etc.
        clean_value = re.sub(r'\s*\^\w*', '', value)
        clean_value = clean_value.replace(',', '').replace(' ', '')
        
        # Check if it's a number (including negative numbers)
        try:
            float(clean_value)
            return True
        except ValueError:
            return False
    
    def clean_value(self, value: str) -> float:
        """Clean and convert value to float"""
        if not value:
            return 0.0
        
        # Remove footnote markers and clean
        clean_value = re.sub(r'\s*\^\w*', '', str(value))
        clean_value = clean_value.replace(',', '').replace(' ', '').strip()
        
        # Handle negative values
        if clean_value.startswith('-'):
            try:
                return -float(clean_value[1:])
            except ValueError:
                return 0.0
        
        try:
            return float(clean_value)
        except ValueError:
            return 0.0
        
    def is_string(self, value: str) -> bool:
        """Check if a value is a non-empty string"""
        return isinstance(value, str) and value.strip() != ""
        
    def is_string_number(self, value: str) -> bool:
        """Check if a string represents a number (with better handling than is_valid_value)"""
        if not self.is_string(value) or not value or value.strip() == "":
            return False
        
        value = value.strip()
        
        # Skip obvious non-numeric values
        if value in ["–", "-", "N/A", "n/a", "x", "", "nan", "NaN", "null", "NULL"]:
            return False
        
        # Remove footnote markers like ^1, ^2, etc.
        clean_value = re.sub(r'\s*\^\w*', '', value)
        # Remove commas and spaces (for thousands separators)
        clean_value = clean_value.replace(',', '').replace(' ', '')
        
        # Check for percentage signs and remove them
        if clean_value.endswith('%'):
            clean_value = clean_value[:-1]
        
        # Handle negative numbers
        if clean_value.startswith('-') or clean_value.startswith('+'):
            clean_value = clean_value[1:]
        
        # Check if it's a valid number
        try:
            float(clean_value)
            return True
        except ValueError:
            return False

    def is_word_string(self, value: str) -> bool:
        """Check if a string contains only word characters (letters, spaces, hyphens, etc.) and is not a number"""
        if not self.is_string(value):
            return False
        
        # First check if it's a number - if so, it's not a word string
        if self.is_string_number(value):
            return False
        
        value = value.strip()
        if not value:
            return False
        
        # Check if string contains only word characters, spaces, hyphens, apostrophes, parentheses, and common punctuation
        # This pattern allows for typical text like "Operating Result", "Vehicle Sales", "€ million", "Customer financing", etc.
        # Added currency symbols (€, $, £, ¥) and other common unit symbols
        word_pattern = re.compile(r'^[a-zA-Z0-9\s\-\'&/\(\),\.%€$£¥°²³]+$')
        
        # Must contain at least one letter to be considered a "word string"
        has_letter = re.search(r'[a-zA-Z]', value)
        
        return bool(word_pattern.match(value) and has_letter)

    def _create_kpi_node(self, kpi_name: str, key: str, value: float, year: int,
                        table_data: Dict, row_idx: int = 0, col_idx: int = 0,
                        header: str = "", row_data: List = None, table_units: str = "N/A") -> KPINode:
        """Create a KPINode with all metadata"""
        return KPINode(
            kpi_name=kpi_name,
            key=key,
            value=value,
            year=year,
            units=table_units,
            evidence={
                'table_id': table_data.get('table_id', ''),
                'doc_id': table_data.get('doc_id', ''),
                'page': table_data.get('page', 0),
                'section': table_data.get('section_name', ''),
                'title': table_data.get('title', ''),
                'row_idx': row_idx,
                'col_idx': col_idx,
            },
        ) 