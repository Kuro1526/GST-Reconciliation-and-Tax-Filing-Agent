import os
import re
import pandas as pd
from datetime import datetime
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

class AdvancedInvoiceExtractor:
    def __init__(self):
        self.keywords = {
            'invoice_from': ['from', 'seller', 'vendor', 'supplier', 'issued by', 'provider'],
            'invoice_to': ['to', 'bill to', 'customer', 'client', 'recipient', 'buyer', 'ship to'],
            'address': ['address', 'street', 'city', 'state', 'zip', 'country', 'location'],
            'invoice_number': ['invoice no', 'invoice number', 'invoice#', 'inv no', 'inv#', 'doc no'],
            'invoice_date': ['invoice date', 'date', 'issue date', 'billing date'],
            'due_date': ['due date', 'payment due', 'due by'],
            'salesperson': ['salesperson', 'sales rep', 'account manager', 'representative', 'contact person'],
            'total_amount': ['total', 'amount due', 'balance due', 'grand total', 'subtotal'],
        }
        
    def get_supported_files(self, folder_path):
        """Get all supported invoice files from the folder"""
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.txt']
        invoice_files = []
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Input folder '{folder_path}' does not exist. Creating it...")
            folder.mkdir(parents=True, exist_ok=True)
            return invoice_files
        
        for ext in supported_extensions:
            invoice_files.extend(folder.glob(f"*{ext}"))
            invoice_files.extend(folder.glob(f"*{ext.upper()}"))
        
        return sorted(invoice_files)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
        return text
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        try:
            # Preprocess image for better OCR
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Could not read image: {image_path}")
                return ""
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Save temporary image
            temp_path = "temp_processed.png"
            cv2.imwrite(temp_path, thresh)
            
            # Perform OCR
            text = pytesseract.image_to_string(Image.open(temp_path))
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return text
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return ""
    
    def extract_text(self, file_path):
        """Extract text from various file types"""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return self.extract_text_from_image(file_path)
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
        else:
            print(f"Unsupported file type: {file_ext}")
            return ""
    
    def find_value_by_keywords(self, text, keyword_list):
        """Find values based on keyword patterns"""
        text_lower = text.lower()
        lines = text.split('\n')
        
        for keyword in keyword_list:
            pattern = rf"{keyword}[:\s]*([^\n]+)"
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # Alternative approach: look for lines containing keywords
        for line in lines:
            line_lower = line.lower()
            for keyword in keyword_list:
                if keyword in line_lower:
                    # Remove the keyword and clean up
                    value = re.sub(keyword, '', line_lower, flags=re.IGNORECASE)
                    value = re.sub(r'^[:\s\-]*', '', value).strip()
                    if value:
                        return value
        return ""
    
    def extract_invoice_number(self, text):
        """Extract invoice number using multiple strategies"""
        # Strategy 1: Look for common invoice number patterns
        patterns = [
            r'invoice\s*(?:no|number|#)?\s*[:\-]?\s*([a-zA-Z0-9\-]+)',
            r'inv\s*(?:no|number|#)?\s*[:\-]?\s*([a-zA-Z0-9\-]+)',
            r'invoice\s+([a-zA-Z0-9\-]+)',
            r'inv\s+([a-zA-Z0-9\-]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # Strategy 2: Look for standalone invoice numbers (like "INVOICE 0012820")
        standalone_pattern = r'INVOICE\s+([A-Z0-9\-]+)'
        matches = re.findall(standalone_pattern, text)
        if matches:
            return matches[0].strip()
        
        return ""
    
    def extract_address(self, text):
        """Extract address information"""
        # Look for address patterns
        address_patterns = [
            r'\d+\s+[\w\s]+\s+(?:street|st|avenue|ave|road|rd|lane|ln|boulevard|blvd)',
            r'[A-Za-z\s]+,\s*[A-Za-z]+\s*\d{5}',
            r'P\.?O\.?\s*Box\s*\d+',
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return self.find_value_by_keywords(text, self.keywords['address'])
    
    def detect_table_format(self, lines):
        """Detect the table format type based on content"""
        text = '\n'.join(lines)
        
        # Format 1: Standard pipe table with quantity, description, unit price, total
        if '| QUANTITY    | DESCRIPTION    | UNIT PRICE  | TOTAL    |' in text:
            return "format1_standard_pipe"
        
        # Format 2: Product with quantity on separate lines
        if '| Product    | Unit Price [EUR] | Total [EUR] |' in text and 'Qty.' in text:
            return "format2_product_separate_qty"
        
        # Format 3: Two-column layout with items on left and prices on right
        if 'Item # Ordered Service' in text and 'Item Price Total' in text:
            return "format3_two_column"
        
        # Format 4: Numbered list with product details
        if re.search(r'\d+\.\s+[A-Za-z]', text) and 'pcs.' in text:
            return "format4_numbered_list"
        
        # Format 5: Service description with quantity and amount
        if 'Service Description' in text and 'quantity' in text.lower() and 'Total Amount' in text:
            return "format5_service_table"
        
        return "unknown"
    
    def parse_format1_standard_pipe(self, lines, invoice_number, file_name):
        """Parse Format 1: Standard pipe table"""
        table_data = []
        in_table = False
        
        for line in lines:
            line_clean = line.strip()
            
            if '| QUANTITY    | DESCRIPTION    | UNIT PRICE  | TOTAL    |' in line:
                in_table = True
                continue
            
            if in_table and line_clean and '|' in line_clean:
                # Skip separator lines
                if re.match(r'^[\|\-\s]+$', line_clean):
                    continue
                
                parts = [part.strip() for part in line_clean.split('|') if part.strip()]
                if len(parts) >= 4 and parts[0].isdigit():
                    table_data.append({
                        'invoice_number': invoice_number,
                        'file_name': file_name,
                        'description': parts[1],
                        'quantity': parts[0],
                        'unit_price': parts[2],
                        'total': parts[3]
                    })
        
        return table_data
    
    def parse_format2_product_separate_qty(self, lines, invoice_number, file_name):
        """Parse Format 2: Product with quantity on separate lines"""
        table_data = []
        current_product = {}
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            if '| Product    | Unit Price [EUR] | Total [EUR] |' in line:
                continue
            
            if line_clean and '|' in line_clean and not any(x in line_clean for x in ['Subtotal', 'Sales Tax', 'Total Due']):
                parts = [part.strip() for part in line_clean.split('|') if part.strip()]
                
                if len(parts) == 3 and not parts[0].startswith('Qty.'):
                    # This is a product line
                    current_product = {
                        'description': parts[0],
                        'unit_price': parts[1],
                        'total': parts[2]
                    }
                elif len(parts) == 1 and parts[0].startswith('Qty.'):
                    # This is a quantity line
                    qty_match = re.search(r'Qty\.\s*(\d+)', parts[0])
                    if qty_match and current_product:
                        current_product['quantity'] = qty_match.group(1)
                        current_product['invoice_number'] = invoice_number
                        current_product['file_name'] = file_name
                        table_data.append(current_product.copy())
                        current_product = {}
        
        return table_data
    
    def parse_format3_two_column(self, lines, invoice_number, file_name):
        """Parse Format 3: Two-column layout"""
        table_data = []
        items_section = []
        prices_section = []
        in_items = False
        in_prices = False
        
        for line in lines:
            line_clean = line.strip()
            
            if 'Item # Ordered Service' in line:
                in_items = True
                in_prices = False
                continue
            elif 'Item Price Total' in line:
                in_items = False
                in_prices = True
                continue
            elif 'Please contact' in line:
                break
            
            if in_items and line_clean:
                items_section.append(line_clean)
            elif in_prices and line_clean:
                prices_section.append(line_clean)
        
        # Match items with prices
        for i, (item_line, price_line) in enumerate(zip(items_section, prices_section)):
            # Parse item line (format: "1 10-700 - Exterior Protection (10)")
            item_parts = item_line.split()
            if len(item_parts) >= 2:
                description = ' '.join(item_parts[1:])
                quantity_match = re.search(r'\((\d+)\)', description)
                quantity = quantity_match.group(1) if quantity_match else ""
                
                # Clean description
                description = re.sub(r'\(\d+\)', '', description).strip()
                
                # Parse price line (format: "40.29 402.9")
                price_parts = price_line.split()
                if len(price_parts) >= 2:
                    table_data.append({
                        'invoice_number': invoice_number,
                        'file_name': file_name,
                        'description': description,
                        'quantity': quantity,
                        'unit_price': price_parts[0],
                        'total': price_parts[1]
                    })
        
        return table_data
    
    def parse_format4_numbered_list(self, lines, invoice_number, file_name):
        """Parse Format 4: Numbered list with product details"""
        table_data = []
        
        for i in range(len(lines)):
            line = lines[i].strip()
            
            # Look for numbered items
            numbered_match = re.match(r'(\d+)\.\s+(.+)$', line)
            if numbered_match:
                item_num = numbered_match.group(1)
                description_start = numbered_match.group(2)
                
                # Look for the next line with quantity and price information
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    
                    # Pattern: "822-79-9581 5 pcs. € 200 € 1000"
                    qty_price_match = re.search(r'(\d+)\s+pcs\.\s+[€€]?\s*([\d.,]+)\s+[€€]?\s*([\d.,]+)', next_line)
                    if qty_price_match:
                        sku = next_line.split()[0] if next_line.split() else ""
                        quantity = qty_price_match.group(1)
                        unit_price = qty_price_match.group(2)
                        total = qty_price_match.group(3)
                        
                        full_description = f"{description_start} {sku}".strip()
                        
                        table_data.append({
                            'invoice_number': invoice_number,
                            'file_name': file_name,
                            'description': full_description,
                            'quantity': quantity,
                            'unit_price': unit_price,
                            'total': total
                        })
        
        return table_data
    
    def parse_format5_service_table(self, lines, invoice_number, file_name):
        """Parse Format 5: Service description table"""
        table_data = []
        in_table = False
        
        for line in lines:
            line_clean = line.strip()
            
            if 'Service Description' in line and 'quantity' in line.lower():
                in_table = True
                continue
            
            if in_table and line_clean and '|' in line_clean:
                # Skip total lines
                if any(x in line_clean for x in ['Total', 'VAT', 'Gross Amount']):
                    continue
                
                parts = [part.strip() for part in line_clean.split('|') if part.strip()]
                if len(parts) >= 4:
                    # Extract numbers from currency strings
                    amount_match = re.search(r'([\d.,]+)', parts[1])
                    unit_price = amount_match.group(1) if amount_match else ""
                    
                    total_match = re.search(r'([\d.,]+)', parts[3])
                    total = total_match.group(1) if total_match else ""
                    
                    table_data.append({
                        'invoice_number': invoice_number,
                        'file_name': file_name,
                        'description': parts[0],
                        'quantity': parts[2],
                        'unit_price': unit_price,
                        'total': total
                    })
        
        return table_data
    
    def extract_table_data(self, text, invoice_number, file_name):
        """Extract table data based on detected format"""
        lines = text.split('\n')
        
        # Detect format
        format_type = self.detect_table_format(lines)
        print(f"  Detected format: {format_type}")
        
        # Parse based on format
        if format_type == "format1_standard_pipe":
            return self.parse_format1_standard_pipe(lines, invoice_number, file_name)
        elif format_type == "format2_product_separate_qty":
            return self.parse_format2_product_separate_qty(lines, invoice_number, file_name)
        elif format_type == "format3_two_column":
            return self.parse_format3_two_column(lines, invoice_number, file_name)
        elif format_type == "format4_numbered_list":
            return self.parse_format4_numbered_list(lines, invoice_number, file_name)
        elif format_type == "format5_service_table":
            return self.parse_format5_service_table(lines, invoice_number, file_name)
        else:
            print(f"  Unknown format, using fallback parser")
            return self.fallback_table_parser(lines, invoice_number, file_name)
    
    def fallback_table_parser(self, lines, invoice_number, file_name):
        """Fallback parser for unknown formats"""
        table_data = []
        
        for line in lines:
            line_clean = line.strip()
            
            # Look for lines with numbers that might be table rows
            numbers = re.findall(r'[\d.,]+', line_clean)
            if len(numbers) >= 2 and len(line_clean) > 10:
                # Try to extract description (text before first number)
                text_before_first_num = re.split(r'[\d.,]+', line_clean)[0].strip()
                
                if text_before_first_num:
                    table_data.append({
                        'invoice_number': invoice_number,
                        'file_name': file_name,
                        'description': text_before_first_num,
                        'quantity': numbers[0] if len(numbers) > 0 else '',
                        'unit_price': numbers[1] if len(numbers) > 1 else '',
                        'total': numbers[2] if len(numbers) > 2 else numbers[1] if len(numbers) > 1 else ''
                    })
        
        return table_data
    
    def extract_invoice_data(self, file_path):
        """Extract all required information from invoice"""
        text = self.extract_text(file_path)
        if not text:
            print(f"Could not extract text from {file_path}")
            return None
        
        # Extract basic info
        invoice_from = self.find_value_by_keywords(text, self.keywords['invoice_from'])
        invoice_to = self.find_value_by_keywords(text, self.keywords['invoice_to'])
        address = self.extract_address(text)
        invoice_number = self.extract_invoice_number(text)
        salesperson = self.find_value_by_keywords(text, self.keywords['salesperson'])
        
        # If invoice number not found, generate one from filename
        if not invoice_number:
            invoice_number = f"INV_{file_path.stem}"
        
        # Extract table data using format-specific parsers
        table_data = self.extract_table_data(text, invoice_number, file_path.name)
        
        invoice_data = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'invoice_from': invoice_from,
            'invoice_to': invoice_to,
            'address': address,
            'invoice_number': invoice_number,
            'salesperson': salesperson,
            'table_data_count': len(table_data),
            'raw_text_preview': text[:300]
        }
        
        return invoice_data, table_data
    
    def save_to_excel(self, invoice_data_list, table_data_list, excel_file='invoice_data.xlsx'):
        """Save extracted data to Excel file with multiple sheets"""
        if not invoice_data_list:
            print("No data to save.")
            return None, None
        
        # Create DataFrames
        df_invoices = pd.DataFrame(invoice_data_list)
        
        # Combine all table data
        all_table_data = []
        for table_data in table_data_list:
            all_table_data.extend(table_data)
        
        df_tables = pd.DataFrame(all_table_data) if all_table_data else pd.DataFrame()
        
        # Check if file exists
        if os.path.exists(excel_file):
            try:
                with pd.ExcelFile(excel_file) as xls:
                    if 'Invoices' in xls.sheet_names:
                        df_existing_invoices = pd.read_excel(xls, 'Invoices')
                    else:
                        df_existing_invoices = pd.DataFrame()
                    
                    if 'Table_Data' in xls.sheet_names:
                        df_existing_tables = pd.read_excel(xls, 'Table_Data')
                    else:
                        df_existing_tables = pd.DataFrame()
            except:
                df_existing_invoices = pd.DataFrame()
                df_existing_tables = pd.DataFrame()
            
            # Avoid duplicates for invoices
            if not df_existing_invoices.empty:
                existing_keys = set(zip(
                    df_existing_invoices['file_name'].astype(str), 
                    df_existing_invoices['invoice_number'].astype(str)
                ))
                new_keys = set(zip(
                    df_invoices['file_name'].astype(str), 
                    df_invoices['invoice_number'].astype(str)
                ))
                
                new_invoice_keys = new_keys - existing_keys
                if new_invoice_keys:
                    mask = df_invoices.apply(
                        lambda row: (str(row['file_name']), str(row['invoice_number'])) in new_invoice_keys, 
                        axis=1
                    )
                    df_invoices_filtered = df_invoices[mask]
                    df_invoices_combined = pd.concat([df_existing_invoices, df_invoices_filtered], ignore_index=True)
                    print(f"Added {len(df_invoices_filtered)} new invoices to existing Excel file.")
                else:
                    print("All invoices have already been processed.")
                    df_invoices_combined = df_existing_invoices
            else:
                df_invoices_combined = df_invoices
            
            # Avoid duplicates for table data
            if not df_existing_tables.empty and not df_tables.empty:
                existing_table_keys = set(zip(
                    df_existing_tables['invoice_number'].astype(str),
                    df_existing_tables['file_name'].astype(str),
                    df_existing_tables['description'].astype(str)
                ))
                
                new_table_keys = set(zip(
                    df_tables['invoice_number'].astype(str),
                    df_tables['file_name'].astype(str),
                    df_tables['description'].astype(str)
                ))
                
                new_table_key_set = new_table_keys - existing_table_keys
                if new_table_key_set:
                    mask = df_tables.apply(
                        lambda row: (str(row['invoice_number']), str(row['file_name']), str(row['description'])) in new_table_key_set,
                        axis=1
                    )
                    df_tables_filtered = df_tables[mask]
                    df_tables_combined = pd.concat([df_existing_tables, df_tables_filtered], ignore_index=True)
                    print(f"Added {len(df_tables_filtered)} new table rows to existing Excel file.")
                else:
                    print("All table data has already been processed.")
                    df_tables_combined = df_existing_tables
            else:
                df_tables_combined = df_tables if not df_tables.empty else pd.DataFrame()
                
        else:
            df_invoices_combined = df_invoices
            df_tables_combined = df_tables
            print(f"Created new Excel file with {len(df_invoices)} invoices and {len(df_tables)} table rows.")
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_invoices_combined.to_excel(writer, sheet_name='Invoices', index=False)
            if not df_tables_combined.empty:
                # Ensure consistent column order for table data
                column_order = ['invoice_number', 'file_name', 'description', 'quantity', 'unit_price', 'total']
                existing_columns = [col for col in column_order if col in df_tables_combined.columns]
                df_tables_combined = df_tables_combined[existing_columns]
                df_tables_combined.to_excel(writer, sheet_name='Table_Data', index=False)
        
        print(f"Data saved to {excel_file}")
        print(f"  - Invoices sheet: {len(df_invoices_combined)} records")
        print(f"  - Table_Data sheet: {len(df_tables_combined)} records")
        
        return df_invoices_combined, df_tables_combined

def main():
    extractor = AdvancedInvoiceExtractor()
    
    # Define input and output paths
    input_folder = "inputs"
    output_file = "invoice_data.xlsx"
    
    # Get all invoice files from inputs folder
    invoice_files = extractor.get_supported_files(input_folder)
    
    if not invoice_files:
        print(f"No supported invoice files found in '{input_folder}' folder.")
        print("Supported formats: PDF, JPG, JPEG, PNG, TIFF, BMP, TXT")
        print(f"Please add your invoice files to the '{input_folder}' folder and run the script again.")
        return
    
    print(f"Found {len(invoice_files)} invoice files in '{input_folder}' folder:")
    for file in invoice_files:
        print(f"  - {file.name}")
    
    print("\nStarting advanced extraction...")
    print("=" * 60)
    
    extracted_invoice_data = []
    extracted_table_data = []
    successful_extractions = 0
    
    for file_path in invoice_files:
        print(f"\nProcessing: {file_path.name}...")
        result = extractor.extract_invoice_data(file_path)
        
        if result:
            invoice_data, table_data = result
            extracted_invoice_data.append(invoice_data)
            extracted_table_data.append(table_data)
            successful_extractions += 1
            
            print("✓ Extracted data:")
            print(f"  Invoice #: {invoice_data['invoice_number']}")
            print(f"  From: {invoice_data['invoice_from'] or 'Not found'}")
            print(f"  To: {invoice_data['invoice_to'] or 'Not found'}")
            print(f"  Table rows: {len(table_data)}")
            
            # Show sample table data
            if table_data:
                print("  Sample table data:")
                for i, row in enumerate(table_data[:2]):
                    print(f"    - {row['description'][:30]}... | Qty: {row['quantity']} | Price: {row['unit_price']} | Total: {row['total']}")
        else:
            print(f"✗ Failed to extract data from {file_path.name}")
    
    print("\n" + "=" * 60)
    print(f"Extraction completed: {successful_extractions}/{len(invoice_files)} files processed successfully")
    
    # Save all extracted data to Excel
    if extracted_invoice_data:
        invoices_df, tables_df = extractor.save_to_excel(extracted_invoice_data, extracted_table_data, output_file)
        
        # Display summary
        print(f"\nFinal Summary:")
        print(f"Total invoices in database: {len(invoices_df)}")
        print(f"Total table rows in database: {len(tables_df) if not tables_df.empty else 0}")
        print(f"Unique invoice numbers: {invoices_df['invoice_number'].nunique()}")
        
    else:
        print("No data was extracted from any files.")

if __name__ == "__main__":
    main()