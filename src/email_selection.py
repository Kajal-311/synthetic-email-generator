import pandas as pd
import numpy as np
import re
import json
import random
import os
import glob
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("email_processing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)



def aggregate_csv_files(csv_dir, num_files=10):
    """
    Aggregate multiple CSV files from a directory
    
    Args:
        csv_dir: Directory containing CSV files
        num_files: Number of files to aggregate (default: 10)
        
    Returns:
        DataFrame containing emails from all the files
    """
    logger.info(f"Aggregating up to {num_files} CSV files from {csv_dir}...")
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    
    # Select a subset of files if there are more than requested
    if len(csv_files) > num_files:
        csv_files = random.sample(csv_files, num_files)
    
    logger.info(f"Processing {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    # Initialize an empty list to hold all dataframes
    all_dfs = []
    
    # Load each CSV file and append to the list
    for file_path in csv_files:
        try:
            logger.info(f"Loading {os.path.basename(file_path)}...")
            df = load_enron_dataset(file_path)
            if df is not None and not df.empty:
                all_dfs.append(df)
                logger.info(f"Successfully loaded {len(df)} emails from {os.path.basename(file_path)}")
            else:
                logger.warning(f"File {os.path.basename(file_path)} yielded no data")
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(file_path)}: {e}")
    
    if not all_dfs:
        raise ValueError("No valid data found in any of the CSV files")
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset contains {len(combined_df)} emails")
    
    return combined_df

def load_enron_dataset(filepath):
    """
    Load the Enron email dataset from CSV
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the emails
    """
    logger.info(f"Loading dataset from {filepath}...")
    try:
        # Try different encodings as the dataset might have mixed encodings
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    logger.info(f"Dataset loaded successfully. Total emails: {len(df)}")
    return df

def extract_headers_from_content(df):
    """
    Extract header information from content field
    
    Args:
        df: DataFrame with emails
        
    Returns:
        DataFrame with extracted headers
    """
    df_processed = df.copy()
    
    # Define patterns to extract header info - making them more flexible
    patterns = {
        'message_id': r'Message-ID:\s*<([^>]+)>',
        'date': r'Date:\s*([^\n]+)',
        'from': r'From:\s*([^\n]+)',
        'to': r'To:\s*([^\n]+)',
        'subject': r'Subject:\s*([^\n]+)'
    }
    
    # Process all rows where content exists
    for field, pattern in patterns.items():
        # Create the field if it doesn't exist
        if field not in df_processed.columns:
            df_processed[field] = None
            
        # Extract from all rows with content, not just those with empty fields
        mask = df_processed['content'].str.contains(pattern, regex=True, na=False)
        if any(mask):
            extracted = df_processed.loc[mask, 'content'].str.extract(pattern, expand=False)
            df_processed.loc[mask, field] = extracted
    
    return df_processed

def clean_dataset(df):
    """
    Clean and prepare the dataset for analysis
    
    Args:
        df: DataFrame with raw email data
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Rename columns if needed (adjust based on your actual CSV structure)
    column_mapping = {
        'Message-ID': 'message_id',
        'Date': 'date',
        'From': 'from',
        'To': 'to',
        'Subject': 'subject',
        'Content-Type': 'content_type',
        'X-Folder': 'folder',
        'X-Origin': 'origin',
        'X-FileName': 'filename'
    }
    
    # Only rename columns that exist
    existing_columns = [col for col in column_mapping if col in df_clean.columns]
    if existing_columns:
        df_clean = df_clean.rename(columns={col: column_mapping[col] for col in existing_columns})
    
    # Identify the column that contains the email body
    # It might be called 'Content', 'Body', or we might need to extract it
    content_columns = [col for col in df_clean.columns if 'content' in col.lower() or 'body' in col.lower() or 'message' in col.lower()]
    
    if content_columns:
        content_column = content_columns[0]
    else:
        # If no obvious content column, look for the column with the longest text on average
        text_lengths = {col: df_clean[col].astype(str).str.len().mean() 
                       for col in df_clean.columns if df_clean[col].dtype == object}
        content_column = max(text_lengths, key=text_lengths.get)
        logger.info(f"Using '{content_column}' as the email body column")
        
    # Rename the content column to 'content' for consistency
    if content_column != 'content':
        df_clean = df_clean.rename(columns={content_column: 'content'})
    
    # [NEW CODE] Extract headers from content if they're missing
    df_clean = extract_headers_from_content(df_clean)
    
    # Remove emails with missing or very short content
    df_clean = df_clean[df_clean['content'].notna()]
    df_clean = df_clean[df_clean['content'].astype(str).str.len() > 100]
    
    # Remove duplicate emails
    if 'message_id' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['message_id'])
    
    logger.info(f"Dataset cleaned. Remaining emails: {len(df_clean)}")
    return df_clean
    
def is_business_email(content, subject=''):
    """
    Determine if an email appears to be business-related based on content analysis
    
    Args:
        content: Email body text
        subject: Email subject
        
    Returns:
        Boolean indicating if the email is business-related
    """
    if not isinstance(content, str):
        return False
    
    subject = str(subject) if subject is not None else ''
    
    # Business term indicators
    business_terms = [
        'transaction', 'agreement', 'contract', 'deal', 'project',
        'report', 'meeting', 'schedule', 'financial', 'trading',
        'market', 'price', 'budget', 'forecast', 'analysis',
        'strategy', 'proposal', 'client', 'customer', 'approval',
        'confidential', 'deadline', 'deliverable', 'invoice', 'payment',
        'revenue', 'expense', 'profit', 'loss', 'presentation',
        'board', 'shareholder', 'legal', 'compliance', 'regulatory'
    ]
    
    # Combine subject and content for analysis
    combined_text = (content + " " + subject).lower()
    
    # Count business terms
    term_count = sum(1 for term in business_terms if term in combined_text)
    
    # Check for greeting patterns
    greeting_patterns = [
        r'dear \w+', r'hello \w+', r'hi \w+',
        r'^[a-z]+\s*,', r'good (morning|afternoon|evening)'
    ]
    has_greeting = any(re.search(pattern, combined_text) for pattern in greeting_patterns)
    
    # Check for closing patterns
    closing_patterns = [
        r'regards', r'sincerely', r'thank you', r'thanks',
        r'best( regards)?', r'cheers', r'yours truly',
        r'respectfully', r'cordially'
    ]
    has_closing = any(re.search(pattern, combined_text) for pattern in closing_patterns)
    
    # Check for forwarded email indicators
    forwarded = 'forwarded by' in combined_text.lower()
    
    # Business format check - needs both greeting/closing and multiple business terms
    is_business_format = (has_greeting or has_closing) and term_count >= 2
    
    # Forwarded business emails are still valuable
    if forwarded and term_count >= 3:
        return True
        
    return is_business_format

def categorize_email(content, subject=''):
    """
    Categorize email based on its primary business focus
    
    Args:
        content: Email body text
        subject: Email subject
        
    Returns:
        Category string
    """
    if not isinstance(content, str):
        return 'unknown'
    
    subject = str(subject) if subject is not None else ''
    combined_text = (content + " " + subject).lower()
    
    # Define categories and their associated terms
    categories = {
        'financial': [
            'financial', 'trading', 'price', 'market', 'cost', 'revenue',
            'budget', 'forecast', 'profit', 'loss', 'expense', 'investment',
            'capital', 'fund', 'finance', 'accounting', 'balance', 'audit',
            'tax', 'fiscal', 'quarter', 'earnings', 'dividend', 'stock'
        ],
        'legal': [
            'legal', 'contract', 'agreement', 'compliance', 'regulatory',
            'terms', 'provisions', 'clause', 'law', 'regulation', 'attorney',
            'counsel', 'liability', 'lawsuit', 'litigation', 'dispute',
            'settlement', 'negotiation', 'confidentiality', 'nda', 'rights'
        ],
        'project': [
            'project', 'schedule', 'deadline', 'progress', 'status', 'update',
            'milestone', 'deliverable', 'timeline', 'task', 'implementation',
            'development', 'plan', 'execution', 'resource', 'stakeholder',
            'requirement', 'specification', 'scope', 'phase', 'objective'
        ],
        'meeting': [
            'meeting', 'agenda', 'discuss', 'conference', 'call', 'schedule',
            'appointment', 'invite', 'calendar', 'availability', 'presentation',
            'attendee', 'participant', 'venue', 'location', 'room', 'minutes'
        ],
        'transaction': [
            'transaction', 'deal', 'purchase', 'sale', 'acquisition', 'merger',
            'divestiture', 'offer', 'bid', 'proposal', 'negotiation', 'term sheet',
            'valuation', 'due diligence', 'closing', 'signing', 'buyer', 'seller'
        ]
    }
    
    # Count terms in each category
    category_counts = {}
    for category, terms in categories.items():
        category_counts[category] = sum(1 for term in terms if term in combined_text)
    
    # Get the category with the most matches
    if sum(category_counts.values()) > 0:
        return max(category_counts, key=category_counts.get)
    
    return 'general'

def select_diverse_sample(df, sample_size=100):
    """
    Select a diverse sample of business emails across different categories
    
    Args:
        df: DataFrame with emails
        sample_size: Number of emails to select
        
    Returns:
        DataFrame with selected emails
    """
    logger.info("Identifying business emails...")
    
    # Add business classification
    df['is_business'] = df.apply(lambda row: is_business_email(row['content'], row.get('subject', '')), axis=1)
    business_emails = df[df['is_business']].copy()
    
    logger.info(f"Found {len(business_emails)} business emails")
    
    if len(business_emails) == 0:
        logger.warning("No business emails found. Check your filtering criteria.")
        return pd.DataFrame()
    
    # Categorize business emails
    logger.info("Categorizing business emails...")
    business_emails['category'] = business_emails.apply(
        lambda row: categorize_email(row['content'], row.get('subject', '')), axis=1
    )
    
    # Get distribution by category
    category_counts = business_emails['category'].value_counts()
    logger.info("\nCategory distribution:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count}")
    
    # Determine how many to take from each category
    categories = ['financial', 'legal', 'project', 'meeting', 'transaction', 'general']
    available_categories = [cat for cat in categories if cat in category_counts]
    
    # Allocate samples per category, ensuring every category gets representation
    category_samples = {}
    base_per_category = max(1, sample_size // len(available_categories))
    
    for category in available_categories:
        category_emails = business_emails[business_emails['category'] == category]
        category_samples[category] = min(base_per_category, len(category_emails))
    
    # Adjust to reach the desired sample size
    remaining = sample_size - sum(category_samples.values())
    
    if remaining > 0:
        # Allocate remaining samples proportionally to category sizes
        proportions = {cat: count/len(business_emails) for cat, count in category_counts.items()}
        
        for category in sorted(available_categories, 
                              key=lambda c: proportions.get(c, 0), 
                              reverse=True):
            can_add = min(remaining, 
                          len(business_emails[business_emails['category'] == category]) - category_samples[category])
            if can_add > 0:
                category_samples[category] += can_add
                remaining -= can_add
            
            if remaining <= 0:
                break
    
    # Select emails from each category
    selected_emails = []
    
    for category, sample_count in category_samples.items():
        if sample_count > 0:
            category_emails = business_emails[business_emails['category'] == category]
            # Seed for reproducibility
            category_sample = category_emails.sample(n=sample_count, random_state=42)
            selected_emails.append(category_sample)
    
    # Combine selected emails
    final_selection = pd.concat(selected_emails) if selected_emails else pd.DataFrame()
    
    logger.info(f"\nSelected {len(final_selection)} emails")
    logger.info("Selected category distribution:")
    selected_counts = final_selection['category'].value_counts()
    for category in categories:
        if category in selected_counts:
            logger.info(f"  {category}: {selected_counts[category]}")
    
    return final_selection

def parse_email_content(content):
    """
    Parse email content to separate header, body, and signature
    
    Args:
        content: Raw email content
        
    Returns:
        Dictionary with parsed components
    """
    if not isinstance(content, str):
        return {'header': '', 'body': '', 'signature': ''}
    
    lines = content.split('\n')
    
    # Identify header section (ends at first blank line)
    header_end = 0
    for i, line in enumerate(lines):
        if not line.strip():
            header_end = i
            break
    
    header = '\n'.join(lines[:header_end]).strip()
    
    # Look for signature indicators
    signature_patterns = [
        r'\s*-{2,}\s*', r'\s*_{2,}\s*',  # Dashes or underscores
        r'\s*regards,\s*', r'\s*sincerely,\s*',
        r'\s*thank you,\s*', r'\s*thanks,\s*',
        r'\s*best,\s*', r'\s*cheers,\s*'
    ]
    
    signature_start = len(lines)
    for pattern in signature_patterns:
        for i in range(header_end + 1, len(lines)):
            if re.match(pattern, lines[i], re.IGNORECASE):
                signature_start = i
                break
        if signature_start < len(lines):
            break
    
    # Extract body and signature
    body = '\n'.join(lines[header_end:signature_start]).strip()
    signature = '\n'.join(lines[signature_start:]).strip()
    
    return {
        'header': header,
        'body': body,
        'signature': signature
    }

def extract_email_metadata(df):
    """
    Extract and add metadata from emails for better analysis
    
    Args:
        df: DataFrame with emails
        
    Returns:
        DataFrame with additional metadata columns
    """
    df_enhanced = df.copy()
    
    # Parse content into components
    logger.info("Parsing email content components...")
    parsed_components = df_enhanced['content'].apply(parse_email_content)
    
    # Add parsed components as new columns
    df_enhanced['header_text'] = parsed_components.apply(lambda x: x['header'])
    df_enhanced['body_text'] = parsed_components.apply(lambda x: x['body'])
    df_enhanced['signature_text'] = parsed_components.apply(lambda x: x['signature'])
    
    # Extract length features
    df_enhanced['content_length'] = df_enhanced['content'].astype(str).str.len()
    df_enhanced['body_length'] = df_enhanced['body_text'].astype(str).str.len()
    
    # Check for characteristics relevant to eDiscovery
    df_enhanced['has_attachment_mention'] = df_enhanced['content'].str.contains(
        'attach|enclosed|document|file|spreadsheet|presentation', 
        case=False, na=False
    )
    
    df_enhanced['has_meeting_mention'] = df_enhanced['content'].str.contains(
        'meeting|discuss|call|conference|agenda', 
        case=False, na=False
    )
    
    df_enhanced['has_financial_mention'] = df_enhanced['content'].str.contains(
        'dollar|payment|invoice|budget|cost|price|financial', 
        case=False, na=False
    )
    
    logger.info("Metadata extraction complete")
    return df_enhanced

def save_selected_emails(df, output_path, include_metadata=True):
    """
    Save selected emails to JSON format
    
    Args:
        df: DataFrame with selected emails
        output_path: Path to save the output file
        include_metadata: Whether to include additional metadata
        
    Returns:
        None
    """
    # Convert to list of dictionaries
    emails_list = []
    
    for _, row in df.iterrows():
        # Create a dictionary with the main email fields
        email_dict = {
            'message_id': row.get('message_id', ''),
            'date': row.get('date', ''),
            'from': row.get('from', ''),
            'to': row.get('to', ''),
            'subject': row.get('subject', ''),
            'content': row.get('content', ''),
            'category': row.get('category', '')
        }
        
        # Add parsed components if available
        if include_metadata:
            email_dict.update({
                'header_text': row.get('header_text', ''),
                'body_text': row.get('body_text', ''),
                'signature_text': row.get('signature_text', ''),
                'content_length': int(row.get('content_length', 0)),
                'has_attachment_mention': bool(row.get('has_attachment_mention', False)),
                'has_meeting_mention': bool(row.get('has_meeting_mention', False)),
                'has_financial_mention': bool(row.get('has_financial_mention', False))
            })
        
        emails_list.append(email_dict)
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(emails_list, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(emails_list)} emails to {output_path}")

def main():
    """Main execution function"""
    # Configuration
    csv_dir = "emails"  # Directory containing CSV files
    output_dir = "selected_emails"
    num_csv_files = 5  # Number of CSV files to aggregate
    sample_size = 100  # Increase sample size for more diversity
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"selected_enron_emails_{timestamp}.json")
    
    try:
        # Aggregate multiple CSV files
        combined_df = aggregate_csv_files(csv_dir, num_csv_files)
        
        # Clean dataset
        df_clean = clean_dataset(combined_df)
        
        # Add metadata
        df_enhanced = extract_email_metadata(df_clean)
        
        # Select diverse sample
        selected_emails = select_diverse_sample(df_enhanced, sample_size)
        
        if len(selected_emails) == 0:
            logger.error("No emails were selected. Check your filtering criteria.")
            return
        
        # Save selected emails
        save_selected_emails(selected_emails, output_path)
        
        # Print a preview of selected emails
        logger.info("\nPreview of selected emails:")
        for i, (_, email) in enumerate(selected_emails.head(3).iterrows()):
            logger.info(f"\nEmail {i+1}:")
            logger.info(f"From: {email.get('from', '')}")
            logger.info(f"Subject: {email.get('subject', '')}")
            logger.info(f"Category: {email.get('category', '')}")
            content_preview = email.get('content', '')[:100]
            logger.info(f"Content preview: {content_preview}...")
        
    except Exception as e:
        logger.error(f"Error in email selection process: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()