import os
import logging
import argparse
import json
from datetime import datetime
import logging.config


#!/usr/bin/env python3
"""
Synthetic Email Generator Pipeline

This script orchestrates the entire process of:
1. Selecting emails from the Enron dataset
2. Generating synthetic emails with Indian context
3. Evaluating the quality of synthetic emails
"""



# Configure logging
def setup_logging():
    """Setup logging configuration"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Check if logging config file exists
        config_path = os.path.join('config', 'logging_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            # Fallback to basic configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler("logs/synthetic_email_pipeline.log"),
                    logging.StreamHandler()
                ]
            )
    except Exception as e:
        # Fallback to basic configuration if loading fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/synthetic_email_pipeline.log"),
                logging.StreamHandler()
            ]
        )
        logging.warning(f"Error setting up logging configuration: {e}. Using basic configuration.")

# Initialize logger after setup
setup_logging()
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from file or use defaults"""
    config_path = os.path.join('config', 'config.json')
    default_path = os.path.join('config', 'default_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    elif os.path.exists(default_path):
        with open(default_path, 'r') as f:
            return json.load(f)
    else:
        return {}

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['emails', 'selected_emails', 'synthetic_emails', 'evaluation_results', 'logs', 'config']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Synthetic Email Generator Pipeline")
    
    parser.add_argument('--mode', choices=['select', 'generate', 'evaluate', 'all'], 
                        default='all', help='Pipeline mode to run')
    
    parser.add_argument('--csv-dir', default='emails',
                        help='Directory containing CSV files with Enron emails')
    
    parser.add_argument('--num-files', type=int, default=5,
                        help='Number of CSV files to process')
    
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Number of emails to select from dataset')
    
    parser.add_argument('--max-emails', type=int, default=50,
                        help='Maximum number of emails to generate/evaluate')
    
    parser.add_argument('--api-key', default=None,
                        help='OpenAI API key for email transformation')
    
    parser.add_argument('--simple-eval', action='store_true',
                        help='Use simplified evaluation instead of comprehensive')
    
    return parser.parse_args()

def run_email_selection(args):
    """Run email selection process"""
    logger.info("Starting email selection process...")
    
    try:
        # Import selection module
        from email_selection import aggregate_csv_files, clean_dataset, extract_email_metadata
        from email_selection import select_diverse_sample, save_selected_emails
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join('selected_emails', f"selected_enron_emails_{timestamp}.json")
        
        # Aggregate multiple CSV files
        combined_df = aggregate_csv_files(args.csv_dir, args.num_files)
        
        # Clean dataset
        df_clean = clean_dataset(combined_df)
        
        # Add metadata
        df_enhanced = extract_email_metadata(df_clean)
        
        # Select diverse sample
        selected_emails = select_diverse_sample(df_enhanced, args.sample_size)
        
        if len(selected_emails) == 0:
            logger.error("No emails were selected. Check filtering criteria.")
            return None
        
        # Save selected emails
        save_selected_emails(selected_emails, output_path)
        
        logger.info(f"Email selection complete. Selected {len(selected_emails)} emails to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in email selection process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_email_generation(args, selected_emails_path=None):
    import os
    """Run synthetic email generation process"""
    logger.info("Starting synthetic email generation process...")
    
    try:
        # Import generation module
        from email_generator import SyntheticEmailGenerator
        
        # Find most recent selected emails file if not specified
        if not selected_emails_path:
            selected_files = [f for f in os.listdir('selected_emails') if f.endswith('.json')]
            if not selected_files:
                logger.error("No selected email files found")
                return None
            
            selected_files.sort(reverse=True)
            selected_emails_path = os.path.join('selected_emails', selected_files[0])
            logger.info(f"Using selected emails from: {selected_emails_path}")
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join('synthetic_emails', f"indian_synthetic_emails_{timestamp}.json")
        
        # Load selected emails - using our own function to avoid import errors
        all_emails = []
        try:
            with open(selected_emails_path, 'r') as f:
                emails_data = json.load(f)
                
            # Handle both list and dictionary formats
            if isinstance(emails_data, list):
                all_emails = emails_data
            elif isinstance(emails_data, dict) and 'emails' in emails_data:
                all_emails = emails_data['emails']
            elif isinstance(emails_data, dict):
                # If it's a single email as a dictionary
                all_emails = [emails_data]
                
            logger.info(f"Loaded {len(all_emails)} emails from {selected_emails_path}")
        except Exception as e:
            logger.error(f"Error loading selected emails: {e}")
            return None
        
        # Limit the number of emails to process if needed
        if args.max_emails and len(all_emails) > args.max_emails:
            logger.info(f"Limiting to {args.max_emails} emails for processing")
            all_emails = all_emails[:args.max_emails]
        
        # Get API key from arguments, config or environment
        config = load_config()
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        api_key = args.api_key or config.get('openai_api_key') or os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI API key
        if api_key:
            import openai
            openai.api_key = api_key
            logger.info("OpenAI API key configured")
        else:
            logger.warning("No OpenAI API key provided. Using fallback transformation.")
        
        # Initialize generator
        entity_mapping_path = os.path.join('config', 'entity_mappings.json')
        generator = SyntheticEmailGenerator(entity_mapping_path)
        
        # Transform emails
        transformed_emails = generator.batch_transform(all_emails, output_path)
        
        logger.info(f"Email generation complete. Generated {len(transformed_emails)} emails to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in email generation process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
        
def run_email_evaluation(args, synthetic_emails_path=None):
    """Run synthetic email evaluation process"""
    logger.info("Starting synthetic email evaluation process...")
    
    try:
        # Find most recent synthetic emails file if not specified
        if not synthetic_emails_path:
            synthetic_files = [f for f in os.listdir('synthetic_emails') 
                              if f.endswith('.json') and not f.endswith('_mappings.json')
                              and not f.endswith('_analysis.json')]
            if not synthetic_files:
                logger.error("No synthetic email files found")
                return None
            
            synthetic_files.sort(reverse=True)
            synthetic_emails_path = os.path.join('synthetic_emails', synthetic_files[0])
        
        if args.simple_eval:
            # Import and use simple evaluator
            from email_evaluator import SimpleEmailEvaluator
            
            evaluator = SimpleEmailEvaluator('selected_emails', synthetic_emails_path, args.max_emails)
            results = evaluator.evaluate()
            
            logger.info("\nEvaluation complete!")
            logger.info(f"De-identification score: {results['deidentification_score']:.2f}")
            logger.info(f"Indian context score: {results['indian_context_score']:.2f}")
        else:
            # Import and use comprehensive evaluator
            from email_evaluator import SyntheticEmailEvaluator
            
            evaluator = SyntheticEmailEvaluator('selected_emails', synthetic_emails_path, args.max_emails)
            results = evaluator.run_all_evaluations()
            
            if 'overall_quality_score' in results:
                logger.info(f"\nOverall quality score: {results['overall_quality_score']:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in email evaluation process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create necessary directories
    create_directories()
    
    # Track output paths for each stage
    selected_emails_path = None
    synthetic_emails_path = None
    
    # Run pipeline based on mode
    if args.mode in ['select', 'all']:
        selected_emails_path = run_email_selection(args)
    
    if args.mode in ['generate', 'all'] and (selected_emails_path or args.mode == 'generate'):
        synthetic_emails_path = run_email_generation(args, selected_emails_path)
    
    if args.mode in ['evaluate', 'all'] and (synthetic_emails_path or args.mode == 'evaluate'):
        evaluation_results = run_email_evaluation(args, synthetic_emails_path)
    
    logger.info("Pipeline execution complete")

if __name__ == "__main__":
    main()