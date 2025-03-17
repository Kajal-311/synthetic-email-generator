# Synthetic Email Generator

A system that transforms Enron emails into synthetic business communications with an Indian context while preserving privacy and structural integrity.

## Overview

This project creates synthetic email data for eDiscovery applications, enabling LLM fine-tuning without privacy concerns. It uses a hybrid approach combining traditional NLP techniques with LLMs to transform Enron emails into realistic Indian business communications across multiple industries.

### Key Features
- Complete de-identification of source emails (100% score)
- Authentic Indian business context adaptation (100% score)
- Multi-domain transformation (finance, technology, healthcare, agriculture, education)
- Structural preservation of email format (81% score)
- Comprehensive evaluation framework (94% overall quality score)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic-email-generator.git
cd synthetic-email-generator

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources (if needed)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

Usage
Basic Usage
# Run the complete pipeline
python src/synthetic_email_pipeline.py --mode all --api-key YOUR_OPENAI_API_KEY

# Run specific stages
python src/synthetic_email_pipeline.py --mode select --num-files 5 --sample-size 100
python src/synthetic_email_pipeline.py --mode generate --max-emails 50
python src/synthetic_email_pipeline.py --mode evaluate --simple-eval
Parameters

--mode: Pipeline mode (select, generate, evaluate, or all)
--csv-dir: Directory containing CSV files (default: emails)
--num-files: Number of CSV files to process (default: 5)
--sample-size: Number of emails to select (default: 100)
--max-emails: Maximum emails to generate/evaluate (default: 50)
--api-key: OpenAI API key for transformation
--simple-eval: Use simplified evaluation (faster)

Results
The system achieves:

100% de-identification score
100% Indian context score
81% structure preservation score
94% overall quality score

Documentation
For more detailed information, see:

Experiment Plan
Experiment Results
Technical Report

License
This project is licensed under the MIT License - see the LICENSE file for details.