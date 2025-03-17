import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("email_evaluation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_emails(file_path_or_dir):
    """
    Load emails from JSON file or directory
    
    Args:
        file_path_or_dir: Path to JSON file or directory containing JSON files
        
    Returns:
        List of email dictionaries
    """
    emails = []
    
    if os.path.isdir(file_path_or_dir):
        # Load all JSON files from directory
        json_files = glob.glob(os.path.join(file_path_or_dir, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        emails.extend(data)
                    elif isinstance(data, dict) and 'emails' in data:
                        emails.extend(data['emails'])
                    elif isinstance(data, dict):
                        emails.append(data)
                        
                logger.info(f"Loaded {len(emails)} emails from {json_file}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
    else:
        # Load single JSON file
        try:
            with open(file_path_or_dir, 'r') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    emails = data
                elif isinstance(data, dict) and 'emails' in data:
                    emails = data['emails']
                elif isinstance(data, dict):
                    emails = [data]
                    
            logger.info(f"Loaded {len(emails)} emails from {file_path_or_dir}")
        except Exception as e:
            logger.error(f"Error loading {file_path_or_dir}: {e}")
    
    return emails


class SimpleEmailEvaluator:
    """A simplified version of the evaluator for quick assessments"""
    
    def __init__(self, original_emails_path, synthetic_emails_path, max_emails=None):
        """
        Initialize the simplified evaluator
        
        Args:
            original_emails_path: Path to original selected emails (file or directory)
            synthetic_emails_path: Path to synthetic emails (file or directory)
            max_emails: Maximum number of emails to evaluate (None for all)
        """
        self.original_emails_path = original_emails_path
        self.synthetic_emails_path = synthetic_emails_path
        
        # Load emails
        self.original_emails = self.load_emails(original_emails_path)
        self.synthetic_emails = self.load_emails(synthetic_emails_path)
        
        # Limit number of emails if specified
        if max_emails:
            if len(self.original_emails) > max_emails:
                self.original_emails = self.original_emails[:max_emails]
            if len(self.synthetic_emails) > max_emails:
                self.synthetic_emails = self.synthetic_emails[:max_emails]
                
        logger.info(f"Loaded {len(self.original_emails)} original emails and {len(self.synthetic_emails)} synthetic emails")
    
    def load_emails(self, file_path_or_dir):
        """Simple loader for emails"""
        import os
        import json
        
        emails = []
        
        if os.path.isdir(file_path_or_dir):
            # Get the most recent file
            json_files = [f for f in os.listdir(file_path_or_dir) if f.endswith('.json')]
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {file_path_or_dir}")
            
            json_files.sort(reverse=True)
            file_path = os.path.join(file_path_or_dir, json_files[0])
        else:
            file_path = file_path_or_dir
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                emails = data
            elif isinstance(data, dict) and 'emails' in data:
                emails = data['emails']
            else:
                emails = [data]
        except Exception as e:
            logger.error(f"Error loading emails from {file_path}: {e}")
            emails = []
        
        return emails
            
    def evaluate(self):
        """
        Run a simplified evaluation
        
        Returns:
            Dictionary with basic evaluation metrics
        """
        # Check for de-identification
        original_emails = set()
        for email in self.original_emails:
            content = email.get('content', '')
            if not isinstance(content, str):
                continue
            emails_found = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
            original_emails.update(emails_found)
        
        # Check how many original emails appear in synthetic data
        found_emails = 0
        for orig_email in original_emails:
            for synth_email in self.synthetic_emails:
                synth_content = synth_email.get('content', '')
                if not isinstance(synth_content, str):
                    continue
                if orig_email in synth_content:
                    found_emails += 1
                    break
                    
        deidentification_score = 1.0 if not original_emails else 1.0 - (found_emails / len(original_emails))
        
        logger.info(f"De-identification score: {deidentification_score:.2f}")
        
        # Check for Indian context
        indian_terms = [
            'rupee', 'crore', 'lakh', 'namaste', 'India', 'Mumbai', 'Delhi', 'Bengaluru',
            'SEBI', 'RBI', 'GST', 'PAN', 'Aadhaar', 'Ministry', 'Pradesh'
        ]
        
        indian_names = [name.lower() for name in 
                       ['Sharma', 'Patel', 'Singh', 'Kumar', 'Gupta', 'Verma', 'Shah',
                        'Aarav', 'Vivaan', 'Ananya', 'Diya', 'Advait', 'Ishaan']]
        
        # Count occurrences in synthetic emails
        indian_context_count = 0
        for email in self.synthetic_emails:
            content = email.get('content', '').lower()
            from_field = email.get('from', '').lower()
            
            # Check for Indian terms
            has_indian_term = any(term.lower() in content for term in indian_terms)
            
            # Check for Indian names
            has_indian_name = any(name in content for name in indian_names)
            
            # Check for Indian domain
            has_indian_domain = any(suffix in from_field for suffix in ['.in', '.co.in', 'org.in', '.edu.in'])
            
            # Check for rupee symbol
            has_rupee = '₹' in content
            
            # Count email as having Indian context if it has any of these features
            if has_indian_term or has_indian_name or has_indian_domain or has_rupee:
                indian_context_count += 1
        
        indian_context_score = indian_context_count / len(self.synthetic_emails) if self.synthetic_emails else 0
        
        logger.info(f"Indian context score: {indian_context_score:.2f}")
        
        # Domain distribution
        domain_counts = {}
        for email in self.synthetic_emails:
            domain = email.get('target_domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.info("Domain distribution:")
        for domain, count in domain_counts.items():
            percentage = count/len(self.synthetic_emails) if self.synthetic_emails else 0
            logger.info(f"  {domain}: {count} ({percentage:.2f})")
        
        # Basic structure check
        structure_scores = {
            'has_greeting': 0,
            'has_closing': 0,
            'is_reasonable_length': 0
        }
        
        for email in self.synthetic_emails:
            content = email.get('content', '')
            if not isinstance(content, str):
                continue
            if re.search(r'(dear|hello|hi|namaste)\s+\w+', content.lower()):
                structure_scores['has_greeting'] += 1
            if re.search(r'(regards|sincerely|thank you|thanks|best)', content.lower()):
                structure_scores['has_closing'] += 1
            if 100 <= len(content) <= 5000:
                structure_scores['is_reasonable_length'] += 1
        
        structure_rate = sum(structure_scores.values()) / (len(self.synthetic_emails) * len(structure_scores)) if self.synthetic_emails else 0
        
        logger.info(f"Structure score: {structure_rate:.2f}")
        
        # Compile results
        results = {
            'deidentification_score': deidentification_score,
            'indian_context_score': indian_context_score,
            'structure_score': structure_rate,
            'domain_counts': domain_counts
        }
        
        # Overall score - simple average
        results['overall_score'] = sum([
            deidentification_score,
            indian_context_score,
            structure_rate
        ]) / 3
        
        logger.info(f"Overall score: {results['overall_score']:.2f}")
        
        return results



class SyntheticEmailEvaluator:
    """System for evaluating quality of synthetic emails with Indian context"""
    
    def __init__(self, original_emails_path, synthetic_emails_path, max_emails=None):
        """
        Initialize the evaluator
        
        Args:
            original_emails_path: Path to original selected emails (file or directory)
            synthetic_emails_path: Path to synthetic emails (file or directory)
            max_emails: Maximum number of emails to evaluate (None for all)
        """
        # Save paths as attributes
        self.original_emails_path = original_emails_path
        self.synthetic_emails_path = synthetic_emails_path
        
        # Load emails
        self.original_emails = load_emails(original_emails_path)
        self.synthetic_emails = load_emails(synthetic_emails_path)
        
        # Limit number of emails if specified
        if max_emails:
            if len(self.original_emails) > max_emails:
                self.original_emails = self.original_emails[:max_emails]
            if len(self.synthetic_emails) > max_emails:
                self.synthetic_emails = self.synthetic_emails[:max_emails]
        
        logger.info(f"Evaluating {len(self.original_emails)} original emails and {len(self.synthetic_emails)} synthetic emails")
        
        # Set up output directory
        self.output_dir = "evaluation_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def evaluate_entity_deidentification(self):
        """
        Evaluate how well entities have been de-identified
        
        Returns:
            Dictionary with entity evaluation metrics
        """
        logger.info("\nEvaluating entity de-identification...")
        
        # Load entity mappings if available
        mappings_path = os.path.splitext(os.path.abspath(self.synthetic_emails_path))[0] + "_mappings.json"
        entity_mappings = {}
        
        if os.path.exists(mappings_path):
            with open(mappings_path, 'r') as f:
                entity_mappings = json.load(f)
        
        # Extract entities from original emails
        original_entities = {
            'people': [],
            'companies': [],
            'emails': [],
            'locations': []
        }
        
        # Patterns for entity extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Extract entities from original emails
        for email in self.original_emails:
            if not isinstance(email, dict):
                logger.warning(f"Skipping non-dictionary original email: {type(email)}")
                continue
                
            content = email.get('content', '')
            if not isinstance(content, str):
                logger.warning(f"Skipping original email with non-string content: {type(content)}")
                continue
            
            # Extract emails
            emails = re.findall(email_pattern, content)
            original_entities['emails'].extend(emails)
            
            # Other entities would be extracted via NER in a real implementation
        
        # Check if original entities appear in synthetic emails
        entities_found = {entity_type: 0 for entity_type in original_entities}
        total_entities = {entity_type: len(entities) for entity_type, entities in original_entities.items()}
        
        for entity_type, entities in original_entities.items():
            for entity in entities:
                for synthetic_email in self.synthetic_emails:
                    if not isinstance(synthetic_email, dict):
                        logger.warning(f"Skipping non-dictionary synthetic email: {type(synthetic_email)}")
                        continue
                        
                    synthetic_content = synthetic_email.get('content', '')
                    if not isinstance(synthetic_content, str):
                        logger.warning(f"Skipping synthetic email with non-string content: {type(synthetic_content)}")
                        continue
                        
                    if entity in synthetic_content:
                        entities_found[entity_type] += 1
                        break
        
        # Calculate deidentification rates
        deidentification_rates = {}
        for entity_type in original_entities:
            if total_entities[entity_type] > 0:
                deidentification_rates[entity_type] = 1 - (entities_found[entity_type] / total_entities[entity_type])
            else:
                deidentification_rates[entity_type] = 1.0
        
        # Calculate overall deidentification rate
        total_original = sum(total_entities.values())
        total_found = sum(entities_found.values())
        
        overall_rate = 1.0 if total_original == 0 else 1 - (total_found / total_original)
        
        results = {
            'entity_counts': total_entities,
            'entities_found': entities_found,
            'deidentification_rates': deidentification_rates,
            'overall_deidentification_rate': overall_rate
        }
        
        logger.info(f"Overall de-identification rate: {overall_rate:.2f}")
        for entity_type, rate in deidentification_rates.items():
            logger.info(f"  {entity_type}: {rate:.2f}")
        
        return results
        
    def evaluate_structural_preservation(self):
        """
        Evaluate how well the structure of emails is preserved
        
        Returns:
            Dictionary with structural preservation metrics
        """
        logger.info("\nEvaluating structural preservation...")
        
        # Define structural elements to check
        structural_elements = {
            'has_greeting': lambda x: bool(re.search(r'(dear|hello|hi|namaste)\s+\w+', x.lower())),
            'has_closing': lambda x: bool(re.search(r'(regards|sincerely|thank you|thanks|best)', x.lower())),
            'has_signature': lambda x: bool(re.search(r'\n-{2,}.+|^\s*-{2,}', x, re.MULTILINE)),
            'has_forwarded': lambda x: 'forwarded' in x.lower(),
            'paragraph_count': lambda x: len([p for p in x.split('\n\n') if p.strip()]),
            'has_bullet_points': lambda x: bool(re.search(r'^\s*[-*•]\s', x, re.MULTILINE))
        }
        
        # Analyze original emails
        original_structure = {
            element: {'count': 0, 'values': []} for element in structural_elements
        }
        
        for email in self.original_emails:
            content = email.get('content', '')
            for element, detector in structural_elements.items():
                value = detector(content)
                if isinstance(value, bool) and value:
                    original_structure[element]['count'] += 1
                if not isinstance(value, bool):
                    original_structure[element]['values'].append(value)
        
        # Analyze synthetic emails
        synthetic_structure = {
            element: {'count': 0, 'values': []} for element in structural_elements
        }
        
        for email in self.synthetic_emails:
            content = email.get('content', '')
            for element, detector in structural_elements.items():
                value = detector(content)
                if isinstance(value, bool) and value:
                    synthetic_structure[element]['count'] += 1
                if not isinstance(value, bool):
                    synthetic_structure[element]['values'].append(value)
        
        # Calculate preservation rates
        preservation_rates = {}
        for element in structural_elements:
            if isinstance(structural_elements[element](self.original_emails[0].get('content', '')), bool):
                original_rate = original_structure[element]['count'] / len(self.original_emails)
                synthetic_rate = synthetic_structure[element]['count'] / len(self.synthetic_emails)
                preservation_rates[element] = 1 - abs(original_rate - synthetic_rate)
            else:
                # For numeric values, compare distributions
                original_mean = np.mean(original_structure[element]['values']) if original_structure[element]['values'] else 0
                synthetic_mean = np.mean(synthetic_structure[element]['values']) if synthetic_structure[element]['values'] else 0
                
                if original_mean == 0:
                    preservation_rates[element] = 1.0 if synthetic_mean == 0 else 0.0
                else:
                    # Calculate how close the means are
                    preservation_rates[element] = max(0, 1 - min(1, abs(original_mean - synthetic_mean) / original_mean))
        
        # Calculate overall preservation rate
        overall_rate = np.mean(list(preservation_rates.values()))
        
        results = {
            'original_structure': original_structure,
            'synthetic_structure': synthetic_structure,
            'preservation_rates': preservation_rates,
            'overall_preservation_rate': overall_rate
        }
        
        logger.info(f"Overall structural preservation rate: {overall_rate:.2f}")
        for element, rate in preservation_rates.items():
            logger.info(f"  {element}: {rate:.2f}")
        
        return results
    
    def evaluate_language_patterns(self):
        """
        Evaluate how well language patterns are preserved
        
        Returns:
            Dictionary with language pattern metrics
        """
        logger.info("\nEvaluating language patterns...")
        
        # Define language pattern metrics
        stop_words = set(stopwords.words('english'))
        
        def get_language_metrics(text):
            if not isinstance(text, str) or not text.strip():
                return {}
                
            tokens = word_tokenize(text.lower())
            words = [word for word in tokens if word.isalpha()]
            sentences = [s for s in re.split(r'[.!?]', text) if s.strip()]
            
            metrics = {
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
                'lexical_diversity': len(set(words)) / len(words) if words else 0,
                'non_stopword_ratio': len([w for w in words if w not in stop_words]) / len(words) if words else 0
            }
            
            return metrics
        
        # Analyze original emails
        original_metrics = []
        for email in self.original_emails:
            content = email.get('content', '')
            metrics = get_language_metrics(content)
            if metrics:
                original_metrics.append(metrics)
        
        # Analyze synthetic emails
        synthetic_metrics = []
        for email in self.synthetic_emails:
            content = email.get('content', '')
            metrics = get_language_metrics(content)
            if metrics:
                synthetic_metrics.append(metrics)
        
        # Calculate average metrics
        original_avg = {metric: np.mean([m[metric] for m in original_metrics]) 
                      for metric in original_metrics[0]} if original_metrics else {}
        
        synthetic_avg = {metric: np.mean([m[metric] for m in synthetic_metrics]) 
                      for metric in synthetic_metrics[0]} if synthetic_metrics else {}
        
        # Calculate similarity scores
        similarity_scores = {}
        for metric in original_avg:
            if original_avg[metric] == 0:
                similarity_scores[metric] = 1.0 if synthetic_avg.get(metric, 0) == 0 else 0.0
            else:
                # Calculate how close the metrics are
                similarity_scores[metric] = max(0, 1 - min(1, abs(original_avg[metric] - synthetic_avg.get(metric, 0)) / original_avg[metric]))
        
        # Calculate overall language pattern similarity
        overall_similarity = np.mean(list(similarity_scores.values())) if similarity_scores else 0.0
        
        results = {
            'original_metrics': original_avg,
            'synthetic_metrics': synthetic_avg,
            'similarity_scores': similarity_scores,
            'overall_language_similarity': overall_similarity
        }
        
        logger.info(f"Overall language pattern similarity: {overall_similarity:.2f}")
        for metric, score in similarity_scores.items():
            logger.info(f"  {metric}: {score:.2f}")
        
        return results
    
    def evaluate_domain_adaptation(self):
        """
        Evaluate how well emails have been adapted to target domains
        
        Returns:
            Dictionary with domain adaptation metrics
        """
        logger.info("\nEvaluating domain adaptation...")
        
        # Count target domains
        target_domains = [email.get('target_domain', 'unknown') for email in self.synthetic_emails]
        domain_counts = Counter(target_domains)
        
        # Define domain-specific terminology
        domain_terms = {
            'technology': ['software', 'platform', 'app', 'cloud', 'data', 'algorithm', 'interface', 'API', 'server', 'code'],
            'healthcare': ['patient', 'medical', 'clinical', 'treatment', 'diagnosis', 'doctor', 'hospital', 'care', 'health', 'therapy'],
            'agriculture': ['crop', 'farm', 'harvest', 'seed', 'soil', 'irrigation', 'agriculture', 'cultivation', 'yield', 'organic'],
            'finance': ['investment', 'portfolio', 'market', 'stock', 'trading', 'asset', 'fund', 'dividend', 'equity', 'bond'],
            'education': ['student', 'course', 'learning', 'curriculum', 'education', 'academic', 'school', 'university', 'teaching', 'professor']
        }
        
        # Check for domain-specific terminology in synthetic emails
        domain_term_usage = {domain: 0 for domain in domain_terms}
        domain_emails = {domain: 0 for domain in domain_terms}
        
        for email in self.synthetic_emails:
            content = email.get('content', '').lower()
            target_domain = email.get('target_domain', 'unknown')
            
            if target_domain in domain_terms:
                domain_emails[target_domain] += 1
                for term in domain_terms[target_domain]:
                    if term.lower() in content:
                        domain_term_usage[target_domain] += 1
        
        # Calculate domain adaptation scores
        domain_scores = {}
        for domain, count in domain_emails.items():
            if count == 0:
                domain_scores[domain] = 0.0
            else:
                domain_scores[domain] = domain_term_usage[domain] / (count * len(domain_terms[domain]))
        
        # Calculate overall domain adaptation score
        overall_score = np.mean(list(domain_scores.values())) if domain_scores else 0.0
        
        results = {
            'domain_counts': dict(domain_counts),
            'domain_term_usage': domain_term_usage,
            'domain_emails': domain_emails,
            'domain_scores': domain_scores,
            'overall_domain_adaptation': overall_score
        }
        
        logger.info(f"Overall domain adaptation score: {overall_score:.2f}")
        for domain, score in domain_scores.items():
            logger.info(f"  {domain}: {score:.2f}")
        
        return results
        
    def evaluate_indian_context(self):
        """
        Evaluate how well the Indian context has been incorporated
        
        Returns:
            Dictionary with Indian context metrics
        """
        logger.info("\nEvaluating Indian context...")
        
        # Define Indian-specific terms to check for
        indian_terms = [
            'rupee', 'crore', 'lakh', 'namaste', 'India', 'Mumbai', 'Delhi', 'Bengaluru',
            'SEBI', 'RBI', 'GST', 'PAN', 'Aadhaar', 'Ministry', 'Pradesh', 'chai',
            'kindly', 'please revert', 'do the needful', 'prepone'
        ]
        
        indian_names = [name.lower() for name in 
                       ['Sharma', 'Patel', 'Singh', 'Kumar', 'Gupta', 'Verma', 'Shah',
                        'Aarav', 'Vivaan', 'Ananya', 'Diya', 'Advait', 'Ishaan']]
        
        # Check for Indian terms in synthetic emails
        term_usage = {term: 0 for term in indian_terms}
        name_count = 0
        
        for email in self.synthetic_emails:
            content = email.get('content', '').lower()
            
            # Check for Indian terminology
            for term in indian_terms:
                if term.lower() in content:
                    term_usage[term] += 1
            
            # Check for Indian names
            for name in indian_names:
                if name in content:
                    name_count += 1
                    break
        
        # Calculate metrics
        term_usage_rate = {term: count / len(self.synthetic_emails) for term, count in term_usage.items()}
        name_usage_rate = name_count / len(self.synthetic_emails)
        
        # Indian currency format check (₹ symbol and crore/lakh usage)
        currency_format_count = sum(1 for email in self.synthetic_emails 
                                 if ('₹' in email.get('content', '') 
                                    and ('crore' in email.get('content', '').lower() 
                                         or 'lakh' in email.get('content', '').lower())))
        currency_format_rate = currency_format_count / len(self.synthetic_emails)
        
        # Indian domain suffixes (.in, .co.in, etc.)
        indian_domain_count = sum(1 for email in self.synthetic_emails 
                               if email.get('from', '').endswith(('.in', '.co.in', '.org.in', '.edu.in')))
        indian_domain_rate = indian_domain_count / len(self.synthetic_emails)
        
        # Overall score (average of key metrics)
        overall_indian_context_score = (
            sum(term_usage_rate.values()) / len(term_usage_rate) + 
            name_usage_rate + 
            currency_format_rate +
            indian_domain_rate
        ) / 4
        
        results = {
            'term_usage': term_usage,
            'term_usage_rate': term_usage_rate,
            'name_usage_rate': name_usage_rate,
            'currency_format_rate': currency_format_rate,
            'indian_domain_rate': indian_domain_rate,
            'overall_indian_context_score': overall_indian_context_score
        }
        
        logger.info(f"Overall Indian context score: {overall_indian_context_score:.2f}")
        logger.info(f"Indian name usage rate: {name_usage_rate:.2f}")
        logger.info(f"Indian currency format rate: {currency_format_rate:.2f}")
        logger.info(f"Indian domain suffix rate: {indian_domain_rate:.2f}")
        
        return results
    
    def evaluate_realism(self):
        """
        Evaluate overall realism of synthetic emails
        This would typically involve human evaluation, but we'll use some heuristics
        
        Returns:
            Dictionary with realism metrics
        """
        logger.info("\nEvaluating email realism...")
        
        # Define realism heuristics
        realism_checks = {
            'has_complete_structure': lambda x: (bool(re.search(r'(dear|hello|hi|namaste)\s+\w+', x.lower())) and 
                                             bool(re.search(r'(regards|sincerely|thank you|thanks|best)', x.lower()))),
            'has_realistic_length': lambda x: 100 <= len(x) <= 5000,
            'has_paragraphs': lambda x: len([p for p in x.split('\n\n') if p.strip()]) >= 2,
            'has_consistent_format': lambda x: bool(re.search(r'\n[A-Za-z\s]+,\n\n', x)) or bool(re.search(r'\n[A-Za-z]+\n', x)),
            'has_coherent_flow': lambda x: len(set(word_tokenize(x.lower()))) / len(word_tokenize(x.lower())) < 0.7 if word_tokenize(x.lower()) else False
        }
        
        # Check synthetic emails against realism heuristics
        realism_scores = {check: 0 for check in realism_checks}
        
        for email in self.synthetic_emails:
            content = email.get('content', '')
            for check, func in realism_checks.items():
                if func(content):
                    realism_scores[check] += 1
        
        # Calculate percentages
        realism_percentages = {check: count / len(self.synthetic_emails) for check, count in realism_scores.items()}
        
        # Calculate overall realism score
        overall_realism = np.mean(list(realism_percentages.values()))
        
        results = {
            'realism_counts': realism_scores,
            'realism_percentages': realism_percentages,
            'overall_realism': overall_realism
        }
        
        logger.info(f"Overall realism score: {overall_realism:.2f}")
        for check, percentage in realism_percentages.items():
            logger.info(f"  {check}: {percentage:.2f}")
        
        return results
    
    def generate_visualizations(self, results):
        """
        Generate visualizations for evaluation results
        
        Args:
            results: Dictionary with all evaluation results
        """
        logger.info("\nGenerating visualizations...")
        
        # Set up plot style
        plt.style.use('ggplot')
        
        # Check if all required data is available
        if not all(k in results for k in ['entity_deidentification', 'structural_preservation', 
                                         'language_patterns', 'domain_adaptation', 
                                         'indian_context', 'realism']):
            logger.error("Incomplete results data for visualization")
            return
        
        try:
            # 1. De-identification rates by entity type
            plt.figure(figsize=(10, 6))
            deidentification_rates = results['entity_deidentification']['deidentification_rates']
            plt.bar(deidentification_rates.keys(), deidentification_rates.values())
            plt.title('De-identification Rate by Entity Type')
            plt.ylabel('De-identification Rate')
            plt.ylim(0, 1)
            plt.savefig(os.path.join(self.output_dir, 'deidentification_rates.png'))
            
            # 2. Structural preservation
            plt.figure(figsize=(12, 6))
            preservation_rates = results['structural_preservation']['preservation_rates']
            plt.bar(preservation_rates.keys(), preservation_rates.values())
            plt.title('Structural Preservation by Element')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Preservation Rate')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'structural_preservation.png'))
            
            # 3. Language pattern similarity
            plt.figure(figsize=(10, 6))
            similarity_scores = results['language_patterns']['similarity_scores']
            plt.bar(similarity_scores.keys(), similarity_scores.values())
            plt.title('Language Pattern Similarity by Metric')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Similarity Score')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'language_similarity.png'))
            
            # 4. Domain adaptation
            plt.figure(figsize=(10, 6))
            domain_scores = results['domain_adaptation']['domain_scores']
            plt.bar(domain_scores.keys(), domain_scores.values())
            plt.title('Domain Adaptation Score by Target Domain')
            plt.ylabel('Adaptation Score')
            plt.ylim(0, 1)
            plt.savefig(os.path.join(self.output_dir, 'domain_adaptation.png'))
            
            # 5. Indian context evaluation
            plt.figure(figsize=(10, 6))
            indian_context = results['indian_context']
            indian_metrics = {
                'Name Usage': indian_context['name_usage_rate'],
                'Currency Format': indian_context['currency_format_rate'],
                'Indian Domain': indian_context['indian_domain_rate'],
                'Terms Usage Avg': sum(indian_context['term_usage_rate'].values()) / len(indian_context['term_usage_rate'])
            }
            plt.bar(indian_metrics.keys(), indian_metrics.values())
            plt.title('Indian Context Metrics')
            plt.ylabel('Rate')
            plt.ylim(0, 1)
            plt.savefig(os.path.join(self.output_dir, 'indian_context.png'))
            
            # 6. Overall metrics comparison
            plt.figure(figsize=(10, 6))
            overall_metrics = {
                'De-identification': results['entity_deidentification']['overall_deidentification_rate'],
                'Structural Preservation': results['structural_preservation']['overall_preservation_rate'],
                'Language Similarity': results['language_patterns']['overall_language_similarity'],
                'Domain Adaptation': results['domain_adaptation']['overall_domain_adaptation'],
                'Indian Context': results['indian_context']['overall_indian_context_score'],
                'Realism': results['realism']['overall_realism']
            }
            
            plt.bar(overall_metrics.keys(), overall_metrics.values())
            plt.title('Overall Evaluation Metrics')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'overall_metrics.png'))
            
            logger.info(f"Visualizations saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def run_all_evaluations(self):
        """
        Run all evaluations and compile results
        
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Running all evaluations...")
        
        try:
            results = {
                'entity_deidentification': self.evaluate_entity_deidentification(),
                'structural_preservation': self.evaluate_structural_preservation(),
                'language_patterns': self.evaluate_language_patterns(),
                'domain_adaptation': self.evaluate_domain_adaptation(),
                'indian_context': self.evaluate_indian_context(),
                'realism': self.evaluate_realism()
            }
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results = {}
        
        if results:
            # Calculate overall quality score
            overall_scores = [
                results.get('entity_deidentification', {}).get('overall_deidentification_rate', 0),
                results.get('structural_preservation', {}).get('overall_preservation_rate', 0),
                results.get('language_patterns', {}).get('overall_language_similarity', 0),
                results.get('domain_adaptation', {}).get('overall_domain_adaptation', 0),
                results.get('indian_context', {}).get('overall_indian_context_score', 0),
                results.get('realism', {}).get('overall_realism', 0)
            ]
            
            results['overall_quality_score'] = np.mean(overall_scores)
            
            logger.info(f"\nOverall quality score: {results['overall_quality_score']:.2f}")
            
            # Generate visualizations
            self.generate_visualizations(results)
            
            # Save results to JSON
            results_path = os.path.join(self.output_dir, 'evaluation_results.json')
            try:
                with open(results_path, 'w') as f:
                    # Convert numpy values to native Python types for JSON serialization
                    def convert_np(obj):
                        if isinstance(obj, np.generic):
                            return obj.item()
                        elif isinstance(obj, dict):
                            return {k: convert_np(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_np(i) for i in obj]
                        else:
                            return obj
                    
                    json.dump(convert_np(results), f, indent=2)
                
                logger.info(f"Evaluation results saved to {results_path}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")
        else:
            logger.error("No evaluation results to process")
        
        return results

def main():
    """Main execution function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Synthetic Email Evaluator")
    parser.add_argument('--original', default='selected_emails',
                        help='Path to original emails directory or file')
    parser.add_argument('--synthetic', default=None,
                        help='Path to synthetic emails file (uses most recent if not specified)')
    parser.add_argument('--max-emails', type=int, default=50,
                        help='Maximum number of emails to evaluate')
    parser.add_argument('--simple', action='store_true',
                        help='Use simplified evaluation')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip generating visualizations')
    
    args = parser.parse_args()
    
    # Configuration
    original_emails_path = args.original
    max_emails = args.max_emails
    output_dir = 'evaluation_results'
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find synthetic emails file if not specified
    synthetic_emails_path = args.synthetic
    if not synthetic_emails_path:
        synthetic_dir = 'synthetic_emails'
        if not os.path.exists(synthetic_dir):
            logger.error(f"Synthetic emails directory {synthetic_dir} not found")
            return
            
        # Find most recent synthetic email file
        synthetic_files = [f for f in os.listdir(synthetic_dir) 
                          if f.endswith('.json') and not f.endswith('_mappings.json')
                          and not f.endswith('_analysis.json')]
        
        if not synthetic_files:
            logger.error("No synthetic email files found!")
            return
        
        # Sort by timestamp in filename (newest first)
        synthetic_files.sort(reverse=True)
        synthetic_emails_path = os.path.join(synthetic_dir, synthetic_files[0])
    
    logger.info(f"Using synthetic email file: {synthetic_emails_path}")
    
    # Run appropriate evaluator
    try:
        if args.simple:
            # Initialize simple evaluator
            evaluator = SimpleEmailEvaluator(original_emails_path, synthetic_emails_path, max_emails)
            results = evaluator.evaluate()
            
            logger.info("\nEvaluation complete!")
            logger.info(f"De-identification score: {results['deidentification_score']:.2f}")
            logger.info(f"Indian context score: {results['indian_context_score']:.2f}")
            logger.info(f"Structure score: {results['structure_score']:.2f}")
            logger.info(f"Overall score: {results['overall_score']:.2f}")
            
            # Save results
            simple_results_path = os.path.join(output_dir, 'simple_evaluation_results.json')
            with open(simple_results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Simple evaluation results saved to {simple_results_path}")
            
        else:
            # Initialize comprehensive evaluator
            evaluator = SyntheticEmailEvaluator(original_emails_path, synthetic_emails_path, max_emails)
            results = evaluator.run_all_evaluations()
            
            if results and 'overall_quality_score' in results:
                logger.info(f"\nOverall quality score: {results['overall_quality_score']:.2f}")
            else:
                logger.warning("Comprehensive evaluation did not complete successfully")
    
    except Exception as e:
        logger.error(f"Error in evaluation process: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()