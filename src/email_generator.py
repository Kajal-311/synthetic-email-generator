import os
import re
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import logging
import json
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/email_transformation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_selected_emails(input_dir):
    """
    Load selected emails from JSON files in the input directory
    
    Args:
        input_dir: Directory containing JSON files with selected emails
        
    Returns:
        List of email dictionaries
    """

    
    logger = logging.getLogger(__name__)
    
    all_emails = []
    
    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Find all JSON files in the directory
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in the {input_dir} directory")
    
    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # Aggregate all emails from all JSON files
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                emails_data = json.load(f)
                
                # Handle both list and dictionary formats
                if isinstance(emails_data, list):
                    all_emails.extend(emails_data)
                elif isinstance(emails_data, dict) and 'emails' in emails_data:
                    all_emails.extend(emails_data['emails'])
                elif isinstance(emails_data, dict):
                    # If it's a single email as a dictionary
                    all_emails.append(emails_data)
                    
                logger.info(f"Loaded emails from {json_file}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Aggregated a total of {len(all_emails)} emails")
    return all_emails


class SyntheticEmailGenerator:
    """System for generating synthetic emails based on Enron emails with Indian context"""
    
    def __init__(self, entity_mapping_path=None):
        """
        Initialize the generator
        
        Args:
            entity_mapping_path: Path to existing entity mapping file (if any)
        """
        # Indian names
        self.indian_first_names = [
            'Aarav', 'Vivaan', 'Aditya', 'Vihaan', 'Arjun', 'Reyansh', 'Ayaan', 'Atharva', 
            'Krishna', 'Ishaan', 'Shaurya', 'Advik', 'Rudra', 'Pranav', 'Advaith', 'Dhruv',
            'Kabir', 'Ananya', 'Diya', 'Avni', 'Aanya', 'Aadhya', 'Aaradhya', 'Saanvi', 
            'Myra', 'Pari', 'Anika', 'Sara', 'Ahana', 'Anvi', 'Prisha', 'Riya', 'Ira', 
            'Amyra', 'Navya', 'Ritvik', 'Rohan', 'Neha', 'Rohit', 'Anjali', 'Vikram',
            'Meera', 'Aryan', 'Divya', 'Arnav', 'Tanvi', 'Karan', 'Pooja', 'Sameer'
        ]

        self.indian_last_names = [
            'Sharma', 'Patel', 'Singh', 'Kumar', 'Gupta', 'Verma', 'Shah', 'Mehta', 
            'Joshi', 'Rao', 'Nair', 'Reddy', 'Iyer', 'Pillai', 'Choudhary', 'Chatterjee', 
            'Mukherjee', 'Malhotra', 'Gandhi', 'Agarwal', 'Banerjee', 'Desai', 'Bose', 'Trivedi',
            'Roy', 'Kapoor', 'Das', 'Khanna', 'Saxena', 'Bhatia', 'Bhatt', 'Mehra', 'Yadav',
            'Chauhan', 'Hegde', 'Naidu', 'Gowda', 'Deshpande', 'Mishra', 'Kaur', 'Menon',
            'Patil', 'Chopra', 'Arora', 'Gill', 'Bajwa', 'Bakshi', 'Kulkarni', 'Shetty'
        ]
        
        # Indian locations
        self.indian_locations = [
            # North India
            'New Delhi', 'Gurugram', 'Noida', 'Chandigarh', 'Jaipur', 'Lucknow', 'Dehradun',
            # West India
            'Mumbai', 'Pune', 'Ahmedabad', 'Surat', 'Indore', 'Nagpur',
            # South India
            'Bengaluru', 'Hyderabad', 'Chennai', 'Kochi', 'Mysuru', 'Coimbatore',
            # East India
            'Kolkata', 'Bhubaneswar', 'Guwahati', 'Patna', 'Ranchi'
        ]
        
        # Load or initialize entity mappings
        if entity_mapping_path and os.path.exists(entity_mapping_path):
            with open(entity_mapping_path, 'r') as f:
                self.entity_mappings = json.load(f)
            logger.info(f"Loaded existing entity mappings from {entity_mapping_path}")
        else:
            self.entity_mappings = {
                'people': {},       # "Andrew Fastow" -> "Vikram Malhotra"
                'emails': {},       # "andrew.fastow@enron.com" -> "vikram.malhotra@techmindz.in"
                'companies': {},    # "Enron" -> "TechMindz"
                'locations': {},    # "Houston" -> "Bengaluru"
                'products': {},     # "synfuel" -> "hybrid seed"
                'projects': {},     # "Project Alpha" -> "Project Horizon"
                'departments': {},  # "Trading" -> "Data Science"
                'amounts': {}       # "$10 million" -> "₹83 crore"
            }
            logger.info("Initialized new entity mappings")
        
        # Initialize target domains with Indian companies and contexts
        self.target_domains = [
            {
                'name': 'technology',
                'companies': [
                    'TechMindz', 'InfoSphere', 'ZenithSoft', 'DataCraft', 'InnovateBharat',
                    'ByteWisdom', 'DigitalDrishti', 'CodeSangam', 'TechVidya', 'IndiaTech',
                    'CyberYuga', 'VirtualVeda', 'BharatSoft', 'IndiaStack', 'CloudMantra'
                ],
                'domain_suffix': [
                    'techmindz.in', 'infosphere.co.in', 'zenithsoft.tech', 'datacraft.in', 'innovatebharat.com',
                    'bytewisdom.in', 'digitaldrishti.net', 'codesangam.tech', 'techvidya.in', 'indiatech.co.in',
                    'cyberyuga.in', 'virtualveda.com', 'bharatsoft.in', 'indiastack.org', 'cloudmantra.tech'
                ],
                'products': [
                    'AI software solution', 'cloud migration platform', 'data analytics suite', 
                    'enterprise management system', 'digital transformation toolkit',
                    'cybersecurity framework', 'mobile app development platform', 
                    'automation solution', 'IoT integration system', 'blockchain implementation'
                ],
                'locations': self.indian_locations,
                'job_titles': [
                    'CTO', 'Software Developer', 'Project Manager', 'Data Scientist', 
                    'Solution Architect', 'DevOps Engineer', 'Product Manager', 
                    'QA Lead', 'Technology Director', 'IT Head'
                ]
            },
            {
                'name': 'healthcare',
                'companies': [
                    'AyushCare', 'SwasthyaSeva', 'JeevanHealth', 'ArogyaMitra', 'NiramayLife',
                    'JanAushadhi', 'VaidyaMed', 'RogyaNirnay', 'SwasthBharat', 'ChikitsaPlus',
                    'ArogyaTech', 'UpchaaryaHealth', 'JeevaRaksha', 'NidanHealth', 'SvasthyaSathi'
                ],
                'domain_suffix': [
                    'ayushcare.in', 'swasthyaseva.co.in', 'jeevanhealth.org', 'arogyamitra.com', 'niramaylife.in',
                    'janaushadhi.gov.in', 'vaidyamed.co.in', 'rogyanirnay.in', 'swasthbharat.org', 'chikitsaplus.in',
                    'arogyatech.com', 'upchaarya.in', 'jeevaraksha.org', 'nidanhealth.co.in', 'svasthyasathi.in'
                ],
                'products': [
                    'electronic health record system', 'telemedicine platform', 
                    'diagnostic management tool', 'healthcare analytics solution',
                    'patient management system', 'medical inventory software',
                    'clinical decision support system', 'remote monitoring platform',
                    'medical compliance framework', 'hospital management system'
                ],
                'locations': self.indian_locations,
                'job_titles': [
                    'Medical Director', 'Healthcare Administrator', 'Clinical Lead', 
                    'Research Coordinator', 'Chief Medical Officer', 'Wellness Program Manager',
                    'Health Informatics Specialist', 'Ayurvedic Practitioner', 'Hospital Manager',
                    'Medical Records Supervisor'
                ]
            },
            {
                'name': 'agriculture',
                'companies': [
                    'KisanBharti', 'AnnapurnaBio', 'FarmVriddhi', 'KrishiUnnati', 'ShetraKrishi',
                    'FasalTech', 'BijMantra', 'DhartiAgro', 'KhetiBharat', 'GraminTech',
                    'SasyaGrow', 'KrishiVikas', 'HaritKrishi', 'BharatSeeds', 'UttamFarms'
                ],
                'domain_suffix': [
                    'kisanbharti.org', 'annapurnabio.in', 'farmvriddhi.co.in', 'krishiunnati.com', 'shetrakrishi.in',
                    'fasaltech.org', 'bijmantra.in', 'dhartiagro.co.in', 'khetibharat.com', 'gramintech.org.in',
                    'sasyagrow.in', 'krishivikas.co.in', 'haritkrishi.com', 'bharatseeds.in', 'uttamfarms.org'
                ],
                'products': [
                    'hybrid seed variety', 'organic fertilizer', 'crop management system',
                    'irrigation solution', 'farm equipment', 'soil analysis kit',
                    'weather monitoring system', 'agricultural drone', 'crop protection formula',
                    'sustainable farming toolkit'
                ],
                'locations': self.indian_locations,
                'job_titles': [
                    'Agricultural Director', 'Farm Manager', 'Crop Specialist', 
                    'Supply Chain Manager', 'Agronomist', 'Rural Development Officer',
                    'Food Processing Head', 'Agricultural Scientist', 'Seed Development Manager',
                    'Organic Farming Consultant'
                ]
            },
            {
                'name': 'finance',
                'companies': [
                    'BharatCapital', 'IndiaWealth', 'DhanLaxmi', 'VibrantMoney', 'PaisaGrow',
                    'SuvidhaFinance', 'ArthaNeeti', 'VaibhavInvest', 'DhanVarsha', 'YuktiFin',
                    'SamriddhiCapital', 'NidhiVest', 'UtkarshFunds', 'PragatiWealth', 'UnnatiFinance'
                ],
                'domain_suffix': [
                    'bharatcapital.in', 'indiawealth.co.in', 'dhanlaxmi.com', 'vibrantmoney.in', 'paisagrow.co.in',
                    'suvidhafinance.com', 'arthaneeti.in', 'vaibhavinvest.co.in', 'dhanvarsha.in', 'yuktifin.com',
                    'samriddhicapital.in', 'nidhivest.co.in', 'utkarshfunds.in', 'pragatiwealth.com', 'unnatifinance.in'
                ],
                'products': [
                    'wealth management solution', 'micro-lending platform', 'digital payment system',
                    'mutual fund portfolio', 'financial planning tool', 'insurance package',
                    'tax optimization service', 'corporate treasury system', 'risk assessment framework',
                    'retail banking solution'
                ],
                'locations': self.indian_locations,
                'job_titles': [
                    'Financial Director', 'Investment Manager', 'Wealth Advisor', 
                    'Risk Assessment Officer', 'Branch Manager', 'Credit Analyst',
                    'Portfolio Manager', 'Lending Specialist', 'Compliance Officer',
                    'Financial Products Head'
                ]
            },
            {
                'name': 'education',
                'companies': [
                    'VidyaBhavan', 'GyanSagar', 'ShikshaMitra', 'BuddhibodhAcademy', 'VidyarthaLearning',
                    'PragyaEdu', 'VidyaVikasInstitute', 'GyanotsavEducation', 'SikshaVahini', 'VidyaPrabodhini',
                    'AkshargyanLearning', 'SaraswatiVidya', 'SamvitAcademy', 'VidyaMandir', 'BharatPathshala'
                ],
                'domain_suffix': [
                    'vidyabhavan.edu.in', 'gyansagar.org', 'shikshamitra.in', 'buddhibodh.ac.in', 'vidyartha.edu',
                    'pragyaedu.co.in', 'vidyavikas.edu.in', 'gyanotsav.org', 'sikshavahini.in', 'vidyaprabodhini.edu',
                    'akshargyan.in', 'saraswatividya.ac.in', 'samvitacademy.edu', 'vidyamandir.org', 'bharatpathshala.in'
                ],
                'products': [
                    'learning management system', 'digital curriculum', 'assessment platform',
                    'virtual classroom solution', 'educational content package', 'career guidance tool',
                    'skill development program', 'eLearning platform', 'academic management system',
                    'educational research database'
                ],
                'locations': self.indian_locations,
                'job_titles': [
                    'Principal', 'Academic Director', 'Curriculum Coordinator', 
                    'Educational Consultant', 'Professor', 'Research Chair',
                    'Dean of Students', 'Training Manager', 'eLearning Specialist',
                    'Educational Technology Head'
                ]
            }
        ]

    def apply_entity_mapping(self, text, entity_mappings, primary_type, target_domain):
        """
        Apply entity mappings to text, prioritizing specific entity types
        
        Args:
            text: Text to transform
            entity_mappings: Dictionary of entity mappings
            primary_type: Primary entity type to check first
            target_domain: Target domain information
            
        Returns:
            Transformed text
        """
        # First check if the exact text is in the primary entity type mappings
        if primary_type in entity_mappings and text in entity_mappings[primary_type]:
            return entity_mappings[primary_type][text]
        
        # Then check all entity types
        for entity_type, mappings in entity_mappings.items():
            for original, replacement in mappings.items():
                if original == text:
                    return replacement
        
        # If no exact match found, try to create mapping if the primary type is emails
        if primary_type == 'emails' and '@' in text:
            return self.get_or_create_mapping(text, 'emails', target_domain)
        
        # No mapping found, return original
        return text
    
    def ensure_entity_consistency(self, content, entity_mappings):
        """
        Perform final check to ensure all entities were consistently replaced
        
        Args:
            content: Transformed content
            entity_mappings: Dictionary of entity mappings
            
        Returns:
            Content with consistent entity replacements
        """
        # Check each entity type for any missed replacements
        for entity_type, mappings in entity_mappings.items():
            for original, replacement in mappings.items():
                if original and replacement and original in content:
                    # Use word boundaries for more precise replacement
                    pattern = r'\b' + re.escape(original) + r'\b'
                    content = re.sub(pattern, replacement, content)
        
        return content
    
    def save_entity_mappings(self, output_path):
        """
        Save entity mappings to a file
        
        Args:
            output_path: Path to save the mappings
        """
        with open(output_path, 'w') as f:
            json.dump(self.entity_mappings, f, indent=2)
        logger.info(f"Saved entity mappings to {output_path}")
    
    def select_target_domain(self, email_category):
        """
        Select an appropriate target domain based on email category
        
        Args:
            email_category: Category of the email
            
        Returns:
            Dictionary with target domain information
        """
        # Map email categories to suitable domains with weights for Indian context
        domain_weights = {
            'financial': [('finance', 0.7), ('technology', 0.3)],
            'legal': [('finance', 0.4), ('healthcare', 0.3), ('technology', 0.3)],
            'project': [('technology', 0.5), ('agriculture', 0.3), ('education', 0.2)],
            'meeting': [('technology', 0.4), ('education', 0.3), ('healthcare', 0.3)],
            'transaction': [('finance', 0.6), ('agriculture', 0.2), ('technology', 0.2)],
            'general': [('technology', 0.3), ('finance', 0.2), ('agriculture', 0.2), 
                      ('healthcare', 0.15), ('education', 0.15)]
        }
        
        # Get weights for this category or use default weights
        if email_category in domain_weights:
            # Use weighted selection for better distribution
            domains, weights = zip(*domain_weights[email_category])
            domain_name = random.choices(domains, weights=weights, k=1)[0]
        else:
            # If no specific weights, use the general category weights
            domains, weights = zip(*domain_weights['general'])
            domain_name = random.choices(domains, weights=weights, k=1)[0]
        
        # Find the domain data
        for domain in self.target_domains:
            if domain['name'] == domain_name:
                return domain
        
        # Default to technology if no match
        return next(domain for domain in self.target_domains if domain['name'] == 'technology')
    
    def get_or_create_mapping(self, original_entity, entity_type, target_domain):
        """
        Get existing mapping or create a new one for an entity
        
        Args:
            original_entity: Original entity from Enron email
            entity_type: Type of entity (people, companies, etc.)
            target_domain: Target domain information
            
        Returns:
            Mapped entity
        """
        # Check if mapping already exists
        if original_entity in self.entity_mappings[entity_type]:
            return self.entity_mappings[entity_type][original_entity]
        
        # Create new mapping based on entity type
        if entity_type == 'companies':
            # Randomly select a company from the target domain
            mapped_entity = random.choice(target_domain['companies'])
        
        elif entity_type == 'emails':
            # Extract username from email
            username_match = re.search(r'(.+)@', original_entity)
            if not username_match:
                return original_entity
            username = username_match.group(1)
            parts = username.split('.')
            
            # Generate new name parts if needed
            if len(parts) == 2:
                first, last = parts
            else:
                first = parts[0]
                last = f"user{random.randint(100, 999)}"
            
            # Create new email with target domain - using Indian domain suffix
            domain_suffix = random.choice(target_domain['domain_suffix'])
            mapped_entity = f"{first}.{last}@{domain_suffix}"
        
        elif entity_type == 'people':
            # Use Indian names for people entities
            mapped_entity = f"{random.choice(self.indian_first_names)} {random.choice(self.indian_last_names)}"
        
        elif entity_type == 'locations':
            mapped_entity = random.choice(target_domain['locations'])
        
        elif entity_type == 'products':
            mapped_entity = random.choice(target_domain['products'])
        
        elif entity_type == 'projects':
            project_names = ['Horizon', 'Genesis', 'Nova', 'Infinity', 'Quantum', 'Nexus', 'Summit', 'Pinnacle', 'Shakti', 'Pragati']
            mapped_entity = f"Project {random.choice(project_names)}"
        
        elif entity_type == 'departments':
            departments = {
                'technology': ['Engineering', 'Product', 'Data Science', 'Infrastructure', 'DevOps'],
                'healthcare': ['Clinical Research', 'Medical Affairs', 'Patient Care', 'Regulatory', 'Diagnostics'],
                'agriculture': ['Crop Development', 'Field Operations', 'Supply Chain', 'R&D', 'Farm Systems'],
                'finance': ['Investment Banking', 'Asset Management', 'Trading', 'Risk', 'Compliance'],
                'education': ['Academic Affairs', 'Research', 'Student Services', 'Admissions', 'Curriculum']
            }
            
            mapped_entity = random.choice(departments.get(target_domain['name'], ['Operations', 'Management']))
        
        elif entity_type == 'amounts':
            # Parse the original amount
            amount_match = re.search(r'\$\s*(\d+(?:[,.]\d+)?)\s*(million|billion|thousand|M|B|K)?', original_entity)
            if not amount_match:
                return original_entity
                
            amount_str = amount_match.group(1).replace(',', '')
            amount = float(amount_str)
            unit = amount_match.group(2) if amount_match.group(2) else ''
            
            # Convert to Indian Rupees and format using Indian terminology
            # USD to INR approximate conversion
            multiplier = 83  # Updated conversion rate
            
            if unit.lower() in ['million', 'm']:
                inr_amount = amount * multiplier
                # Convert to crores (1 crore = 10 million rupees)
                crore_amount = inr_amount / 10
                mapped_entity = f"₹{crore_amount:.2f} crore"
            elif unit.lower() in ['billion', 'b']:
                inr_amount = amount * multiplier
                # Convert to crores (1 billion = 100 crores)
                crore_amount = inr_amount * 10
                mapped_entity = f"₹{crore_amount:.2f} crore"
            elif unit.lower() in ['thousand', 'k']:
                inr_amount = amount * multiplier * 1000
                # Convert to lakhs (1 lakh = 100,000 rupees)
                lakh_amount = inr_amount / 100000
                mapped_entity = f"₹{lakh_amount:.2f} lakh"
            else:
                # Small amounts in thousands of rupees
                inr_amount = amount * multiplier
                if inr_amount > 100000:
                    # Convert to lakhs
                    lakh_amount = inr_amount / 100000
                    mapped_entity = f"₹{lakh_amount:.2f} lakh"
                else:
                    mapped_entity = f"₹{inr_amount:.0f}"
        
        else:
            # For other entity types, create a generic mapping
            mapped_entity = f"Synthetic_{entity_type}_{random.randint(1000, 9999)}"
        
        # Store the mapping for future reference
        self.entity_mappings[entity_type][original_entity] = mapped_entity
        
        return mapped_entity
    
    def update_dates(self, original_date_text):
        """
        Update dates to more recent times while preserving patterns
        
        Args:
            original_date_text: Original date string
            
        Returns:
            Updated date string
        """
        try:
            # Handle common date formats
            date_patterns = [
                # Format: 2001-06-07 07:48:00
                (r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})', 
                 lambda m: datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 
                                   int(m.group(4)), int(m.group(5)), int(m.group(6)))),
                
                # Format: Thu, 25 Jan 2001 09:21:00
                (r'(?:\w+, )?(\d{1,2}) (\w{3}) (\d{4}) (\d{1,2}):(\d{1,2}):(\d{1,2})', 
                 lambda m: datetime(int(m.group(3)), 
                                   {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 
                                    'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}[m.group(2)], 
                                   int(m.group(1)), int(m.group(4)), int(m.group(5)), int(m.group(6))))
            ]
            
            # Try to parse the date with different patterns
            original_date = None
            format_used = None
            
            for pattern, parser in date_patterns:
                match = re.search(pattern, original_date_text)
                if match:
                    original_date = parser(match)
                    format_used = pattern
                    break
            
            if not original_date or not format_used:
                return original_date_text
            
            # Move date forward by approximately 20-24 years
            years_forward = random.randint(20, 24)
            new_date = original_date + timedelta(days=365 * years_forward)
            
            # Format the new date according to the original format
            if format_used == date_patterns[0][0]:  # YYYY-MM-DD HH:MM:SS
                return new_date.strftime('%Y-%m-%d %H:%M:%S')
            elif format_used == date_patterns[1][0]:  # Weekday, DD MMM YYYY HH:MM:SS
                return new_date.strftime('%a, %d %b %Y %H:%M:%S')
            else:
                return original_date_text
        
        except Exception as e:
            logger.error(f"Error updating date {original_date_text}: {e}")
            return original_date_text
    
    def identify_entities(self, text):
        """
        Identify entities in text using improved pattern matching
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of identified entities
        """
        identified_entities = {
            'people': [],
            'emails': [],
            'companies': [],
            'locations': [],
            'products': [],
            'projects': [],
            'departments': [],
            'amounts': []
        }
        
        # Extract emails with improved pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        identified_entities['emails'] = list(set(re.findall(email_pattern, text)))
        
        # Extract monetary amounts with improved pattern
        amount_pattern = r'\$\s*\d+(?:,\d+)*(?:\.\d+)?\s*(?:million|billion|thousand|M|B|K)?'
        identified_entities['amounts'] = list(set(re.findall(amount_pattern, text)))
        
        # Extract company names (expanded for more coverage)
        company_candidates = [
            'Enron', 'DPR', 'Dakota', 'LLC', 'Corp', 'Inc', 'Company', 'Dynegy',
            'ExxonMobil', 'Chevron', 'BP', 'Shell', 'Reliant', 'El Paso', 'Halliburton',
            'Arthur Andersen', 'Merrill Lynch', 'Goldman Sachs', 'Morgan Stanley', 'Citigroup',
            'JPMorgan', 'Bank of America', 'Lehman Brothers', 'Duke Energy', 'Constellation'
        ]
        for company in company_candidates:
            pattern = r'\b' + re.escape(company) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                identified_entities['companies'].append(company)
        
        # Extract project names with improved pattern
        project_pattern = r'(?:Project|Initiative|Program)\s+[A-Z][a-z]+'
        identified_entities['projects'] = list(set(re.findall(project_pattern, text)))
        
        # Extract locations (expanded to include more US locations that appear in Enron emails)
        location_candidates = [
            'Houston', 'New York', 'Chicago', 'California', 'Texas', 'London', 'Boston',
            'Denver', 'Portland', 'San Francisco', 'Los Angeles', 'Seattle', 'Atlanta',
            'Dallas', 'Washington', 'D.C.', 'Sacramento', 'Austin', 'New Orleans',
            'San Diego', 'Florida', 'Miami', 'Philadelphia', 'US', 'USA', 'United States',
            'America', 'Canada', 'Mexico', 'UK', 'Europe', 'Asia'
        ]
        for location in location_candidates:
            pattern = r'\b' + re.escape(location) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                identified_entities['locations'].append(location)
        
        # Extract people names with improved patterns
        # Formal titles with names
        name_patterns = [
            r'(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # Titles with full names
            r'(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+',  # Titles with first names
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Potential full names
        ]
        
        for pattern in name_patterns:
            names = re.findall(pattern, text)
            identified_entities['people'].extend(names)
        
        # Add common Enron executive names
        executive_names = [
            'Andrew Fastow', 'Kenneth Lay', 'Jeffrey Skilling', 'Rick Buy', 'Richard Causey',
            'Rebecca Mark', 'Lou Pai', 'Greg Whalley', 'Mark Frevert', 'John Lavorato',
            'David Delainey', 'Mark Haedicke', 'Vince Kaminski', 'Ben Glisan', 'Ken Rice',
            'Jeff McMahon', 'Steven Kean', 'Stanley Horton', 'James Derrick', 'Michael Kopper'
        ]
        for name in executive_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, text):
                identified_entities['people'].append(name)
        
        # Remove duplicates and common false positives
        common_words = ['United States', 'Annual Report', 'Power Plant', 'Natural Gas', 
                       'Board Meeting', 'Project Manager', 'Chief Executive', 'Vice President']
        identified_entities['people'] = [name for name in set(identified_entities['people']) 
                                      if name not in common_words]
        
        # Extract department names (expanded list)
        department_candidates = [
            'Trading', 'Legal', 'HR', 'Finance', 'Operations', 'Marketing',
            'Accounting', 'Research', 'IT', 'Information Technology', 'Risk Management',
            'Regulatory Affairs', 'Government Relations', 'Public Relations', 'Sales',
            'Business Development', 'Strategic Planning', 'Treasury', 'Tax', 'Audit',
            'Compliance', 'Engineering', 'Human Resources', 'Administration', 'Executive'
        ]
        for dept in department_candidates:
            pattern = r'\b' + re.escape(dept) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                identified_entities['departments'].append(dept)
        
        # Extract products/services
        product_candidates = [
            'gas', 'oil', 'power', 'electricity', 'energy', 'natural gas', 'crude',
            'pipeline', 'storage', 'futures', 'derivatives', 'option', 'swap',
            'contract', 'transmission', 'generation', 'distribution', 'refinery'
        ]
        for product in product_candidates:
            pattern = r'\b' + re.escape(product) + r'\b'
            if re.search(pattern, text.lower(), re.IGNORECASE):
                identified_entities['products'].append(product)
        
        return identified_entities 
        
    def transform_with_openai(self, original_email, target_domain, entity_mappings):
        """
        Transform email using OpenAI API (Updated for v1.0.0+)
        
        Args:
            original_email: Dictionary containing original email data
            target_domain: Target domain information
            entity_mappings: Dictionary of entity mappings
            
        Returns:
            Transformed email text
        """
        # Extract relevant information
        original_content = original_email.get('content', '')
        original_subject = original_email.get('subject', '')
        email_category = original_email.get('category', 'general')

        if not original_subject and 'content' in original_email:
            subject_match = re.search(r'Subject:\s*([^\n]+)', original_content)
            if subject_match:
                original_subject = subject_match.group(1).strip()

        
        # Create a prompt for the transformation
        prompt = f"""
        Transform this email from an energy company (Enron) to an Indian {target_domain['name']} company called {random.choice(target_domain['companies'])}.
        Maintain the same structure, tone, and purpose, but change all specific details to be relevant to the new domain in an Indian context.
        
        Use these entity mappings exactly as provided - do not invent new mappings:
        {json.dumps(entity_mappings, indent=2)}
        
        This is a {email_category} email. Make sure the transformed email feels authentic for an Indian {target_domain['name']} company.
        Use appropriate Indian terminology, locations, and business contexts. If monetary values are mentioned, convert to Indian Rupees using crores and lakhs.
        
        Indian business specifics to incorporate:
        1. Use Indian greetings like "Namaste," "Warm regards," etc. where appropriate
        2. Reference Indian business practices and terminology 
        3. Use Indian locations from this list: {', '.join(self.indian_locations[:5])}
        4. If there are regulatory mentions, refer to Indian regulators (SEBI, RBI, etc.) instead of US ones
        5. Maintain the same paragraph structure and formatting as the original
        6. Keep a similar level of formality and professional tone
        7. Ensure all original business meaning is preserved in the transformation
        
        Original email:
        Subject: {original_subject}
        
        {original_content}
        
        Transformed email:
        """
        
        try:
            # Updated OpenAI API call for v1.0.0+
            import openai
            client = openai.OpenAI()  # Initialize the client
            
            response = client.chat.completions.create(
                model="gpt-4",  # Use appropriate model based on your access
                messages=[
                    {"role": "system", "content": "You are a system that transforms business emails from a US energy company context to an Indian business context while preserving their structure, purpose, and tone. Create realistic synthetic emails that look like authentic correspondence from Indian companies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,  # Lower temperature for more consistent results
                max_tokens=1500
            )
            
            # Extract the transformed content (updated for new API format)
            transformed_text = response.choices[0].message.content.strip()
            
            # Remove "Transformed email:" if it appears at the beginning
            transformed_text = re.sub(r'^Transformed email:\s*', '', transformed_text)
            
            # Validate the transformation
            transformed_text = self.validate_transformation(original_content, transformed_text, entity_mappings)
            
            return transformed_text
        
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Fallback: Simple rule-based transformation
            logger.info("Falling back to simple rule-based transformation")
            return self.simple_transform(original_email, target_domain, entity_mappings)
    
    def validate_transformation(self, original, transformed, entity_mappings):
        """
        Validate that the transformation maintained key structural elements
        
        Args:
            original: Original email content
            transformed: Transformed email content
            entity_mappings: Dictionary of entity mappings
            
        Returns:
            Validated and potentially corrected transformed text
        """
        # Check for missing entities
        for entity_type, mappings in entity_mappings.items():
            for original_entity, new_entity in mappings.items():
                if original_entity in original and new_entity not in transformed:
                    # Add the missing entity
                    transformed = transformed.replace(original_entity, new_entity)
        
        # Check structure preservation
        orig_lines = original.split('\n')
        trans_lines = transformed.split('\n')
        
        # If significant length difference, try to fix structure
        if abs(len(orig_lines) - len(trans_lines)) > len(orig_lines) * 0.3:
            # Try to restore paragraph structure
            paragraphs = re.split(r'\n\s*\n', original)
            if len(paragraphs) > 1 and len(paragraphs) < 10:  # Reasonable number of paragraphs
                # Simple formatting restoration
                transformed += "\n\n[Note: Please maintain original paragraph structure]"
        
        return transformed
    
    def simple_transform(self, original_email, target_domain, entity_mappings):
        """
        Enhanced rule-based transformation as fallback
        
        Args:
            original_email: Dictionary containing original email data
            target_domain: Target domain information
            entity_mappings: Dictionary of entity mappings
            
        Returns:
            Transformed email text
        """
        content = original_email.get('content', '')
        
        # Replace entities based on mappings
        for entity_type, mappings in entity_mappings.items():
            for original, replacement in mappings.items():
                if original and replacement:  # Avoid empty strings
                    # Use word boundaries for more precise replacement
                    pattern = r'\b' + re.escape(original) + r'\b'
                    content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        # Replace dates
        date_patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'(?:\w+, )?\d{1,2} \w{3} \d{4} \d{1,2}:\d{2}:\d{2}'
        ]
        
        for pattern in date_patterns:
            for date_match in re.finditer(pattern, content):
                original_date = date_match.group(0)
                new_date = self.update_dates(original_date)
                content = content.replace(original_date, new_date)
        
        # Replace industry-specific terms with Indian context
        industry_terms = {
            'technology': {
                'energy': 'technology',
                'power': 'computing power',
                'pipeline': 'data pipeline',
                'gas': 'cloud storage',
                'oil': 'data',
                'well': 'server',
                'drilling': 'programming',
                'reservoir': 'database',
                'Enron': 'TechMindz',
                'Houston': 'Bengaluru',
                'Texas': 'Karnataka',
                'US': 'India',
                'USA': 'India',
                'United States': 'India',
                'America': 'India',
                'CEO': 'CEO',
                'CFO': 'CFO',
                'CTO': 'CTO',
                'VP': 'VP'
            },
            'healthcare': {
                'energy': 'healthcare',
                'power': 'medical care',
                'pipeline': 'treatment pathway',
                'gas': 'medical supplies',
                'oil': 'pharmaceutical',
                'well': 'patient',
                'drilling': 'diagnosing',
                'reservoir': 'patient database',
                'Enron': 'AyushCare',
                'Houston': 'Mumbai',
                'Texas': 'Maharashtra',
                'US': 'India',
                'USA': 'India',
                'United States': 'India',
                'America': 'India',
                'CEO': 'CEO',
                'CFO': 'CFO',
                'CTO': 'Chief Medical Officer',
                'VP': 'Medical Director'
            },
            'agriculture': {
                'energy': 'agriculture',
                'power': 'farming capacity',
                'pipeline': 'supply chain',
                'gas': 'fertilizer',
                'oil': 'crop',
                'well': 'field',
                'drilling': 'planting',
                'reservoir': 'grain silo',
                'Enron': 'KisanBharti',
                'Houston': 'Punjab',
                'Texas': 'Haryana',
                'US': 'India',
                'USA': 'India',
                'United States': 'India',
                'America': 'India',
                'CEO': 'CEO',
                'CFO': 'CFO',
                'CTO': 'Agricultural Director',
                'VP': 'Farm Operations Head'
            },
            'finance': {
                'energy': 'finance',
                'power': 'market power',
                'pipeline': 'investment pipeline',
                'gas': 'liquid assets',
                'oil': 'capital',
                'well': 'fund',
                'drilling': 'investing',
                'reservoir': 'portfolio',
                'Enron': 'BharatCapital',
                'Houston': 'Mumbai',
                'Texas': 'Maharashtra',
                'US': 'India',
                'USA': 'India',
                'United States': 'India',
                'America': 'India',
                'CEO': 'CEO',
                'CFO': 'CFO',
                'CTO': 'Chief Investment Officer',
                'VP': 'Senior Portfolio Manager'
            },
            'education': {
                'energy': 'education',
                'power': 'academic resources',
                'pipeline': 'curriculum',
                'gas': 'learning materials',
                'oil': 'knowledge',
                'well': 'classroom',
                'drilling': 'teaching',
                'reservoir': 'library',
                'Enron': 'VidyaBhavan',
                'Houston': 'Delhi',
                'Texas': 'Uttar Pradesh',
                'US': 'India',
                'USA': 'India',
                'United States': 'India',
                'America': 'India',
                'CEO': 'Director',
                'CFO': 'Administrative Head',
                'CTO': 'Academic Director',
                'VP': 'Department Chair'
            }
        }
        
        domain_name = target_domain['name']
        if domain_name in industry_terms:
            for original, replacement in industry_terms[domain_name].items():
                # Use word boundaries to avoid partial replacements
                pattern = r'\b' + re.escape(original) + r'\b'
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        # Add Indian English expressions and business terminology
        indian_business_terms = {
            'please revert': 'please respond',
            'kindly do the needful': 'please take necessary action',
            'prepone': 'move to an earlier time',
            'lakh': '100,000',
            'crore': '10 million',
            'today morning': 'this morning',
            'today evening': 'this evening'
        }
        
        # Only change expressions, not values
        for term, meaning in indian_business_terms.items():
            if term in content.lower():
                pattern = r'\b' + term + r'\b'
                # Don't replace terms in amounts
                if term not in ['lakh', 'crore']:
                    content = re.sub(pattern, meaning, content, flags=re.IGNORECASE)
        
        # Add Indian greeting if missing
        if 'dear' not in content.lower() and 'hi' not in content.lower() and 'hello' not in content.lower():
            lines = content.split('\n')
            has_greeting = False
            for i, line in enumerate(lines):
                if line.strip().endswith(',') and i < 5:
                    has_greeting = True
                    break
            
            if not has_greeting:
                first_blank_line = -1
                for i, line in enumerate(lines):
                    if not line.strip() and i < 10:
                        first_blank_line = i
                        break
                
                if first_blank_line > 0:
                    lines.insert(first_blank_line + 1, "Namaste,\n")
                else:
                    lines.insert(0, "Namaste,\n")
                
                content = '\n'.join(lines)
        
        # Add Indian-style closing if missing
        if 'regards' not in content.lower() and 'thank' not in content.lower() and 'sincerely' not in content.lower():
            indian_closings = [
                '\nWarm Regards,',
                '\nRegards,',
                '\nWith Best Regards,',
                '\nThank You,',
                '\nSincerely,'
            ]
            
            # Find a good place for the closing - near the end but not in a signature
            lines = content.split('\n')
            has_closing = False
            for i in range(len(lines) - 5, len(lines) - 1):
                if i >= 0 and any(closing.lower()[1:].strip() in lines[i].lower() for closing in indian_closings):
                    has_closing = True
                    break
            
            if not has_closing:
                # Add before any signature
                signature_line = -1
                for i, line in enumerate(lines):
                    if line.strip() == '--' or line.strip() == '---' or line.strip() == '_________':
                        signature_line = i
                        break
                
                if signature_line > 0:
                    lines.insert(signature_line, random.choice(indian_closings))
                else:
                    lines.append("\n" + random.choice(indian_closings))
                
                content = '\n'.join(lines)
        
        return content

    def transform_subject(self, subject, entity_mappings):
        """
        Transform email subject using entity mappings
        
        Args:
            subject: Original subject
            entity_mappings: Dictionary of entity mappings
            
        Returns:
            Transformed subject
        """
        if not subject:
            return subject
            
        transformed_subject = subject
        
        # Replace entities in subject
        for entity_type, mappings in entity_mappings.items():
            for original, replacement in mappings.items():
                if original and replacement:  # Avoid empty strings
                    # Use word boundaries for more precise replacement
                    pattern = r'\b' + re.escape(original) + r'\b'
                    transformed_subject = re.sub(pattern, replacement, transformed_subject, flags=re.IGNORECASE)
        
        return transformed_subject
    
    def batch_transform(self, emails_data, output_path):
        """
        Transform a batch of emails and save results
        
        Args:
            emails_data: List of email dictionaries
            output_path: Path to save transformed emails
            
        Returns:
            List of transformed emails
        """
        transformed_emails = []
        
        # Load Enron specific terminology mappings
        enron_terms = self.load_enron_specific_terminology()
        
        for i, email in enumerate(emails_data):
            logger.info(f"\nTransforming email {i+1}/{len(emails_data)}")
            
            # Pre-process for Enron specific terminology
            if isinstance(email.get('content', ''), str):
                email['content'] = self.replace_enron_terminology(email['content'], enron_terms)
            
            transformed_email = self.transform_email(email)
            transformed_emails.append(transformed_email)
            
            # Periodically save results
            if (i + 1) % 5 == 0 or (i + 1) == len(emails_data):
                with open(output_path, 'w') as f:
                    json.dump(transformed_emails, f, indent=2)
                logger.info(f"Saved {len(transformed_emails)} transformed emails to {output_path}")
                
                # Also save entity mappings
                mappings_path = os.path.splitext(output_path)[0] + "_mappings.json"
                self.save_entity_mappings(mappings_path)
        
        return transformed_emails
    
    def load_enron_specific_terminology(self):
        """
        Load Enron-specific terminology to improve transformations
        
        Returns:
            Dictionary of Enron terms and their replacements
        """
        # This could be loaded from a file in a real implementation
        # For now, we'll hardcode some common Enron terms
        return {
            # Enron business units and subsidiaries
            'EnronOnline': {'tech': 'TechGlobalOnline', 'finance': 'BharatInvestOnline', 
                          'healthcare': 'AyushCareOnline', 'agriculture': 'KisanMandiOnline', 
                          'education': 'VidyaPathOnline'},
            'EBS': {'tech': 'TBS', 'finance': 'BFS', 'healthcare': 'HBS', 
                   'agriculture': 'ABS', 'education': 'EBS'},
            'Enron North America': {'tech': 'TechMindz India', 'finance': 'BharatCapital India', 
                                 'healthcare': 'AyushCare India', 'agriculture': 'KisanBharti India', 
                                 'education': 'VidyaBhavan India'},
            'ENA': {'tech': 'TMI', 'finance': 'BCI', 'healthcare': 'ACI', 
                   'agriculture': 'KBI', 'education': 'VBI'},
            'ENE': {'tech': 'TMZ', 'finance': 'BCL', 'healthcare': 'ACL', 
                   'agriculture': 'KBL', 'education': 'VBL'},
            
            # Enron job titles and departments
            'Gas Trading': {'tech': 'Software Development', 'finance': 'Asset Management', 
                         'healthcare': 'Patient Care', 'agriculture': 'Crop Management', 
                         'education': 'Curriculum Development'},
            'Power Trading': {'tech': 'Cloud Infrastructure', 'finance': 'Portfolio Management', 
                           'healthcare': 'Medical Services', 'agriculture': 'Farm Operations', 
                           'education': 'Academic Programs'},
            'Origination': {'tech': 'Product Development', 'finance': 'Client Acquisition', 
                         'healthcare': 'Patient Enrollment', 'agriculture': 'Farmer Network', 
                         'education': 'Student Recruitment'},
            
            # Enron-specific business terms
            'DASH': {'tech': 'Project Overview', 'finance': 'Investment Summary', 
                   'healthcare': 'Treatment Plan', 'agriculture': 'Farm Plan', 
                   'education': 'Curriculum Plan'},
            'EOL': {'tech': 'TGL', 'finance': 'BIL', 'healthcare': 'ACL', 
                  'agriculture': 'KML', 'education': 'VPL'},
            'FERC': {'tech': 'TRAI', 'finance': 'SEBI', 'healthcare': 'NMC', 
                   'agriculture': 'APEDA', 'education': 'UGC'}
        }
    
    def replace_enron_terminology(self, content, enron_terms):
        """
        Replace Enron-specific terminology before the main transformation
        
        Args:
            content: Email content
            enron_terms: Dictionary of Enron terms and their replacements
            
        Returns:
            Content with some Enron-specific terms replaced
        """
        if not isinstance(content, str):
            return content
            
        # Do some basic replacements for very Enron-specific terms
        for term, replacements in enron_terms.items():
            if term in content:
                # We'll use the technology replacement as default for now
                # The domain-specific replacement will be handled in the main transformation
                replacement = replacements.get('tech', term)
                pattern = r'\b' + re.escape(term) + r'\b'
                content = re.sub(pattern, replacement, content)
        
        return content
        
    def transform_email(self, email_data):
        """
        Transform an Enron email into a synthetic email with Indian context
        
        Args:
            email_data: Dictionary containing original email data
            
        Returns:
            Dictionary with transformed email
        """
        # Select target domain based on email category
        email_category = email_data.get('category', 'general')
        target_domain = self.select_target_domain(email_category)
        
        logger.info(f"Transforming {email_category} email to Indian {target_domain['name']} domain")
        
        # FIRST PASS: Create a unified text for entity identification
        # Combine all fields that might contain entities to ensure consistency
        from_email = email_data.get('from', '')
        to_email = email_data.get('to', '')
        subject = email_data.get('subject', '')
        content = email_data.get('content', '')
        original_date = email_data.get('date', '')
        
        # Handle missing header fields by extracting from content if possible
        if (not from_email or not to_email or not original_date or not subject) and 'content' in email_data:
            content_text = email_data.get('content', '')
            
            # Try to extract missing fields from content
            if not from_email:
                from_match = re.search(r'From:\s*([^\n]+)', content_text)
                if from_match:
                    from_email = from_match.group(1).strip()
                    
            if not to_email:
                to_match = re.search(r'To:\s*([^\n]+)', content_text)
                if to_match:
                    to_email = to_match.group(1).strip()
                    
            if not original_date:
                date_match = re.search(r'Date:\s*([^\n]+)', content_text)
                if date_match:
                    original_date = date_match.group(1).strip()
                    
            if not subject:
                subject_match = re.search(r'Subject:\s*([^\n]+)', content_text)
                if subject_match:
                    subject = subject_match.group(1).strip()
        
        # Combine all text fields for comprehensive entity detection
        combined_text = f"From: {from_email}\nTo: {to_email}\nSubject: {subject}\n\n{content}"
        
        # Identify entities across the entire email
        identified_entities = self.identify_entities(combined_text)
        
        # Create mappings for identified entities
        entity_mappings = {}
        for entity_type, entities in identified_entities.items():
            entity_mappings[entity_type] = {}
            for entity in entities:
                mapped_entity = self.get_or_create_mapping(entity, entity_type, target_domain)
                entity_mappings[entity_type][entity] = mapped_entity
        
        # Extract email usernames to ensure consistent transformations
        email_usernames = {}
        for email_addr in identified_entities['emails']:
            username_match = re.search(r'(.+)@', email_addr)
            if username_match:
                username = username_match.group(1).lower()
                # Store for cross-reference in people names
                if username not in email_usernames:
                    email_usernames[username] = email_addr
        
        # Ensure consistency between email addresses and people names
        for person in identified_entities['people']:
            words = person.split()
            if len(words) >= 2:
                # Try various combinations that might match email usernames
                potential_usernames = [
                    words[0].lower(),  # First name
                    words[-1].lower(),  # Last name
                    f"{words[0].lower()}.{words[-1].lower()}",  # first.last
                    f"{words[-1].lower()}.{words[0].lower()}"   # last.first
                ]
                
                for username in potential_usernames:
                    if username in email_usernames:
                        # Get the corresponding email address
                        email_addr = email_usernames[username]
                        # Ensure the person name maps to the same Indian name used in the email
                        if email_addr in entity_mappings['emails']:
                            mapped_email = entity_mappings['emails'][email_addr]
                            # Extract the name part from the mapped email
                            name_match = re.search(r'(.+)@', mapped_email)
                            if name_match:
                                # Generate consistent Indian name
                                mapped_name_parts = name_match.group(1).split('.')
                                if len(mapped_name_parts) >= 2:
                                    # Create a consistent name based on email username
                                    first_name = mapped_name_parts[0].capitalize()
                                    last_name = mapped_name_parts[1].capitalize()
                                    consistent_name = f"{first_name} {last_name}"
                                    
                                    # Update the entity mapping for this person
                                    entity_mappings['people'][person] = consistent_name
        
        # SECOND PASS: Apply entity mappings consistently to all parts
        # Transform email addresses maintaining consistency
        new_from_email = from_email
        if '@' in from_email:
            new_from_email = self.apply_entity_mapping(from_email, entity_mappings, 'emails', target_domain)
            
        new_to_email = to_email
        if '@' in to_email:
            new_to_email = self.apply_entity_mapping(to_email, entity_mappings, 'emails', target_domain)
        
        # Update date with consistent transformation
        new_date = self.update_dates(original_date)
        
        # Transform the subject with entity mappings
        new_subject = self.transform_subject(subject, entity_mappings)
        
        # Check for attachment references and transform them
        has_attachment = self.check_for_attachment_references(content)
        
        # Transform the content using OpenAI (or fallback to simple transform)
        # Make sure to provide the complete entity mappings
        transformed_content = self.transform_with_openai(email_data, target_domain, entity_mappings)
        
        # Check if the transformation maintained attachment references
        if has_attachment and "attach" not in transformed_content.lower():
            # Add a reminder about attachment if it was lost in transformation
            transformed_content = self.add_attachment_reference(transformed_content, target_domain)
        
        # Perform a final consistency check on all transformed entities
        transformed_content = self.ensure_entity_consistency(transformed_content, entity_mappings)
        
        # Construct transformed email with consistent entities
        transformed_email = {
            'original_id': email_data.get('message_id', ''),
            'date': new_date,
            'from': new_from_email,
            'to': new_to_email,
            'subject': new_subject,
            'content': transformed_content,
            'category': email_category,
            'source_domain': 'energy',
            'target_domain': target_domain['name'],
            'indian_context': True
        }
        
        return transformed_email        
    def check_for_attachment_references(self, content):
        """
        Check if an email references attachments
        
        Args:
            content: Email content
            
        Returns:
            Boolean indicating if the email has attachment references
        """
        attachment_keywords = [
            'attach', 'enclosed', 'attached', 'attachment', 
            '.doc', '.xls', '.pdf', '.ppt', 'file', 'document'
        ]
        
        return any(keyword in content.lower() for keyword in attachment_keywords)
    
    def add_attachment_reference(self, content, target_domain):
        """
        Add an appropriate attachment reference if one was lost in transformation
        
        Args:
            content: Transformed email content
            target_domain: Target domain information
            
        Returns:
            Updated content with attachment reference
        """
        # Get domain-specific attachment types
        attachment_types = {
            'technology': ['technical specification', 'product roadmap', 'system design document'],
            'healthcare': ['medical report', 'patient data analysis', 'clinical study'],
            'agriculture': ['crop analysis', 'yield projection', 'field survey data'],
            'finance': ['financial statement', 'investment portfolio', 'market analysis'],
            'education': ['curriculum plan', 'student assessment', 'research paper']
        }
        
        domain_name = target_domain['name']
        attachment_type = random.choice(attachment_types.get(domain_name, ['document']))
        
        # Find a good place to mention the attachment
        lines = content.split('\n')
        
        # Check if there's already a closing line to add it before
        for i, line in enumerate(lines):
            if i > len(lines) // 2 and ("regards" in line.lower() or "thank" in line.lower() or "sincerely" in line.lower()):
                # Add before closing
                attachment_line = f"\nPlease find the {attachment_type} attached for your reference.\n"
                lines.insert(i, attachment_line)
                return '\n'.join(lines)
        
        # If no good place found, add near the end
        if len(lines) > 5:
            attachment_line = f"\nPlease find the {attachment_type} attached for your reference.\n"
            lines.insert(len(lines) - 3, attachment_line)
        else:
            # Very short email, add at the end
            lines.append(f"\nPlease find the {attachment_type} attached for your reference.")
        
        return '\n'.join(lines)
def main():
    """Main execution function"""
    # Configuration
    input_dir = 'selected_emails'
    output_dir = 'synthetic_emails'
    api_key = os.getenv("OPENAI_API_KEY")  # Get API key from environment variable
    mappings_file = 'config/entity_mappings.json'  # Use a more descriptive name
       
    # Set OpenAI API key if provided
    if api_key:
        import openai
        openai.api_key = api_key
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"indian_synthetic_emails_{timestamp}.json")
    
    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Find all JSON files in the directory
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in the {input_dir} directory")
    
    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # Aggregate all emails from all JSON files
    all_emails = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                emails_data = json.load(f)
                
                # Handle both list and dictionary formats
                if isinstance(emails_data, list):
                    all_emails.extend(emails_data)
                elif isinstance(emails_data, dict) and 'emails' in emails_data:
                    all_emails.extend(emails_data['emails'])
                elif isinstance(emails_data, dict):
                    # If it's a single email as a dictionary
                    all_emails.append(emails_data)
                    
                logger.info(f"Loaded emails from {json_file}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Aggregated a total of {len(all_emails)} emails for transformation")
    
    # Initialize generator
    generator = SyntheticEmailGenerator(mappings_file)
    
    # Transform emails
    transformed_emails = generator.batch_transform(all_emails, output_path)
    
    logger.info(f"\nTransformation complete. {len(transformed_emails)} emails transformed to Indian context.")
    
    # Print a sample transformed email
    if transformed_emails:
        logger.info("\nSample transformed email:")
        sample = transformed_emails[0]
        logger.info(f"From: {sample['from']}")
        logger.info(f"To: {sample['to']}")
        logger.info(f"Subject: {sample['subject']}")
        logger.info(f"Date: {sample['date']}")
        logger.info(f"Target Domain: {sample['target_domain']}")
        logger.info("Content preview:")
        logger.info(sample['content'][:300] + "...")
        
        # Save a detailed analysis of the transformations
        analysis_path = os.path.join(output_dir, f"transformation_analysis_{timestamp}.json")
        transformation_analysis = {
            "total_emails": len(transformed_emails),
            "source_files": json_files,
            "domain_distribution": {},
            "entity_replacements": {
                "people": len(generator.entity_mappings['people']),
                "companies": len(generator.entity_mappings['companies']),
                "locations": len(generator.entity_mappings['locations']),
                "emails": len(generator.entity_mappings['emails']),
                "amounts": len(generator.entity_mappings['amounts'])
            },
            "sample_mappings": {
                "people": dict(list(generator.entity_mappings['people'].items())[:5]),
                "companies": dict(list(generator.entity_mappings['companies'].items())[:5]),
                "locations": dict(list(generator.entity_mappings['locations'].items())[:5]),
                "emails": dict(list(generator.entity_mappings['emails'].items())[:5])
            }
        }
        
        # Count domain distribution
        for email in transformed_emails:
            domain = email['target_domain']
            transformation_analysis["domain_distribution"][domain] = transformation_analysis["domain_distribution"].get(domain, 0) + 1
        
        with open(analysis_path, 'w') as f:
            json.dump(transformation_analysis, f, indent=2)
        
        logger.info(f"Saved transformation analysis to {analysis_path}")
        
if __name__ == "__main__":
    main()