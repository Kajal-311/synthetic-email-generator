# Technical Report: Synthetic Email Generation for eDiscovery

## Introduction

eDiscovery, or electronic discovery, is the process of identifying, collecting, and producing digital information in response to legal cases or investigations. As digital data volumes expand, eDiscovery has become pivotal in modern legal practices, enabling thorough and efficient analysis of electronic evidence.

A significant challenge in developing AI applications for eDiscovery is the need for high-quality training data that respects privacy concerns. This project addresses this challenge by creating a synthetic email dataset based on the Enron corpus, transforming it into authentic Indian business communications while ensuring complete de-identification.

## Approach and Implementation

### Hybrid Architecture

We developed a hybrid system that combines the strengths of traditional Natural Language Processing (NLP) with modern Large Language Models (LLMs). This architecture leverages:

1. **Traditional NLP techniques** for entity detection, structural analysis, and consistent entity replacement
2. **Modern LLM capabilities** for contextual understanding, natural language generation, and cultural adaptation

This hybrid approach achieves superior results compared to either technique used alone, particularly in maintaining consistency and structure while transforming context.

### System Components

The implementation consists of four primary modules:

#### 1. Email Selection Module (`email_selection.py`)
- Extracts business-relevant emails from the Enron dataset
- Categorizes emails into business functions (financial, legal, project, etc.)
- Selects a representative sample for transformation

Key findings from dataset analysis:
- 50,000 Enron emails analyzed, identifying 10,838 business-relevant communications
- Business emails distributed across categories: Financial (40.2%), Legal (26.2%), Meeting (18.6%), Project (9.0%), Transaction (5.8%), General (0.2%)

#### 2. Email Generation Module (`email_generator.py`)
- Implements the SyntheticEmailGenerator class for email transformation
- Creates consistent entity mappings across multiple domains
- Integrates with OpenAI API for contextual transformation
- Provides fallback mechanisms for API failures

Key innovations:
- **Unified Entity Processing**: Analyzes headers and content together to ensure consistency
- **Cross-Reference System**: Maintains relationships between entities across email components
- **Domain-Specific Adaptation**: Tailors transformations to target industries (technology, finance, healthcare, agriculture, education)

#### 3. Email Evaluation Module (`email_evaluator.py`)
- Measures de-identification effectiveness
- Assesses Indian context integration
- Evaluates structural preservation
- Calculates comprehensive quality scores

Final evaluation metrics:
- De-identification Score: 1.00 (100%)
- Indian Context Score: 1.00 (100%)
- Structure Preservation Score: 0.81 (81%)
- Overall Quality Score: 0.94 (94%)

#### 4. Pipeline Orchestration (`synthetic_email_pipeline.py`)
- Coordinates the end-to-end process
- Manages configuration and logging
- Provides a unified command-line interface

### Technical Challenges and Solutions

#### Entity Consistency and Cross-Referencing

A significant challenge was maintaining consistency between email headers and content. We developed a unified entity processing approach that:

1. Analyzes the entire email (headers + content) as a single unit
2. Cross-references entities between components (e.g., matching email addresses to names)
3. Applies consistent transformations across all instances
4. Performs a final validation pass to catch any missed replacements

This approach achieved 100% entity consistency in our evaluation.

#### Token Limit Management

The OpenAI API has context length limitations (8,192 tokens). We addressed this by:

1. Implementing content chunking for long emails
2. Using a fallback rule-based transformation for emails exceeding token limits
3. Optimizing prompts to reduce token usage

This solution successfully handled all cases, with approximately 10% of emails (5 out of 50) requiring the fallback mechanism.

#### Cultural and Domain Adaptation

To create authentic Indian business communications, we implemented:

1. Indian name and location databases
2. Domain-specific terminology mappings for five industries
3. Currency conversion to Indian format (rupees, crores, lakhs)
4. Indian business expressions and communication patterns

This approach achieved a 100% Indian context score in our evaluation.

## Implementation Details

### Entity Detection and Mapping

The system uses a combination of regex patterns and rule-based detection to identify entities in emails:

```python
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
    
    # Additional entity extraction logic...

Unified Entity Processing
To ensure consistency, we implemented a unified entity processing approach:
def transform_email(self, email_data):
    """
    Transform an Enron email into a synthetic email with Indian context
    """
    # FIRST PASS: Create a unified text for entity identification
    from_email = email_data.get('from', '')
    to_email = email_data.get('to', '')
    subject = email_data.get('subject', '')
    content = email_data.get('content', '')
    
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

LLM Integration
The system integrates with the OpenAI API using carefully crafted prompts:
def transform_with_openai(self, original_email, target_domain, entity_mappings):
    """
    Transform email using OpenAI API
    """
    # Create a prompt for the transformation
    prompt = f"""
    Transform this email from an energy company (Enron) to an Indian {target_domain['name']} company.
    Maintain the same structure, tone, and purpose, but change all specific details to be relevant 
    to the new domain in an Indian context.
    
    Use these entity mappings exactly as provided - do not invent new mappings:
    {json.dumps(entity_mappings, indent=2)}
    
    This is a {email_category} email. Make sure the transformed email feels authentic for an
    Indian {target_domain['name']} company.
    
    Indian business specifics to incorporate:
    1. Use Indian greetings like "Namaste," "Warm regards," etc. where appropriate
    2. Reference Indian business practices and terminology 
    3. Use Indian locations from this list: {', '.join(self.indian_locations[:5])}
    4. If there are regulatory mentions, refer to Indian regulators (SEBI, RBI, etc.)
    5. Maintain the same paragraph structure and formatting as the original
    6. Keep a similar level of formality and professional tone
    7. Ensure all original business meaning is preserved in the transformation
    """
    
    # OpenAI API call and error handling...
Future Enhancements
Entity Relationship Management
To improve entity consistency and relationship tracking:
class EntityGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def add_entity(self, original_entity, mapped_entity, entity_type, domain):
        with self.driver.session() as session:
            session.run(
                "MERGE (o:OriginalEntity {name: $original, type: $type}) "
                "MERGE (m:MappedEntity {name: $mapped, domain: $domain}) "
                "MERGE (o)-[:MAPS_TO {domain: $domain}]->(m)",
                original=original_entity, mapped=mapped_entity, 
                type=entity_type, domain=domain
            )

Improved Token Management
To handle long emails more efficiently:
def optimize_for_token_limit(content, max_tokens=7000):
    # Estimate tokens (rough approximation: 4 chars ≈ 1 token)
    estimated_tokens = len(content) // 4
    
    if estimated_tokens <= max_tokens:
        return content
        
    # Split by paragraphs
    paragraphs = content.split('\n\n')
    
    # Process content in chunks
    if len(paragraphs) > 1:
        # Keep introduction and important parts
        essential_parts = paragraphs[0:2]
        conclusion_parts = paragraphs[-2:]
        
        # Calculate remaining token budget
        used_tokens = (len('\n\n'.join(essential_parts + conclusion_parts))) // 4
        remaining_tokens = max_tokens - used_tokens
        
        # Select middle paragraphs that fit within budget
        middle_paragraphs = []
        for p in paragraphs[2:-2]:
            p_tokens = len(p) // 4
            if remaining_tokens - p_tokens > 0:
                middle_paragraphs.append(p)
                remaining_tokens -= p_tokens
            else:
                break
                
        # Combine parts
        return '\n\n'.join(essential_parts + middle_paragraphs + conclusion_parts)

Domain-Specific Enhancement
To improve domain adaptation:
class DomainSpecificTransformer:
    def __init__(self):
        self.domain_models = {
            'finance': FinanceDomainModel(),
            'technology': TechnologyDomainModel(),
            'healthcare': HealthcareDomainModel(),
            'agriculture': AgricultureDomainModel(),
            'education': EducationDomainModel()
        }
        
    def transform(self, content, source_domain, target_domain):
        # Get domain-specific models
        source_model = self.domain_models.get(source_domain)
        target_model = self.domain_models.get(target_domain)
        
        if not source_model or not target_model:
            return self.fallback_transform(content)
            
        # Extract domain-specific concepts
        concepts = source_model.extract_concepts(content)
        
        # Map concepts to target domain
        mapped_concepts = source_model.map_to_target(concepts, target_model)
        
        # Generate text with target domain concepts
        return target_model.generate_with_concepts(content, mapped_concepts)

Conclusion
The hybrid approach to synthetic email generation demonstrates the potential for creating high-quality, de-identified datasets that preserve the essential characteristics of business communication while adapting to new contexts. The implementation achieves impressive results with a 1.0 de-identification score, 1.0 Indian context score, 0.81 structure preservation score, and 0.94 overall quality score.
Our synthetic email generation system successfully addresses the core requirements for eDiscovery applications, creating synthetic data that maintains the complexity and variability needed for LLM fine-tuning while ensuring complete privacy protection. The multi-stage pipeline provides a flexible framework that can be extended to accommodate additional domains, languages, and cultural contexts.
One of the key innovations in our approach is the unified entity processing system that ensures consistency between email headers and content. By analyzing the entire email as a single unit for entity detection and maintaining cross-references between different components, we've solved one of the most challenging issues in synthetic email generation - inconsistent entity replacement. This is particularly important for eDiscovery applications where maintaining entity relationships across documents is essential.
The domain distribution in our generated dataset (46% finance, 32% technology, 12% healthcare, 6% agriculture, and 4% education) reflects a realistic cross-section of Indian business communications, ensuring the dataset has broad applicability across industries. This diversity is important for training LLMs that need to understand different business contexts.
While the current implementation has some limitations in terms of API reliability and content length handling, these challenges can be addressed through the proposed enhancements. The future roadmap includes distributed processing architecture, advanced entity relationship management, improved token handling, and domain-specific enhancement that will transform this prototype into a robust, production-ready system.
The project demonstrates the viability of synthetic data as a solution to the challenge of developing AI applications in sensitive domains like eDiscovery, where access to real data is constrained by privacy considerations. By combining traditional NLP techniques with modern LLMs in a carefully designed pipeline, the system achieves a balance of control, quality, and efficiency that would be difficult to attain with either approach alone.