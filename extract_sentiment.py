# Sentiment extraction functions for movie reviews
import os
import json
import time
import logging
from datetime import datetime
import pandas as pd
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from utils import get_key

# Load environment variables
load_dotenv()

# Global variables for API key rotation
GEMINI_KEYS = []
CURRENT_KEY_INDEX = 0
CONSECUTIVE_FAILED_SWITCHES = 0  # Track consecutive failed switches
LAST_SUCCESSFUL_KEY_INDEX = -1   # Track last successful key


# ============================================================================
# LEVEL 0 FUNCTIONS - Public API
# ============================================================================

def process_movie_reviews(
    movie_review_df: pd.DataFrame,
    resume_mode: bool = False,
    output_file: str = 'output/processed_reviews.csv',
    log_file: str = 'processing_log.txt',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process movie reviews with incremental save (auto-save after each review and flush after each movie).
    
    This function saves each processed review immediately to prevent data loss from crashes or interruptions.
    It can resume from the last processed review if interrupted.
    
    Args:
        movie_review_df (pd.DataFrame): DataFrame with columns 'id', 'title', and 'all_reviews'
        resume_mode (bool): If True, continue from last processed review. If False, start from scratch
        output_file (str): Path to output CSV file (default: 'processed_reviews_incremental.csv')
        log_file (str): Path to log file (default: 'processing_log.txt')
        verbose (bool): If True, print progress to console
    
    Returns:
        pd.DataFrame: DataFrame with processed reviews (same as reading output_file)
    
    Example:
        # First run - process from beginning
        df_result = process_movie_reviews_incremental(df, resume_mode=False)
        
        # If interrupted, resume from where it stopped
        df_result = process_movie_reviews_incremental(df, resume_mode=True)
    """
    _initialize_gemini_client()
    
    start_index, write_header = _determine_starting_point(
        movie_review_df, resume_mode, output_file, log_file, verbose
    )
    
    _log_to_file(log_file, f"Total movies to process: {len(movie_review_df)}")
    
    cumulative_index = 0
    total_reviews_processed = 0
    
    for movie_idx, row in movie_review_df.iterrows():
        title, reviews = _extract_movie_data(row)
        
        _log_movie_progress(movie_idx, len(movie_review_df), title, len(reviews), log_file, verbose)
        
        skipped_count = 0
        for review_idx, review in enumerate(reviews):
            if cumulative_index <= start_index:
                cumulative_index += 1
                skipped_count += 1
                if verbose:
                    print(f"  ⏭️  Skipping review {review_idx + 1}/{len(reviews)} (already processed)")
                continue
            
            # Show skipped summary if any reviews were skipped
            if skipped_count > 0 and review_idx == skipped_count:
                print(f"  ⏭️  Skipped {skipped_count} already processed reviews")
                skipped_count = 0
            
            write_header = _process_and_save_review(
                review, review_idx, len(reviews), title, title,
                output_file, write_header, log_file, verbose
            )
            
            total_reviews_processed += 1
            cumulative_index += 1
        
        _finalize_movie_processing(movie_idx, title, total_reviews_processed, log_file, verbose)
    
    _display_incremental_summary(total_reviews_processed, output_file, log_file, verbose)
    
    return pd.read_csv(output_file)


# ============================================================================
# LEVEL 1 FUNCTIONS - Called by Level 0
# ============================================================================

def _create_logger():
    """Create base logger instance"""
    logger = logging.getLogger('movie_review_processor')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    return logger


def _create_file_handler(log_file: str):
    """Create and configure file handler for logger"""
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    return file_handler


def _setup_logging(verbose: bool):
    """Setup logger and log file if not verbose mode"""
    if verbose:
        return None, None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'processing_log_{timestamp}.log'
    logger = __setup_logger(log_file)
    logger.info(f"=== Starting movie review processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    return logger, log_file


def _create_log_message_handler(verbose: bool, logger):
    """Create a logging function based on verbose setting"""
    def log_message(message: str, level: str = 'info', show_progress: bool = False):
        """Helper function to handle logging/printing based on verbose setting"""
        if verbose:
            print(message)
        else:
            if logger:
                if level == 'info':
                    logger.info(message)
                elif level == 'warning':
                    logger.warning(message)
                elif level == 'error':
                    logger.error(message)
            if show_progress:
                print(message)
    return log_message


def _initialize_gemini_client():
    """Initialize Gemini API client with multiple keys support"""
    gemini_keys = get_key('gemini')
    # Store keys globally for rotation
    global GEMINI_KEYS, CURRENT_KEY_INDEX, CONSECUTIVE_FAILED_SWITCHES, LAST_SUCCESSFUL_KEY_INDEX
    GEMINI_KEYS = gemini_keys
    CURRENT_KEY_INDEX = 0
    CONSECUTIVE_FAILED_SWITCHES = 0
    LAST_SUCCESSFUL_KEY_INDEX = -1
    # Configure with first key
    genai.configure(api_key=GEMINI_KEYS[CURRENT_KEY_INDEX])
    print(f"🔑 Initialized with {len(GEMINI_KEYS)} API key(s)")
    return genai


def _initialize_openai_client():
    """Initialize OpenAI API client"""
    openai_keys = get_key('openai')
    # Use first key for OpenAI
    return OpenAI(api_key=openai_keys[0])
    return OpenAI(api_key=openai_key)


def _extract_movie_data(row):
    """Extract movie data from dataframe row"""
    title = row['title']
    all_reviews = row['all_reviews']
    reviews = [r.strip() for r in all_reviews.split('|') if r.strip()]
    return title, reviews


def _log_movie_header(idx, total_movies, title, num_reviews, log_message):
    """Log movie processing header"""
    log_message(f"\n{'='*80}", show_progress=True)
    log_message(f"🎬 Movie {idx + 1}/{total_movies}: {title}", show_progress=True)
    log_message(f"{'='*80}", show_progress=True)
    log_message(f"�� Total reviews: {num_reviews}", show_progress=True)
    
    log_message(f"\n{'='*80}")
    log_message(f"Movie {idx + 1}: {title}")
    log_message(f"{'='*80}")
    log_message(f"Total reviews: {num_reviews}\n")


def _extract_sentiment_from_review(
    review: str, 
    review_idx: int, 
    total_reviews: int,
    log_message,
    verbose: bool,
    max_retries: int = 3
) -> dict:
    """Extract sentiment from a single review using Gemini API with retry and key rotation logic"""
    keys_tried = 0
    global CURRENT_KEY_INDEX, CONSECUTIVE_FAILED_SWITCHES, LAST_SUCCESSFUL_KEY_INDEX
    
    # Try all available keys
    while keys_tried < len(GEMINI_KEYS):
        retry_count = 0
        
        # Try up to max_retries times with current key
        while retry_count < max_retries:
            try:
                log_message(f"  Processing review {review_idx}/{total_reviews}...")
                log_message(f"     Review: {review}\n")
                
                extracted_json = _call_gemini_api(review)
                
                _display_extracted_sentiments(extracted_json, log_message)
                
                # Success! Reset consecutive failed switches counter
                CONSECUTIVE_FAILED_SWITCHES = 0
                LAST_SUCCESSFUL_KEY_INDEX = CURRENT_KEY_INDEX
                
                return extracted_json
                
            except Exception as e:
                error_str = str(e)
                
                if __is_rate_limit_error(error_str):
                    retry_count += 1
                    if retry_count < max_retries:
                        # Still have retries left with current key
                        __handle_rate_limit_retry(retry_count, max_retries, verbose, log_message)
                        continue
                    else:
                        # Exhausted all retries with current key, try next key
                        log_message(f"Max retries ({max_retries}) reached with current key. Rotating to next key...", level='WARNING')
                        if __rotate_gemini_key(log_message):
                            keys_tried += 1
                            break  # Break inner loop to retry with new key
                        else:
                            # No more keys to try
                            __log_all_keys_exhausted(review_idx, error_str, log_message)
                            return None
                else:
                    __log_processing_error(review_idx, error_str, log_message)
                    return None
    
    # All keys exhausted
    __log_all_keys_exhausted(review_idx, "All API keys exhausted", log_message)
    return None


def _create_extracted_row(movie_id, title, review_idx, extracted_json):
    """Create a row for extracted data"""
    return {
        'title': title,
        'review_idx': review_idx,
        'director': '|'.join(extracted_json.get('director', [])) if extracted_json.get('director', []) else '',
        'actors': '|'.join(extracted_json.get('actors', [])) if extracted_json.get('actors', []) else '',
        'content': '|'.join(extracted_json.get('content', [])) if extracted_json.get('content', []) else '',
        'other': '|'.join(extracted_json.get('other', [])) if extracted_json.get('other', []) else ''
    }


def _validate_random_extraction(movie_extractions, openai_client, log_message):
    """Select and validate one random extraction"""
    import random
    
    selected_extraction = random.choice(movie_extractions)
    selected_idx = selected_extraction['review_idx']
    review_text = selected_extraction['review']
    extracted_json = selected_extraction['extracted_json']
    
    return _validate_extraction(review_text, extracted_json, selected_idx, openai_client, log_message)


def _validate_extraction(
    review_text: str,
    extracted_json: dict,
    selected_idx: int,
    openai_client,
    log_message
) -> dict:
    """Validate extraction using OpenAI API"""
    log_message(f"  🔍 Validating review #{selected_idx + 1}...", show_progress=True)
    log_message(f"\n  Validating review #{selected_idx + 1} with OpenAI...")
    
    try:
        validation_result = _call_openai_api(review_text, extracted_json, openai_client)
        is_correct = validation_result.startswith('yes')
        
        result_text = '✅ CORRECT' if is_correct else '❌ NOT CORRECT'
        
        log_message(f"     {result_text}\n")
        log_message(f"  {result_text}", show_progress=True)
        
        return {
            'review': review_text,
            'result': 'correct' if is_correct else 'not correct'
        }
        
    except Exception as e:
        log_message(f"     ⚠️  Error validating extraction: {str(e)}\n", level='error')
        log_message(f"  ⚠️  Validation error", show_progress=True)
        return {
            'review': review_text,
            'result': 'error'
        }


def _display_summary(
    extracted_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    validation: bool,
    log_message,
    logger,
    log_file: str,
    verbose: bool
):
    """Display processing summary"""
    _display_summary_progress(extracted_df, validation_df, validation, log_message)
    _display_summary_detailed(extracted_df, validation_df, validation, log_message)
    
    if not verbose and logger:
        log_message(f"\n=== Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"\n💾 Detailed logs saved to: {log_file}")


def _determine_starting_point(movie_review_df, resume_mode, output_file, log_file, verbose):
    """Determine starting point for incremental processing"""
    start_index = -1
    write_header = True
    
    if resume_mode and os.path.exists(output_file):
        start_index = _find_last_processed_index(movie_review_df, output_file)
        write_header = False
        
        if start_index >= 0:
            _log_to_file(log_file, f"Resuming from index {start_index + 1}")
            if verbose:
                print(f"📍 Resuming from review index {start_index + 1}")
        else:
            _log_to_file(log_file, "Could not find last index, starting from beginning")
            if verbose:
                print(f"⚠️  Could not determine last processed review, starting from beginning")
    else:
        _clear_old_files(output_file, log_file)
        _log_to_file(log_file, f"Starting new processing session")
        if verbose:
            print(f"🚀 Starting new processing session")
    
    return start_index, write_header


def _log_movie_progress(movie_idx, total_movies, title, num_reviews, log_file, verbose):
    """Log movie processing progress"""
    # Always show progress summary
    print(f"\n🎬 Movie {movie_idx + 1}/{total_movies}: {title} ({num_reviews} reviews)")
    
    # Show detailed separator only in verbose mode
    if verbose:
        print(f"{'='*80}")
    
    _log_to_file(log_file, f"Processing movie {movie_idx + 1}/{total_movies}: {title} ({num_reviews} reviews)")


def _process_and_save_review(
    review, review_idx, total_reviews, movie_id, title,
    output_file, write_header, log_file, verbose
):
    """Process single review and save to CSV"""
    # Always show review progress
    print(f"  ⏳ Processing review {review_idx + 1}/{total_reviews}...", end='', flush=True)
    
    # Create a proper log_message function that writes to file
    def log_message(msg, level='INFO', show_progress=False):
        """Log message to file and optionally to console"""
        if show_progress:
            print(msg)
        _log_to_file(log_file, msg, level=level)
    
    try:
        extracted_json = _extract_sentiment_from_review(
            review,
            review_idx + 1,
            total_reviews,
            log_message,
            verbose=False,
            max_retries=3
        )
        
        if extracted_json is None:
            extracted_json = {'director': [], 'actors': [], 'content': [], 'other': []}
            _log_to_file(log_file, f"Failed to extract review {review_idx + 1} for movie {movie_id}", level='ERROR')
        
        _append_review_to_csv(output_file, movie_id, title, review_idx, extracted_json, write_header)
        
        # Always show completion status
        print(f" ✅")
        
        # Show detailed info only in verbose mode
        if verbose:
            print(f"  Saved review {review_idx + 1}/{total_reviews}")
        
        # Wait 1s after successful review to avoid rate limit
        time.sleep(1)
        
        return False
        
    except Exception as e:
        _log_to_file(log_file, f"Error processing review {review_idx + 1} for movie {movie_id}: {str(e)}", level='ERROR')
        
        _append_review_to_csv(
            output_file, movie_id, title, review_idx,
            {'director': [], 'actors': [], 'content': [], 'other': []},
            write_header
        )
        
        # Always show error status
        print(f" ❌")
        
        # Show detailed error only in verbose mode
        if verbose:
            print(f"  Error on review {review_idx + 1}/{total_reviews}: {str(e)}")
        
        return False


def _finalize_movie_processing(movie_idx, title, total_reviews_processed, log_file, verbose):
    """Finalize processing for a movie"""
    # Always show completion message
    print(f"  💾 Completed movie: {title} (Total reviews processed: {total_reviews_processed})")
    
    _log_to_file(log_file, f"Completed movie {movie_idx + 1}: {title} (Total reviews so far: {total_reviews_processed})")


def _display_incremental_summary(total_reviews_processed, output_file, log_file, verbose):
    """Display incremental processing summary"""
    # Always show completion summary
    print(f"\n{'='*80}")
    print(f"✨ Processing Complete!")
    print(f"📊 Total reviews processed: {total_reviews_processed}")
    print(f"💾 Output saved to: {output_file}")
    
    # Show detailed info only in verbose mode
    if verbose:
        print(f"📋 Log saved to: {log_file}")
    
    print(f"{'='*80}\n")
    
    _log_to_file(log_file, f"Processing completed. Total reviews: {total_reviews_processed}")


def _display_extracted_sentiments(extracted_json: dict, log_message):
    """Display extracted sentiments"""
    log_message(f"     ✓ Extracted sentiments:")
    if extracted_json.get('director'):
        log_message(f"       • Director: {', '.join(extracted_json['director'])}")
    if extracted_json.get('actors'):
        log_message(f"       • Actors: {', '.join(extracted_json['actors'])}")
    if extracted_json.get('content'):
        log_message(f"       • Content: {', '.join(extracted_json['content'])}")
    if extracted_json.get('other'):
        log_message(f"       • Other: {', '.join(extracted_json['other'])}")
    if not any([extracted_json.get(cat) for cat in ['director', 'actors', 'content', 'other']]):
        log_message(f"       • (No sentiments extracted)")
    log_message("")


def _find_last_processed_index(df: pd.DataFrame, csv_file: str) -> int:
    """Find the index of the last processed review in the dataframe"""
    if not os.path.exists(csv_file):
        return -1
    
    try:
        existing_df = pd.read_csv(csv_file)
        if len(existing_df) == 0:
            return -1
        
        last_row = existing_df.iloc[-1]
        last_title = last_row['title']
        last_review_idx = last_row['review_idx']
        
        cumulative_index = 0
        for idx, row in df.iterrows():
            title = row['title']
            all_reviews = row['all_reviews']
            reviews = [r.strip() for r in all_reviews.split('|') if r.strip()]
            
            if title == last_title:
                return cumulative_index + last_review_idx
            
            cumulative_index += len(reviews)
        
        return -1
        
    except Exception as e:
        print(f"Warning: Could not read last processed index: {e}")
        return -1


def _append_review_to_csv(
    csv_file: str,
    movie_id: str,
    title: str,
    review_idx: int,
    extracted_json: dict,
    write_header: bool = False
):
    """Append a single processed review to CSV file"""
    import csv
    
    mode = 'w' if write_header else 'a'
    
    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow(['title', 'review_idx', 'director', 'actors', 'content', 'other'])
        
        row = [
            title,
            review_idx,
            '|'.join(extracted_json.get('director', [])) if extracted_json.get('director', []) else '',
            '|'.join(extracted_json.get('actors', [])) if extracted_json.get('actors', []) else '',
            '|'.join(extracted_json.get('content', [])) if extracted_json.get('content', []) else '',
            '|'.join(extracted_json.get('other', [])) if extracted_json.get('other', []) else ''
        ]
        
        writer.writerow(row)


def _log_to_file(log_file: str, message: str, level: str = 'INFO'):
    """Append a log message to log file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp} | {level} | {message}\n")


def _call_gemini_api(review):
    """Call Gemini API to extract sentiments"""
    extraction_prompt = __get_extract_prompt(review)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(extraction_prompt)
    response_text = response.text.strip()
    return _parse_gemini_response(response_text)


def _call_openai_api(review_text, extracted_json, openai_client):
    """Call OpenAI API for validation"""
    validation_prompt = __get_validation_prompt(review_text, json.dumps(extracted_json))
    
    validation_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": validation_prompt}],
        temperature=0.3,
        max_tokens=100
    )
    
    return validation_response.choices[0].message.content.strip().lower()


def _parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini API response and extract JSON"""
    if '```json' in response_text:
        json_str = response_text.split('```json')[1].split('```')[0].strip()
    elif '```' in response_text:
        json_str = response_text.split('```')[1].split('```')[0].strip()
    else:
        json_str = response_text
    
    return json.loads(json_str)


def _display_summary_progress(extracted_df, validation_df, validation, log_message):
    """Display summary for progress output"""
    log_message(f"\n{'='*80}", show_progress=True)
    log_message(f"✨ Processing Complete!", show_progress=True)
    log_message(f"{'='*80}", show_progress=True)
    log_message(f"�� Summary:", show_progress=True)
    log_message(f"   • Total reviews processed: {len(extracted_df)}", show_progress=True)
    
    if validation:
        log_message(f"   • Total validations: {len(validation_df)}", show_progress=True)
        log_message(f"   • Correct extractions: {len(validation_df[validation_df['result'] == 'correct'])}", show_progress=True)
        log_message(f"   • Incorrect extractions: {len(validation_df[validation_df['result'] == 'not correct'])}", show_progress=True)
        log_message(f"   • Validation errors: {len(validation_df[validation_df['result'] == 'error'])}", show_progress=True)
    else:
        log_message(f"   • Validation: Skipped (validation=False)", show_progress=True)
    
    log_message(f"{'='*80}\n", show_progress=True)


def _display_summary_detailed(extracted_df, validation_df, validation, log_message):
    """Display detailed summary for log file"""
    log_message(f"\n{'='*80}")
    log_message(f"Processing Complete!")
    log_message(f"{'='*80}")
    log_message(f"Summary:")
    log_message(f"   Total reviews processed: {len(extracted_df)}")
    
    if validation:
        log_message(f"   Total validations: {len(validation_df)}")
        log_message(f"   Correct extractions: {len(validation_df[validation_df['result'] == 'correct'])}")
        log_message(f"   Incorrect extractions: {len(validation_df[validation_df['result'] == 'not correct'])}")
        log_message(f"   Validation errors: {len(validation_df[validation_df['result'] == 'error'])}")
    else:
        log_message(f"   Validation: Skipped")
    
    log_message(f"{'='*80}\n")


def _clear_old_files(output_file, log_file):
    """Clear old output and log files"""
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(log_file):
        os.remove(log_file)


# ============================================================================
# LEVEL 2 FUNCTIONS - Called by Level 1
# ============================================================================

def __get_extract_prompt(review):
    """
    Generate the extraction prompt for sentiment analysis
    
    Args:
        review (str): The movie review text
    
    Returns:
        str: Formatted prompt for sentiment extraction
    """
    prompt = f"""ROLE: You are a professional sentiment and aspect-based opinion mining assistant specializing in analyzing movie reviews.
ACTION: Analyze the provided movie review to extract sentiment expressions and classify them into appropriate film-related aspects.
TASK:
Extract sentiment expressions, defined as:
- A single sentiment ADVERB or ADJECTIVE, optionally preceded by a modifier:
- Negations (e.g., not, never)
- Intensifiers (e.g., very, extremely, too, so)
- Do not extract phrases longer than 2 words.
- Keep the modifier together with the ADVERB or ADJECTIVE (e.g., not good, very bad), and treat it as one sentiment expression.
- Aspects to classify into:
	+) director: Sentiment expressions related to the director's influence on the film, including decisions about storytelling, style, tone, pacing, and overall direction.
	+) actors: Sentiment expressions related to the performances of the cast, including how characters are portrayed, expressed, and received by the audience.
	+) content: Sentiment expressions related to any aspect of the film's substance and presentation — including narrative, visuals, sound, emotional impact, and production elements. Do not limit this only to technical components; anything reflecting the content or experience of the film belongs here.
	+) other: Any sentiment expression not clearly attributable to the three categories above, such as marketing, hype, expectations, comparisons, or general opinions.

INSTRUCTIONS
- Only extract sentiment expressions (single-word ADVERB or ADJECTIVE or modifier + ADVERB or ADJECTIVE).
- Do not include neutral or factual terms without sentiment.
- Do not infer or generate new expressions — only extract from the actual review.
- Classify each sentiment expression under EXACTLY ONE aspect with CLEAR evidence, if double classified, mark as "other".

EXAMPLES:
1.Review #1: The actors were not convincing, and the dialogue felt forced.
Extract:
- actors: not convincing
- content: forced
2. Review #2: A visually stunning movie with breathtaking shots and very immersive sound design.
Extract:
- content: stunning, breathtaking, very immersive
3.Review #3: The director's choices were bold but sometimes confusing.
Extract:
- director: bold, confusing
4.Review #4: Although the film had a slow start, the lead actor delivered an exceptionally powerful performance.
Extract:
- content: slow
- actors: exceptionally powerful
5.Review: Not bad, but not great either. Just average.
Extract:
- other: not bad, not great, average

OUTPUT FORMAT:
{{
  "director": ["..."],
  "actors": ["..."],
  "content": ["..."],
  "other": ["..."]
}}

Review: "{review}"
"""
    return prompt


def __get_validation_prompt(review, extracted_json):
    """
    Generate the validation prompt for extracted sentiments
    
    Args:
        review (str): The movie review text
        extracted_json (str): JSON string of extracted sentiments
    
    Returns:
        str: Formatted prompt for validation
    """
    prompt = f"""ROLE: You are a validation assistant for aspect-based sentiment extraction in movie reviews.

TASK: Your job is to verify whether all sentiment expressions extracted from a movie review have been correctly classified into their appropriate aspects.
You will be given:
	- The original movie review
	- A JSON object containing extracted sentiment expressions, grouped under four aspects: director, actors, content, and other
Your validation criteria:
	1.Each sentiment expression must appear in the review exactly as written.
	2.Each expression must be assigned to the correct aspect, based on the meaning of the sentence in which it appears.
	- director: Related to creative direction, pacing, tone, vision.
	- actors: Related to cast performance or character portrayal.
	- content: Related to story, visuals, sound, production, or any film experience.
	- other: General opinions not fitting the above.
	3.If even one sentiment expression is misclassified, the output is "No".
	4.If all expressions are correctly assigned, the output is "Yes".
OUTPUT FORMAT:
	- Yes: if all expressions are correctly classified
	- No: if any expression is misclassified

EXAMPLE:
Review #1:
The director's choices were bold but sometimes confusing. The actors delivered very powerful performances. The visuals were stunning.
Extracted JSON:
{{{{
  "director": ["bold", "confusing"],
  "actors": ["very powerful"],
  "content": ["stunning"],
  "other": []
}}}}
Output: Yes
Another Example:

Review #2:
The film looked amazing but the main actor felt flat.
Extracted JSON:
{{{{
  "director": [],
  "actors": ["amazing"],
  "content": ["flat"],
  "other": []
}}}}
Output: No

Review: "{review}"
Extracted JSON: {extracted_json}
"""
    return prompt


def __setup_logger(log_file: str) -> logging.Logger:
    """
    Set up a logger that writes to a file.
    
    Args:
        log_file (str): Path to the log file
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = _create_logger()
    file_handler = _create_file_handler(log_file)
    logger.addHandler(file_handler)
    return logger


def __is_rate_limit_error(error_str):
    """Check if error is a rate limit error"""
    return '429' in error_str or 'quota' in error_str.lower()


def __rotate_gemini_key(log_message):
    """
    Rotate to next Gemini API key
    
    Returns:
        bool: True if rotated to new key, False if all keys exhausted
    """
    global CURRENT_KEY_INDEX, GEMINI_KEYS, CONSECUTIVE_FAILED_SWITCHES, LAST_SUCCESSFUL_KEY_INDEX
    
    if len(GEMINI_KEYS) <= 1:
        return False
    
    old_index = CURRENT_KEY_INDEX
    
    # Move to next key
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(GEMINI_KEYS)
    CONSECUTIVE_FAILED_SWITCHES += 1
    
    # Check if we've tried all keys without success
    # This happens when we've made len(GEMINI_KEYS) consecutive failed switches
    if CONSECUTIVE_FAILED_SWITCHES >= len(GEMINI_KEYS):
        log_message(f"🛑 All {len(GEMINI_KEYS)} API keys have been tried without success. All keys appear to be rate limited. Stopping processing.", level='ERROR')
        print(f"\n🛑 CRITICAL: All {len(GEMINI_KEYS)} API keys exhausted without any successful requests.")
        print(f"   Consecutive failed switches: {CONSECUTIVE_FAILED_SWITCHES}")
        print(f"   Last successful key was #{LAST_SUCCESSFUL_KEY_INDEX + 1 if LAST_SUCCESSFUL_KEY_INDEX >= 0 else 'None'}")
        print(f"   All keys appear to be rate limited. Stopping program.")
        raise SystemExit(f"Program stopped: All {len(GEMINI_KEYS)} API keys are rate limited. No successful requests after trying all keys.")
    
    # Reconfigure Gemini with new key
    genai.configure(api_key=GEMINI_KEYS[CURRENT_KEY_INDEX])
    
    # Log the rotation
    log_message(f"🔄 API Key Rotation: Key #{old_index + 1} → Key #{CURRENT_KEY_INDEX + 1} (Total keys: {len(GEMINI_KEYS)}, Consecutive failures: {CONSECUTIVE_FAILED_SWITCHES}/{len(GEMINI_KEYS)})", level='INFO')
    
    # Wait 15s after switching keys to let the previous key cool down
    wait_time = 15
    print(f"  🔄 Switched to API key {CURRENT_KEY_INDEX + 1}/{len(GEMINI_KEYS)} (Failed switches: {CONSECUTIVE_FAILED_SWITCHES}/{len(GEMINI_KEYS)})")
    print(f"  ⏸️  Waiting {wait_time}s after key switch...")
    log_message(f"Waiting {wait_time}s after key switch to let previous key cool down", level='INFO')
    
    for i in range(wait_time):
        if i % 5 == 0:  # Show countdown every 5 seconds
            print(f"        ⏳ {wait_time - i}s remaining...", end="\r", flush=True)
        time.sleep(1)
    print()  # New line after countdown
    
    return True


def __log_all_keys_exhausted(review_idx, error_str, log_message):
    """Log when all API keys have been exhausted"""
    log_message(f"     ✗ All API keys exhausted for review {review_idx}: {error_str}\n", level='error')
    log_message(f"  ✗ All keys exhausted - Skipping review {review_idx}", show_progress=True)


def __handle_rate_limit_retry(retry_count, max_retries, verbose, log_message):
    """Handle rate limit retry logic with fixed 15s wait time"""
    wait_time = 15
    # Only log to file, not to console
    log_message(f"Rate limit encountered! Waiting {wait_time}s before retry... (Attempt {retry_count}/{max_retries})", level='WARNING')
    # Show progress to console only
    print(f"  ⏸️  Rate limit - Retrying {retry_count}/{max_retries} (waiting {wait_time}s)...")
    for i in range(wait_time):
        if verbose or i % 5 == 0:  # Show countdown every 5 seconds
            print(f"        ⏳ {wait_time - i}s remaining...", end="\r", flush=True)
        time.sleep(1)
    print()  # New line after countdown


def __log_max_retries_reached(review_idx, error_str, log_message):
    """Log when max retries reached"""
    log_message(f"     ✗ Max retries reached. Skipping review {review_idx}: {error_str}\n", level='error')
    log_message(f"  ✗ Skipping review {review_idx}", show_progress=True)


def __log_processing_error(review_idx, error_str, log_message):
    """Log processing error"""
    log_message(f"     ✗ Error processing review {review_idx}: {error_str}\n", level='error')
    log_message(f"  ✗ Error on review {review_idx}", show_progress=True)
